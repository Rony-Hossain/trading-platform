from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..bandit.cts import Arm, ConstrainedTS
from ..core.config import settings
from ..features.aggregator import FeatureAggregator
from ..models.onnx_runtime import RlcPredictor
from ..obs.metrics import (
    bandit_arm_selection_total,
    rlc_policy_updates,
)
from ..policy.publisher import PolicyPublisher
from ..policy.synthesizer import PolicySynthesizer
from ..schemas.policy import BatchHints, PolicyBundle, TierQuota, TokenPolicy
from ..store import timescale

log = logging.getLogger("rlc.loop")


@dataclass
class ProviderState:
    last_rps: float


class PolicyLoop:
    """Main loop combining telemetry, inference, bandit selection, and policy publishing."""

    def __init__(
        self,
        predictor: RlcPredictor,
        synthesizer: PolicySynthesizer,
        publisher: PolicyPublisher,
        providers: List[str],
    ) -> None:
        self.predictor = predictor
        self.synthesizer = synthesizer
        self.publisher = publisher
        self.providers = providers
        self.aggregator = FeatureAggregator()
        self.state: Dict[str, ProviderState] = {
            provider: ProviderState(last_rps=settings.default_rps) for provider in providers
        }
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if not self._task:
            self._task = asyncio.create_task(self._run(), name="policy-loop")
            log.info("Policy loop started (providers=%s interval=%ss)", self.providers, settings.loop_interval_s)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    def _decide_target_rps(self, pred_p95_ms: float, last_rps: float) -> float:
        slo = settings.p95_slo_ms
        rps = last_rps
        if pred_p95_ms > 1.05 * slo:
            rps *= 0.8
        elif pred_p95_ms < 0.8 * slo:
            rps *= 1.15
        return min(max(self.synthesizer.min_refill, rps), self.synthesizer.max_refill)

    async def _tick_provider(self, provider: str) -> None:
        snap, feats = await self.aggregator.snapshot_and_features(provider)
        X = np.asarray(feats, dtype=np.float32)[None, :]

        lat, err_prob = self.predictor.predict(X)
        pred_p95 = float(lat[0])
        pred_err = float(err_prob[0])

        state = self.state[provider]
        target_rps = self._decide_target_rps(pred_p95, state.last_rps)
        state.last_rps = target_rps

        budget_breach = snap.est_cost_usd_per_min > settings.budget_usd_per_min

        feasible = [
            idx for idx, _ in enumerate(settings.bandit_arms) if pred_err <= settings.error_constraint
        ] or [0]

        batch_hints: Optional[BatchHints] = None
        arm: Optional[Arm] = None
        if settings.bandit_enabled:
            arms = [Arm(idx, *arm_cfg) for idx, arm_cfg in enumerate(settings.bandit_arms)]
            bandit = ConstrainedTS(provider, snap.regime, arms)
            await bandit.load()
            arm = bandit.select(feasible)
            reward = -(0.6 * pred_p95 + 0.3 * (pred_err * 100.0) + 0.1 * snap.est_cost_usd_per_min)
            bandit.update(arm.idx, reward)
            await bandit.persist()
            arm_label = f"{arm.batch_size}x{arm.delay_ms}"
            bandit_arm_selection_total.labels(provider=provider, arm=arm_label).inc()
            batch_hints = BatchHints(batch_size=arm.batch_size, inter_batch_delay_ms=arm.delay_ms)

        quotas = self.synthesizer.quotas_from_budget(
            t0_demand=snap.t0_symbols,
            t1_demand=snap.t1_symbols,
            budget_breach=budget_breach,
            min_t0_floor=settings.min_t0_floor,
        )

        token_params = self.synthesizer.token_bucket_from_rate(
            target_rps=target_rps,
            desired_burst_seconds=1.5 if snap.regime in ("mid", "after") else 2.0,
            jitter_fraction=0.1,
            ttl_s=60,
        )

        bundle = PolicyBundle(
            provider=provider,
            token_policy=TokenPolicy(
                refill_rate=token_params.refill_rate,
                burst=token_params.burst,
                jitter_ms=token_params.jitter_ms,
                ttl_s=token_params.ttl_s,
            ),
            tier_quota=TierQuota(
                t0_max=quotas.t0_max,
                t1_max=quotas.t1_max,
                t2_mode=quotas.t2_mode,
            ),
            mode=settings.mode,
            batch_hints=batch_hints,
        )

        await self.publisher.publish_policy(bundle)
        rlc_policy_updates.labels(settings.mode, provider).inc()

        assert timescale.ts_pool and timescale.ts_pool.pool, "Timescale pool not initialised"
        async with timescale.ts_pool.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO rlc_decisions(
                    ts, provider, regime, mode, arm_id, batch_size, delay_ms,
                    target_rps, token_refill_rate, token_burst, token_jitter_ms, token_ttl_s
                )
                VALUES (now(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                provider,
                snap.regime,
                settings.mode,
                arm.idx if arm else None,
                batch_hints.batch_size if batch_hints else None,
                batch_hints.inter_batch_delay_ms if batch_hints else None,
                float(target_rps),
                float(token_params.refill_rate),
                int(token_params.burst),
                int(token_params.jitter_ms),
                int(token_params.ttl_s),
            )

        log.info(
            "policy %s: rps=%.1f p95=%.0fms err=%.3f regime=%s budget_breach=%s arm=%s",
            provider,
            target_rps,
            pred_p95,
            pred_err,
            snap.regime,
            budget_breach,
            f"{arm.batch_size}x{arm.delay_ms}" if arm else "disabled",
        )

    async def _run(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.gather(*(self._tick_provider(p) for p in self.providers))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=settings.loop_interval_s)
                except asyncio.TimeoutError:
                    continue
        except Exception:
            log.exception("Policy loop crashed")
            raise
