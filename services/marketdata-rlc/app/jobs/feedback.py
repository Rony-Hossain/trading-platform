from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List

from ..bandit.cts import Arm, ConstrainedTS
from ..core.config import settings
from ..obs.metrics import (
    rlc_error_constraint_violations_total,
    rlc_observed_error_rate,
)
from ..store import timescale

log = logging.getLogger("rlc.feedback")


class FeedbackJob:
    """Periodic job that updates bandit posteriors using observed metrics."""

    def __init__(self, providers: List[str]) -> None:
        self.providers = providers
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not settings.feedback_enabled:
            log.info("Feedback job disabled")
            return
        if not self._task:
            self._task = asyncio.create_task(self._run(), name="feedback-job")
            log.info("Feedback job started")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick_all()
            except Exception:
                log.exception("Feedback tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=settings.feedback_interval_s)
            except asyncio.TimeoutError:
                continue

    async def _tick_all(self) -> None:
        assert timescale.ts_pool and timescale.ts_pool.pool, "Timescale pool not initialised"
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(minutes=settings.feedback_lookback_min)

        async with timescale.ts_pool.pool.acquire() as conn:
            for provider in self.providers:
                decision = await conn.fetchrow(
                    """
                    SELECT ts, regime, arm_id, batch_size, delay_ms
                    FROM rlc_decisions
                    WHERE provider = $1 AND ts >= $2
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    provider,
                    lookback,
                )
                if not decision or decision["arm_id"] is None:
                    continue

                start_ts = decision["ts"]
                end_ts = start_ts + timedelta(minutes=1)

                obs = await conn.fetchrow(
                    """
                    WITH calls AS (
                      SELECT latency_ms,
                             CASE WHEN status >= 400 THEN 1 ELSE 0 END AS err
                      FROM api_calls
                      WHERE provider = $1 AND ts >= $2 AND ts < $3
                    ),
                    stats AS (
                      SELECT
                        COALESCE(percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms), 220) AS p95,
                        COALESCE(AVG(err)::float, 0.01) AS err_rate
                      FROM calls
                    ),
                    cost AS (
                      SELECT COALESCE(SUM(cost_usd), 0.35) AS cost_pm
                      FROM cost_ledger
                      WHERE provider = $1 AND ts >= $2 AND ts < $3
                    )
                    SELECT stats.p95, stats.err_rate, cost.cost_pm
                    FROM stats, cost
                    """,
                    provider,
                    start_ts,
                    end_ts,
                )

                obs_p95 = float(obs["p95"])
                obs_err = float(obs["err_rate"])
                obs_cost = float(obs["cost_pm"])

                arms = [Arm(idx, *cfg) for idx, cfg in enumerate(settings.bandit_arms)]
                bandit = ConstrainedTS(provider, decision["regime"], arms)
                await bandit.load()

                reward = -(0.6 * obs_p95 + 0.3 * (obs_err * 100.0) + 0.1 * obs_cost)
                bandit.update(decision["arm_id"], reward)
                await bandit.persist()

                rlc_observed_error_rate.labels(provider=provider).set(obs_err)
                if obs_err > settings.error_constraint:
                    rlc_error_constraint_violations_total.labels(provider=provider).inc()

                log.info(
                    "feedback %s: arm=%s p95=%.0f err=%.3f cost=%.3f reward=%.2f",
                    provider,
                    decision["arm_id"],
                    obs_p95,
                    obs_err,
                    obs_cost,
                    reward,
                )
