from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

from ..store import timescale

log = logging.getLogger("rlc.bandit")


@dataclass(frozen=True)
class Arm:
    idx: int
    batch_size: int
    delay_ms: int


@dataclass
class ArmState:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences


class ConstrainedTS:
    """Gaussian Thompson Sampling with persisted state per provider/regime."""

    def __init__(self, provider: str, regime: str, arms: Iterable[Arm]) -> None:
        self.provider = provider
        self.regime = regime
        self.arms = list(arms)
        self.state: Dict[int, ArmState] = {arm.idx: ArmState() for arm in self.arms}

    async def load(self) -> None:
        assert timescale.ts_pool and timescale.ts_pool.pool, "Timescale pool not initialised"
        async with timescale.ts_pool.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT arm_id, n, mean_reward, m2
                FROM rlc_bandit_state
                WHERE provider = $1 AND regime = $2
                """,
                self.provider,
                self.regime,
            )
        for row in rows:
            self.state[row["arm_id"]] = ArmState(
                n=row["n"],
                mean=row["mean_reward"],
                m2=row["m2"],
            )

    async def persist(self) -> None:
        assert timescale.ts_pool and timescale.ts_pool.pool, "Timescale pool not initialised"
        async with timescale.ts_pool.pool.acquire() as conn:
            for arm in self.arms:
                s = self.state[arm.idx]
                await conn.execute(
                    """
                    INSERT INTO rlc_bandit_state(provider, regime, arm_id, batch_size, delay_ms, n, mean_reward, m2, updated_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8, now())
                    ON CONFLICT (provider, regime, arm_id)
                    DO UPDATE SET
                      batch_size = EXCLUDED.batch_size,
                      delay_ms = EXCLUDED.delay_ms,
                      n = EXCLUDED.n,
                      mean_reward = EXCLUDED.mean_reward,
                      m2 = EXCLUDED.m2,
                      updated_at = now()
                    """,
                    self.provider,
                    self.regime,
                    arm.idx,
                    arm.batch_size,
                    arm.delay_ms,
                    s.n,
                    s.mean,
                    s.m2,
                )

    def select(self, feasible_arm_ids: List[int]) -> Arm:
        best_val = float("-inf")
        best_arm: Arm | None = None

        for arm in self.arms:
            if arm.idx not in feasible_arm_ids:
                continue
            state = self.state[arm.idx]
            if state.n <= 1:
                sample = random.normalvariate(0, 1)
            else:
                variance = (state.m2 / (state.n - 1)) if state.n > 1 else 1.0
                std = math.sqrt(max(variance, 1e-3)) / math.sqrt(state.n)
                sample = random.normalvariate(state.mean, std)
            if sample > best_val or best_arm is None:
                best_val = sample
                best_arm = arm

        return best_arm or self.arms[0]

    def update(self, arm_id: int, reward: float) -> None:
        state = self.state[arm_id]
        state.n += 1
        delta = reward - state.mean
        state.mean += delta / state.n
        state.m2 += delta * (reward - state.mean)
