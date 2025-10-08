import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Tuple

from ..store import timescale


@dataclass
class Snapshot:
    p50_ms: float
    p95_ms: float
    err_rate: float
    tokens_remaining: float
    t0_symbols: int
    t1_symbols: int
    est_cost_usd_per_min: float
    regime: str


class FeatureAggregator:
    """Fetches recent telemetry from TimescaleDB and produces feature vectors."""

    @staticmethod
    def _regime(now: datetime) -> str:
        hhmm = now.hour * 100 + now.minute
        if 930 <= hhmm < 1100:
            return "open"
        if 1100 <= hhmm < 1500:
            return "mid"
        if 1500 <= hhmm <= 1600:
            return "close"
        return "after"

    async def snapshot_and_features(self, provider: str) -> Tuple[Snapshot, list[float]]:
        assert timescale.ts_pool and timescale.ts_pool.pool, "Timescale pool not initialised"
        now = datetime.now(timezone.utc)
        ten_min_ago = now - timedelta(minutes=10)
        one_min_ago = now - timedelta(minutes=1)

        async with timescale.ts_pool.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                WITH calls AS (
                  SELECT latency_ms,
                         CASE WHEN status >= 400 THEN 1 ELSE 0 END AS err
                  FROM api_calls
                  WHERE provider = $1 AND ts >= $2
                ),
                stats AS (
                  SELECT
                    COALESCE(percentile_disc(0.5) WITHIN GROUP (ORDER BY latency_ms), 120) AS p50,
                    COALESCE(percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms), 220) AS p95,
                    COALESCE(AVG(err)::float, 0.01) AS err_rate
                  FROM calls
                ),
                cost AS (
                  SELECT COALESCE(SUM(cost_usd), 0.35) AS cost_pm
                  FROM cost_ledger
                  WHERE provider = $1 AND ts >= $3
                )
                SELECT stats.p50, stats.p95, stats.err_rate, cost.cost_pm
                FROM stats, cost
                """,
                provider,
                ten_min_ago,
                one_min_ago,
            )

        p50 = float(row["p50"])
        p95 = float(row["p95"])
        err = float(row["err_rate"])
        cost_pm = float(row["cost_pm"])

        tokens_remaining = 300.0
        t0_symbols = 150
        t1_symbols = 900
        regime = self._regime(datetime.now())

        snap = Snapshot(
            p50_ms=p50,
            p95_ms=p95,
            err_rate=err,
            tokens_remaining=tokens_remaining,
            t0_symbols=t0_symbols,
            t1_symbols=t1_symbols,
            est_cost_usd_per_min=cost_pm,
            regime=regime,
        )

        minute = datetime.now().hour * 60 + datetime.now().minute
        rad = 2 * math.pi * (minute / (24 * 60))
        mod_sin = math.sin(rad)
        mod_cos = math.cos(rad)

        r_open = 1.0 if regime == "open" else 0.0
        r_mid = 1.0 if regime == "mid" else 0.0
        r_close = 1.0 if regime == "close" else 0.0
        r_after = 1.0 if regime == "after" else 0.0

        feats = [
            mod_sin,
            mod_cos,
            r_open,
            r_mid,
            r_close,
            r_after,
            p50,
            p95,
            err,
            tokens_remaining,
            float(t0_symbols),
            float(t1_symbols),
        ]

        return snap, feats
