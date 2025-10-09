import asyncpg
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.config import get_settings, settings
from ..observability.metrics import INGESTION_LAG, WRITE_BATCH_LATENCY

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for storing and retrieving historical and macro data."""

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=5,
                max_size=settings.database_pool_size,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
            )
            logger.info("Database connection pool initialized")
        except Exception as exc:  # pragma: no cover - connectivity failure
            logger.error("Failed to initialize database pool: %s", exc)
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Yield a database connection from the pool."""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.pool.acquire() as conn:
            yield conn

    async def store_candle_data(self, symbol: str, candles: List[Dict]) -> int:
        """Store historical OHLCV candles."""
        if not candles:
            return 0

        try:
            normalized = []
            now = datetime.utcnow()
            for candle in candles:
                ts = self._coerce_timestamp(candle["timestamp"])
                normalized.append(
                    {
                        "symbol": symbol.upper(),
                        "ts": ts,
                        "interval": "1d",
                        "o": float(candle["open"]),
                        "h": float(candle["high"]),
                        "l": float(candle["low"]),
                        "c": float(candle["close"]),
                        "v": float(candle["volume"]),
                        "status": "final",
                        "as_of": now,
                    }
                )

            if not normalized:
                return 0

            settings = get_settings()
            inserted = await self.upsert_bars_bulk(
                normalized,
                batch_size=settings.LIVE_BATCH_SIZE,
                provider_used="legacy",
            )
            logger.info("Stored %s candles for %s", inserted, symbol)
            return inserted
        except Exception as exc:
            logger.error("Error storing candle data for %s: %s", symbol, exc)
            raise

    async def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000,
        interval: str = "1d",
    ) -> List[Dict]:
        """Retrieve historical OHLCV candles."""
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT ts, o, h, l, c, v, status, as_of
                    FROM candles_intraday
                    WHERE symbol = $1 AND interval = $2
                """
                params: List = [symbol.upper(), interval]
                param_index = 2

                if start_date:
                    param_index += 1
                    query += f" AND ts >= ${param_index}"
                    params.append(start_date)

                if end_date:
                    param_index += 1
                    query += f" AND ts <= ${param_index}"
                    params.append(end_date)

                query += " ORDER BY ts DESC"

                if limit:
                    param_index += 1
                    query += f" LIMIT ${param_index}"
                    params.append(limit)

                rows = await conn.fetch(query, *params)

                candles: List[Dict] = []
                for row in rows:
                    candles.append(
                        {
                            "timestamp": row["ts"].isoformat(),
                            "open": float(row["o"]),
                            "high": float(row["h"]),
                            "low": float(row["l"]),
                            "close": float(row["c"]),
                            "volume": int(row["v"]),
                            "status": row["status"],
                            "as_of": row["as_of"].isoformat(),
                        }
                    )

                logger.info("Retrieved %s candles for %s", len(candles), symbol)
                return candles
        except Exception as exc:
            logger.error("Error retrieving historical data for %s: %s", symbol, exc)
            raise

    async def get_latest_candle(self, symbol: str, interval: str = "1d") -> Optional[Dict]:
        """Return the most recent candle for a symbol."""
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT ts, o, h, l, c, v, status, as_of
                    FROM candles_intraday
                    WHERE symbol = $1 AND interval = $2
                    ORDER BY ts DESC
                    LIMIT 1
                """
                row = await conn.fetchrow(query, symbol.upper(), interval)
                if not row:
                    return None
                return {
                    "timestamp": row["ts"].isoformat(),
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": int(row["v"]),
                    "status": row["status"],
                    "as_of": row["as_of"].isoformat(),
                }
        except Exception as exc:
            logger.error("Error getting latest candle for %s: %s", symbol, exc)
            return None

    async def get_data_coverage(self, symbol: str, interval: str = "1d") -> Optional[Tuple[datetime, datetime]]:
        """Return earliest and latest timestamps for a symbol."""
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT MIN(ts) AS earliest, MAX(ts) AS latest
                    FROM candles_intraday
                    WHERE symbol = $1 AND interval = $2
                """
                row = await conn.fetchrow(query, symbol.upper(), interval)
                if row and row["earliest"] and row["latest"]:
                    return (row["earliest"], row["latest"])
                return None
        except Exception as exc:
            logger.error("Error getting coverage for %s: %s", symbol, exc)
            return None

    async def upsert_symbol_universe(self, rows: Iterable[Dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0

        sql = """
            INSERT INTO symbol_universe(symbol, exchange, asset_type, adv_21d, mkt_cap, tier, active, provider_symbol_map)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (symbol) DO UPDATE SET
              exchange = EXCLUDED.exchange,
              asset_type = EXCLUDED.asset_type,
              adv_21d = EXCLUDED.adv_21d,
              mkt_cap = EXCLUDED.mkt_cap,
              tier = EXCLUDED.tier,
              active = EXCLUDED.active,
              provider_symbol_map = EXCLUDED.provider_symbol_map,
              updated_at = NOW()
        """

        payload = []
        for row in rows:
            payload.append(
                (
                    row["symbol"].upper(),
                    row.get("exchange", ""),
                    row.get("asset_type", "equity"),
                    row.get("adv_21d"),
                    row.get("mkt_cap"),
                    row.get("tier", "T2"),
                    bool(row.get("active", True)),
                    row.get("provider_symbol_map", {}),
                )
            )

        async with self.get_connection() as conn:
            async with conn.transaction():
                await conn.executemany(sql, payload)

        return len(payload)

    async def record_universe_version(self, version_tag: str, source_meta: Dict[str, Any]) -> None:
        sql = """
            INSERT INTO universe_versions(version_tag, source_meta)
            VALUES ($1, $2)
        """
        async with self.get_connection() as conn:
            await conn.execute(sql, version_tag, json.dumps(source_meta))

    async def get_symbols_by_tier(self, tier: str, limit: int, offset: int = 0) -> List[str]:
        sql = """
            SELECT symbol
            FROM symbol_universe
            WHERE active = TRUE AND tier = $1
            ORDER BY adv_21d DESC NULLS LAST
            LIMIT $2 OFFSET $3
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(sql, tier, limit, offset)
        return [row["symbol"] for row in rows]

    async def get_cursor(self, symbol: str, interval: str, source: str) -> Optional[datetime]:
        sql = """
            SELECT last_ts
            FROM ingestion_cursor
            WHERE symbol = $1 AND interval = $2 AND source = $3
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(sql, symbol.upper(), interval, source)
        return row["last_ts"] if row else None

    async def update_cursor(
        self,
        symbol: str,
        interval: str,
        source: str,
        last_ts: datetime,
        status: str = "ok",
    ) -> None:
        sql = """
            INSERT INTO ingestion_cursor(symbol, interval, source, last_ts, last_status, updated_at)
            VALUES ($1,$2,$3,$4,$5,NOW())
            ON CONFLICT (symbol, interval, source)
            DO UPDATE SET last_ts = EXCLUDED.last_ts,
                          last_status = EXCLUDED.last_status,
                          updated_at = NOW()
        """
        async with self.get_connection() as conn:
            await conn.execute(sql, symbol.upper(), interval, source, last_ts, status)

    async def enqueue_backfill(
        self,
        symbol: str,
        interval: str,
        start_ts: datetime,
        end_ts: datetime,
        priority: str,
    ) -> None:
        sql = """
            INSERT INTO backfill_jobs(symbol, interval, start_ts, end_ts, priority, status, attempts, created_at)
            VALUES ($1,$2,$3,$4,$5,'queued',0,NOW())
            ON CONFLICT (symbol, interval, start_ts, end_ts, priority)
            DO NOTHING
        """
        async with self.get_connection() as conn:
            await conn.execute(sql, symbol.upper(), interval, start_ts, end_ts, priority)

    async def lease_backfill(self, priorities: Sequence[str], limit: int) -> List[Dict[str, Any]]:
        sql = """
            UPDATE backfill_jobs
            SET status = 'leased', leased_at = NOW(), attempts = attempts + 1, updated_at = NOW()
            WHERE id IN (
              SELECT id
              FROM backfill_jobs
              WHERE status = 'queued' AND priority = ANY($1::text[])
              ORDER BY
                CASE priority WHEN 'T0' THEN 0 WHEN 'T1' THEN 1 ELSE 2 END,
                created_at
              FOR UPDATE SKIP LOCKED
              LIMIT $2
            )
            RETURNING id, symbol, interval, start_ts, end_ts, priority, attempts
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(sql, list(priorities), limit)

        return [dict(row) for row in rows]

    async def complete_backfill(self, job_id: int, ok: bool, reason: Optional[str] = None) -> None:
        sql = """
            UPDATE backfill_jobs
            SET status = $2,
                last_error = $3,
                updated_at = NOW()
            WHERE id = $1
        """
        status = "done" if ok else "failed"
        async with self.get_connection() as conn:
            await conn.execute(sql, job_id, status, reason or "")

    async def backfill_queue_depth(self, priority: str) -> int:
        sql = """
            SELECT COUNT(1)
            FROM backfill_jobs
            WHERE status = 'queued' AND priority = $1
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(sql, priority)
        return int(row["count"]) if row else 0

    async def upsert_bars_bulk(
        self,
        bars: Iterable[Dict],
        batch_size: int,
        provider_used: str,
    ) -> int:
        bars_list = list(bars)
        if not bars_list:
            return 0

        sql = """
            INSERT INTO candles_intraday(symbol, ts, interval, o, h, l, c, v, provider, as_of, status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (symbol, ts, interval) DO UPDATE SET
              o = EXCLUDED.o,
              h = EXCLUDED.h,
              l = EXCLUDED.l,
              c = EXCLUDED.c,
              v = EXCLUDED.v,
              provider = EXCLUDED.provider,
              as_of = EXCLUDED.as_of,
              status = EXCLUDED.status
        """

        total = 0
        async with self.get_connection() as conn:
            async with conn.transaction():
                chunk: List[Tuple] = []
                for bar in bars_list:
                    chunk.append(
                        (
                            bar["symbol"].upper(),
                            bar["ts"],
                            bar["interval"],
                            float(bar["o"]),
                            float(bar["h"]),
                            float(bar["l"]),
                            float(bar["c"]),
                            float(bar.get("v", 0)),
                            provider_used,
                            bar.get("as_of", datetime.utcnow()),
                            bar.get("status", "final"),
                        )
                    )

                    if len(chunk) >= batch_size:
                        started = perf_counter()
                        await conn.executemany(sql, chunk)
                        elapsed_ms = (perf_counter() - started) * 1000.0
                        WRITE_BATCH_LATENCY.observe(elapsed_ms)
                        total += len(chunk)
                        chunk.clear()

                if chunk:
                    started = perf_counter()
                    await conn.executemany(sql, chunk)
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    WRITE_BATCH_LATENCY.observe(elapsed_ms)
                    total += len(chunk)

        try:
            newest = max(bar["ts"] for bar in bars_list if bar.get("ts"))
            interval = bars_list[0]["interval"]
            lag_seconds = (datetime.utcnow() - newest).total_seconds()
            INGESTION_LAG.labels(interval=interval).set(max(lag_seconds, 0.0))
        except Exception:  # pragma: no cover - defensive
            pass

        return total

    async def cleanup_old_data(self) -> Optional[int]:
        """Prune candles older than the retention policy."""
        if not settings.historical_data_retention_days:
            return None

        cutoff = datetime.now() - timedelta(days=settings.historical_data_retention_days)
        try:
            async with self.get_connection() as conn:
                result = await conn.execute("DELETE FROM candles WHERE ts < $1", cutoff)
                deleted = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                logger.info("Cleaned up %s old candles", deleted)
                return deleted
        except Exception as exc:
            logger.error("Error during candle cleanup: %s", exc)
            raise

    async def store_macro_points(self, factor_key: str, points: List[Dict]) -> int:
        """Upsert macro factor readings."""
        if not points:
            return 0

        try:
            async with self.get_connection() as conn:
                query = """
                    INSERT INTO macro_factors (factor_key, ts, value, source, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (factor_key, ts)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata
                """
                inserted = 0
                async with conn.transaction():
                    for point in points:
                        timestamp = self._coerce_timestamp(point["timestamp"])
                        value = float(point["value"])
                        source = point.get("source")
                        metadata = json.dumps(point.get("metadata", {}))
                        await conn.execute(query, factor_key, timestamp, value, source, metadata)
                        inserted += 1
                logger.info("Stored %s macro readings for %s", inserted, factor_key)
                return inserted
        except Exception as exc:
            logger.error("Error storing macro data for %s: %s", factor_key, exc)
            raise

    async def get_macro_series(
        self,
        factor_key: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 5000,
    ) -> List[Dict]:
        """Fetch macro factor series from storage."""
        try:
            async with self.get_connection() as conn:
                query = """
                    SELECT ts, value, source, metadata
                    FROM macro_factors
                    WHERE factor_key = $1
                """
                params: List = [factor_key.upper()]
                param_index = 1

                if start_date:
                    param_index += 1
                    query += f" AND ts >= ${param_index}"
                    params.append(start_date)
                if end_date:
                    param_index += 1
                    query += f" AND ts <= ${param_index}"
                    params.append(end_date)

                query += " ORDER BY ts DESC"
                if limit:
                    param_index += 1
                    query += f" LIMIT ${param_index}"
                    params.append(limit)

                rows = await conn.fetch(query, *params)
                return [
                    {
                        "timestamp": row["ts"].isoformat(),
                        "value": float(row["value"]),
                        "source": row["source"],
                        "metadata": row["metadata"] or {},
                    }
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Error retrieving macro data for %s: %s", factor_key, exc)
            raise

    async def get_latest_macro_point(self, factor_key: str) -> Optional[Dict]:
        """Return most recent macro reading for a factor."""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT ts, value, source, metadata
                    FROM macro_factors
                    WHERE factor_key = $1
                    ORDER BY ts DESC
                    LIMIT 1
                    """,
                    factor_key.upper(),
                )
                if not row:
                    return None
                return {
                    "timestamp": row["ts"].isoformat(),
                    "value": float(row["value"]),
                    "source": row["source"],
                    "metadata": row["metadata"] or {},
                }
        except Exception as exc:
            logger.error("Error fetching latest macro value for %s: %s", factor_key, exc)
            return None

    async def store_options_metrics(self, record: Dict[str, Any]) -> None:
        query = """
            INSERT INTO options_metrics (
                symbol,
                as_of,
                expiry,
                underlying_price,
                atm_strike,
                atm_iv,
                call_volume,
                put_volume,
                call_open_interest,
                put_open_interest,
                put_call_volume_ratio,
                put_call_oi_ratio,
                straddle_price,
                implied_move_pct,
                implied_move_upper,
                implied_move_lower,
                iv_25d_call,
                iv_25d_put,
                iv_skew_25d,
                iv_skew_25d_pct,
                metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21::jsonb
            )
            ON CONFLICT (symbol, as_of, expiry) DO UPDATE SET
                underlying_price = EXCLUDED.underlying_price,
                atm_strike = EXCLUDED.atm_strike,
                atm_iv = EXCLUDED.atm_iv,
                call_volume = EXCLUDED.call_volume,
                put_volume = EXCLUDED.put_volume,
                call_open_interest = EXCLUDED.call_open_interest,
                put_open_interest = EXCLUDED.put_open_interest,
                put_call_volume_ratio = EXCLUDED.put_call_volume_ratio,
                put_call_oi_ratio = EXCLUDED.put_call_oi_ratio,
                straddle_price = EXCLUDED.straddle_price,
                implied_move_pct = EXCLUDED.implied_move_pct,
                implied_move_upper = EXCLUDED.implied_move_upper,
                implied_move_lower = EXCLUDED.implied_move_lower,
                iv_25d_call = EXCLUDED.iv_25d_call,
                iv_25d_put = EXCLUDED.iv_25d_put,
                iv_skew_25d = EXCLUDED.iv_skew_25d,
                iv_skew_25d_pct = EXCLUDED.iv_skew_25d_pct,
                metadata = EXCLUDED.metadata
        """
        metadata_json = json.dumps(record.get('metadata', {}))
        params = (
            record['symbol'],
            record['as_of'],
            record['expiry'],
            record.get('underlying_price'),
            record.get('atm_strike'),
            record.get('atm_iv'),
            record.get('call_volume'),
            record.get('put_volume'),
            record.get('call_open_interest'),
            record.get('put_open_interest'),
            record.get('put_call_volume_ratio'),
            record.get('put_call_oi_ratio'),
            record.get('straddle_price'),
            record.get('implied_move_pct'),
            record.get('implied_move_upper'),
            record.get('implied_move_lower'),
            record.get('iv_25d_call'),
            record.get('iv_25d_put'),
            record.get('iv_skew_25d'),
            record.get('iv_skew_25d_pct'),
            metadata_json,
        )
        async with self.get_connection() as conn:
            await conn.execute(query, *params)

    async def get_latest_options_metrics(self, symbol: str) -> Optional[Dict]:
        query = """
            SELECT *
            FROM options_metrics
            WHERE symbol = $1
            ORDER BY as_of DESC
            LIMIT 1
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(query, symbol.upper())
            if not row:
                return None
            return self._row_to_options_metrics(row)

    async def get_options_metrics_history(self, symbol: str, limit: int = 50) -> List[Dict]:
        query = """
            SELECT *
            FROM options_metrics
            WHERE symbol = $1
            ORDER BY as_of DESC
            LIMIT $2
        """
        async with self.get_connection() as conn:
            rows = await conn.fetch(query, symbol.upper(), limit)
            return [self._row_to_options_metrics(row) for row in rows]

    def _row_to_options_metrics(self, row) -> Dict:
        if row is None:
            return {}
        metadata = row['metadata'] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        return {
            'symbol': row['symbol'],
            'as_of': row['as_of'].isoformat(),
            'expiry': row['expiry'].isoformat() if row['expiry'] else None,
            'underlying_price': float(row['underlying_price']) if row['underlying_price'] is not None else None,
            'atm_strike': float(row['atm_strike']) if row['atm_strike'] is not None else None,
            'atm_iv': float(row['atm_iv']) if row['atm_iv'] is not None else None,
            'call_volume': int(row['call_volume']) if row['call_volume'] is not None else 0,
            'put_volume': int(row['put_volume']) if row['put_volume'] is not None else 0,
            'call_open_interest': int(row['call_open_interest']) if row['call_open_interest'] is not None else 0,
            'put_open_interest': int(row['put_open_interest']) if row['put_open_interest'] is not None else 0,
            'put_call_volume_ratio': float(row['put_call_volume_ratio']) if row['put_call_volume_ratio'] is not None else None,
            'put_call_oi_ratio': float(row['put_call_oi_ratio']) if row['put_call_oi_ratio'] is not None else None,
            'straddle_price': float(row['straddle_price']) if row['straddle_price'] is not None else None,
            'implied_move_pct': float(row['implied_move_pct']) if row['implied_move_pct'] is not None else None,
            'implied_move_upper': float(row['implied_move_upper']) if row['implied_move_upper'] is not None else None,
            'implied_move_lower': float(row['implied_move_lower']) if row['implied_move_lower'] is not None else None,
            'iv_25d_call': float(row['iv_25d_call']) if row['iv_25d_call'] is not None else None,
            'iv_25d_put': float(row['iv_25d_put']) if row['iv_25d_put'] is not None else None,
            'iv_skew_25d': float(row['iv_skew_25d']) if row['iv_skew_25d'] is not None else None,
            'iv_skew_25d_pct': float(row['iv_skew_25d_pct']) if row['iv_skew_25d_pct'] is not None else None,
            'metadata': metadata,
        }

    @staticmethod
    def _coerce_timestamp(value) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return datetime.fromtimestamp(float(value))
        raise ValueError(f"Unsupported timestamp value: {value!r}")


# Global singleton used throughout the service

db_service = DatabaseService()
