import asyncio
import math
from typing import Any, Dict, Optional

import aiohttp


class CompositeAlgo:
    def __init__(self, market_data_url: str, sentiment_api_url: str, fundamentals_api_url: str):
        self.market_data_url = market_data_url.rstrip("/")
        self.sentiment_api_url = sentiment_api_url.rstrip("/")
        self.fundamentals_api_url = fundamentals_api_url.rstrip("/")

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict[str, Any]]:
        try:
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None

    def _score_from_quick(self, quick: Dict[str, Any]) -> float:
        try:
            market = quick.get("market_status", {})
            quick_tech = quick.get("quick_technical", {})
            direction = market.get("direction")
            base = 0.0
            if direction == "UP":
                base += 0.3
            elif direction == "DOWN":
                base -= 0.3
            # price_change_5d normalized ~ 10% -> +/- 1.0 cap
            pc5 = float(quick_tech.get("price_change_5d", 0))
            base += max(min(pc5 / 10.0, 1.0), -1.0)
            # modest penalty for very high volatility (> 0.4 treated as elevated)
            vol = float(quick_tech.get("volatility_20d", 0))
            base -= max(0.0, vol - 0.4) * 0.5
            return max(min(base, 1.0), -1.0)
        except Exception:
            return 0.0

    def _score_from_sentiment(self, sentiment_summary: Optional[Dict[str, Any]]) -> float:
        if not sentiment_summary:
            return 0.0
        try:
            # prefer numeric score if present (-1..1), otherwise map label
            if "sentiment_score" in sentiment_summary and sentiment_summary["sentiment_score"] is not None:
                val = float(sentiment_summary["sentiment_score"])
                return max(min(val, 1.0), -1.0)
            label = (sentiment_summary.get("current_sentiment") or "").upper()
            mapping = {"BULLISH": 0.5, "BEARISH": -0.5, "NEUTRAL": 0.0}
            return mapping.get(label, 0.0)
        except Exception:
            return 0.0

    def _score_from_fundamentals(self, metrics: Optional[Dict[str, Any]]) -> float:
        if not metrics:
            return 0.0
        try:
            # Heuristic composite from common metrics; fallbacks to mid if missing
            growth = float(metrics.get("revenue_growth", 0.0))  # percent
            roe = float(metrics.get("roe", 0.0))  # percent
            debt = float(metrics.get("debt_ratio", 0.0))  # 0..1
            margin = float(metrics.get("net_margin", 0.0))  # percent

            # Normalize: map typical ranges to [-1,1]
            growth_s = max(min(growth / 20.0, 1.0), -1.0)  # +/-20% ~ +/-1
            roe_s = max(min(roe / 20.0, 1.0), -1.0)
            margin_s = max(min(margin / 20.0, 1.0), -1.0)
            debt_s = -max(min(debt / 0.8, 1.0), 0.0)  # higher debt worse

            score = 0.35 * growth_s + 0.35 * roe_s + 0.2 * margin_s + 0.1 * debt_s
            return max(min(score, 1.0), -1.0)
        except Exception:
            return 0.0

    async def compute(self, symbol: str, quick_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # Parallel fetch sentiment summary and fundamentals metrics
        async with aiohttp.ClientSession() as session:
            sent_url = f"{self.sentiment_api_url}/summary/{symbol}"
            fund_url = f"{self.fundamentals_api_url}/metrics/{symbol}"
            sentiment_task = asyncio.create_task(self._fetch_json(session, sent_url))
            fundamentals_task = asyncio.create_task(self._fetch_json(session, fund_url))
            sentiment_summary, fundamentals_metrics = await asyncio.gather(sentiment_task, fundamentals_task)

        technical_score = self._score_from_quick(quick_snapshot)
        sentiment_score = self._score_from_sentiment(sentiment_summary)
        fundamentals_score = self._score_from_fundamentals(fundamentals_metrics)

        # Weighting regime based on volatility
        vol = float(quick_snapshot.get("quick_technical", {}).get("volatility_20d", 0) or 0)
        if vol > 0.3:
            weights = {"technical": 0.5, "sentiment": 0.3, "fundamentals": 0.2}
        else:
            weights = {"technical": 0.4, "sentiment": 0.2, "fundamentals": 0.4}

        composite = (
            weights["technical"] * technical_score +
            weights["sentiment"] * sentiment_score +
            weights["fundamentals"] * fundamentals_score
        )

        return {
            "symbol": symbol,
            "scores": {
                "technical": technical_score,
                "sentiment": sentiment_score,
                "fundamentals": fundamentals_score,
            },
            "weights": weights,
            "composite_score": round(composite, 4),
            "inputs": {
                "sentiment_summary": sentiment_summary,
                "fundamentals_metrics": fundamentals_metrics,
            }
        }

