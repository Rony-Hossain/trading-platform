"""Finnhub fundamentals client helpers for consensus, analyst revisions, and insider flow."""

import os
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class FinnhubFundamentalsClient:
    """Thin async wrapper around Finnhub endpoints used by the fundamentals service."""

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        self.api_key = (api_key or os.getenv("FINNHUB_API_KEY") or "").strip()
        self.timeout = timeout

    @property
    def is_enabled(self) -> bool:
        return bool(self.api_key)

    async def fetch_consensus_estimates(self, symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
        """Return recent earnings consensus/actual results."""
        payload = await self._get(
            "stock/earnings",
            {
                "symbol": symbol.upper(),
                "limit": limit,
            },
        )
        if not payload:
            return []

        if isinstance(payload, list):
            records: List[Dict[str, Any]] = payload
        else:
            records = payload.get("earnings") or payload.get("data") or []

        results: List[Dict[str, Any]] = []
        for item in records[:limit]:
            period = item.get("period") or item.get("date")
            report_date = self._parse_date(period)
            fiscal_period = item.get("quarter") or self._infer_fiscal_period(report_date)
            fiscal_year = item.get("fiscalYear") or (report_date.year if report_date else None)
            data = {
                "symbol": symbol.upper(),
                "report_date": report_date or date.today(),
                "fiscal_period": fiscal_period or "NA",
                "fiscal_year": int(fiscal_year) if fiscal_year else date.today().year,
                "analyst_count": self._safe_number(item.get("numberAnalysts"), as_int=True),
                "estimate_eps": self._safe_number(item.get("estimate") or item.get("consensusEPS")),
                "estimate_eps_high": self._safe_number(item.get("estimateHigh") or item.get("epsHigh")),
                "estimate_eps_low": self._safe_number(item.get("estimateLow") or item.get("epsLow")),
                "actual_eps": self._safe_number(item.get("actual") or item.get("actualEPS")),
                "surprise_percent": self._safe_number(item.get("surprisePercent")),
                "estimate_revenue": self._safe_number(item.get("revenueEstimate") or item.get("estimateRevenue"), as_int=True),
                "estimate_revenue_high": self._safe_number(item.get("revenueEstimateHigh") or item.get("estimateRevenueHigh"), as_int=True),
                "estimate_revenue_low": self._safe_number(item.get("revenueEstimateLow") or item.get("estimateRevenueLow"), as_int=True),
                "actual_revenue": self._safe_number(item.get("revenueActual") or item.get("actualRevenue"), as_int=True),
                "guidance_eps": self._safe_number(item.get("epsGuidance")),
                "guidance_revenue": self._safe_number(item.get("revenueGuidance"), as_int=True),
                "source": "finnhub",
            }
            results.append(data)
        return results

    async def fetch_institutional_holdings(
        self, symbol: str, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Return institutional fund ownership (13F-style) data."""
        payload = await self._get(
            "stock/fund-ownership",
            {"symbol": symbol.upper(), "limit": limit},
        )
        if not payload:
            return []

        records = payload.get("ownership") if isinstance(payload, dict) else []
        results: List[Dict[str, Any]] = []
        for item in records:
            report_date = self._parse_date(item.get("reportDate")) or date.today()
            market_value = self._safe_number(item.get("marketValue"))
            shares_held = self._safe_number(item.get("share"), as_int=True)
            if shares_held is None:
                shares_held = 0

            entry = {
                "symbol": symbol.upper(),
                "institution_name": (item.get("entityProperName") or item.get("symbol") or "Unknown Institution").strip(),
                "institution_cik": (item.get("cik") or "UNKNOWN").strip(),
                "filing_date": report_date,
                "quarter_end": report_date,
                "shares_held": shares_held,
                "market_value": market_value if market_value is not None else 0.0,
                "percentage_ownership": self._safe_number(item.get("percentage")),
                "shares_change": self._safe_number(item.get("shareChange"), as_int=True),
                "shares_change_pct": self._safe_number(item.get("shareChangePercent")),
                "form13f_url": item.get("filingUrl"),
                "is_new_position": bool(item.get("isNew")),
                "is_sold_out": bool(item.get("isSoldOut")),
            }
            results.append(entry)
        return results

    async def fetch_insider_transactions(self, symbol: str, lookback_days: int = 180) -> List[Dict[str, Any]]:
        """Return insider transactions (Form 4 style) for the lookback window."""
        end = date.today()
        start = end - timedelta(days=lookback_days)
        payload = await self._get(
            "stock/insider-transactions",
            {
                "symbol": symbol.upper(),
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
        )
        if not payload:
            return []

        records = payload.get("data") if isinstance(payload, dict) else []
        results: List[Dict[str, Any]] = []
        for item in records:
            transaction_date = self._parse_date(item.get("transactionDate")) or end
            filing_date = self._parse_date(item.get("filingDate"))
            data = {
                "symbol": symbol.upper(),
                "insider": (item.get("name") or "").strip() or "Unknown",
                "relationship": (item.get("position") or item.get("relationship") or "").strip() or None,
                "transaction_date": transaction_date,
                "transaction_type": (item.get("transactionCode") or item.get("transactionType") or "").strip() or None,
                "shares": self._safe_number(item.get("share"), as_int=True),
                "share_change": self._safe_number(item.get("change") or item.get("shareChange"), as_int=True),
                "price": self._safe_number(item.get("transactionPrice")),
                "total_value": self._safe_number(item.get("transactionValue") or item.get("total"), as_int=True),
                "filing_date": filing_date,
                "link": item.get("link") or item.get("url"),
                "source": "finnhub",
            }
            results.append(data)
        return results

    async def fetch_analyst_revisions(self, symbol: str, lookback_days: int = 120) -> List[Dict[str, Any]]:
        """Return analyst upgrades/downgrades and price target revisions."""
        end = date.today()
        start = end - timedelta(days=lookback_days)
        payload = await self._get(
            "stock/upgrade-downgrade",
            {
                "symbol": symbol.upper(),
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
        )
        if not payload:
            return []

        records = payload if isinstance(payload, list) else payload.get("data") or []
        results: List[Dict[str, Any]] = []
        for item in records:
            revision_date = self._parse_date(item.get("gradeTime") or item.get("publishedDate")) or end
            data = {
                "symbol": symbol.upper(),
                "revision_date": revision_date,
                "analyst": (item.get("analyst") or item.get("analystName") or "").strip() or None,
                "firm": (item.get("company") or item.get("firm") or "").strip() or None,
                "action": (item.get("action") or item.get("grade") or "").strip() or None,
                "from_rating": (item.get("fromGrade") or item.get("fromRating")) or None,
                "to_rating": (item.get("toGrade") or item.get("toRating")) or None,
                "old_price_target": self._safe_number(item.get("ptPrior") or item.get("targetPricePrior")),
                "new_price_target": self._safe_number(item.get("pt") or item.get("targetPrice")),
                "rating_score": self._safe_number(item.get("score")),
                "notes": item.get("notes"),
                "source": "finnhub",
            }
            results.append(data)
        return results

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not self.is_enabled:
            logger.debug("Finnhub API key not configured; skipping request to %s", path)
            return None

        params = params.copy() if params else {}
        params.setdefault("token", self.api_key)
        url = f"{self.BASE_URL}/{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                if not response.text:
                    return None
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Finnhub returned %s for %s: %s", exc.response.status_code, path, exc)
        except httpx.HTTPError as exc:
            logger.warning("Finnhub request error for %s: %s", path, exc)
        return None

    @staticmethod
    def _parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except Exception:
                return None

    @staticmethod
    def _infer_fiscal_period(report_date: Optional[date]) -> Optional[str]:
        if report_date is None:
            return None
        month = ((report_date.month - 1) // 3) + 1
        return f"Q{month}"

    async def get_earnings_calendar(self, from_date: str, to_date: str) -> Dict[str, Any]:
        """Get earnings calendar for date range"""
        payload = await self._get(
            "calendar/earnings",
            {
                "from": from_date,
                "to": to_date,
            },
        )
        return payload or {}
    
    async def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financial metrics"""
        payload = await self._get(
            "stock/metric",
            {
                "symbol": symbol.upper(),
                "metric": "all",
            },
        )
        return payload or {}
    
    async def get_company_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """Get company quarterly earnings data"""
        payload = await self._get(
            "stock/earnings",
            {
                "symbol": symbol.upper(),
            },
        )
        if isinstance(payload, list):
            return payload
        return payload.get("earnings", []) if payload else []
    
    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        payload = await self._get(
            "stock/profile2",
            {
                "symbol": symbol.upper(),
            },
        )
        return payload or {}

    @staticmethod
    def _safe_number(value: Any, *, as_int: bool = False) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            number = float(value)
            if as_int:
                return int(number)
            return number
        except (TypeError, ValueError):
            return None

__all__ = ["FinnhubFundamentalsClient"]
