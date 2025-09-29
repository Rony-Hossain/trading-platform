"""Event impact scoring heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass
class ImpactScoreResult:
    score: int
    components: Dict[str, Any]


class EventImpactScorer:
    """Estimate market-moving potential for scheduled events (1-10 scale).

    The scorer combines category priors with metadata such as market cap,
    implied move expectations, historical sensitivity, and qualitative flags.
    It intentionally prefers deterministic heuristics so the calendar can
    produce a reasonable score even before live price data is available.
    """

    DEFAULT_CATEGORY_BASE: Dict[str, int] = {
        "earnings": 7,
        "earnings_call": 7,
        "earnings_release": 7,
        "dividend": 4,
        "dividend_change": 5,
        "product_launch": 6,
        "analyst_day": 5,
        "investor_day": 6,
        "guidance_update": 6,
        "mna": 8,
        "merger": 8,
        "regulatory": 8,
        "fda": 9,
        "fda_approval": 9,
        "clinical": 7,
        "clinical_trial": 7,
        "economic": 8,
        "macro": 8,
        "default": 5,
    }

    def __init__(self, category_base: Optional[Dict[str, int]] = None) -> None:
        base = dict(self.DEFAULT_CATEGORY_BASE)
        if category_base:
            base.update({k.lower(): v for k, v in category_base.items()})
        self._category_base = base

    # Public API ---------------------------------------------------------
    def score_event(self, event: Dict[str, Any]) -> ImpactScoreResult:
        category = (event.get("category") or "").lower()
        metadata = self._ensure_dict(event.get("metadata"))
        components: Dict[str, Any] = {}

        base_score = self._category_base.get(category, self._category_base["default"])
        components["category_base"] = base_score
        score = float(base_score)

        # Market cap tiers ------------------------------------------------
        market_cap = self._first_number(metadata, [
            "market_cap",
            "marketCap",
            "market_capitalization",
            "market_cap_usd",
            "company_market_cap",
        ])
        if market_cap is not None:
            tier_adjust, tier_label = self._market_cap_adjustment(market_cap)
            score += tier_adjust
            components["market_cap"] = {
                "value": market_cap,
                "tier": tier_label,
                "adjustment": tier_adjust,
            }

        # Expected / implied move ----------------------------------------
        implied_move = self._first_number(metadata, [
            "implied_move",
            "implied_move_pct",
            "expected_move",
            "expected_move_pct",
            "expected_price_move",
            "expected_move_percent",
        ])
        if implied_move is None:
            implied_move = self._first_number(metadata, ["implied_move_bp"])
            if implied_move is not None:
                implied_move = implied_move / 100.0
        if implied_move is not None:
            normalized_move = self._normalize_percent(implied_move)
            move_adjust = self._implied_move_adjustment(normalized_move)
            score += move_adjust
            components["implied_move"] = {
                "value_pct": normalized_move,
                "adjustment": move_adjust,
            }

        # Historical price sensitivity -----------------------------------
        historical_move = self._first_number(metadata, [
            "historical_avg_move",
            "avg_post_move",
            "historical_move_pct",
            "historical_abs_move",
        ])
        if historical_move is not None:
            normalized_hist = self._normalize_percent(historical_move)
            hist_adjust = self._historical_adjustment(normalized_hist)
            score += hist_adjust
            components["historical_sensitivity"] = {
                "value_pct": normalized_hist,
                "adjustment": hist_adjust,
            }

        # Liquidity / volume context -------------------------------------
        adv = self._first_number(metadata, [
            "avg_daily_volume",
            "average_daily_volume",
            "adv",
        ])
        if adv is not None:
            adv_adjust = self._adv_adjustment(adv)
            score += adv_adjust
            components["liquidity"] = {
                "avg_daily_volume": adv,
                "adjustment": adv_adjust,
            }

        # Qualitative flags -----------------------------------------------
        importance_flag = self._extract_flag(metadata, [
            "is_major",
            "is_critical",
            "is_flagship",
            "is_high_importance",
        ])
        if importance_flag:
            score += 1.5
            components["importance_flag"] = 1.5

        scope = (metadata.get("event_scope") or metadata.get("importance") or "").lower()
        if scope in {"global", "national", "tier1"}:
            score += 1.0
            components["scope"] = {"label": scope, "adjustment": 1.0}
        elif scope in {"regional", "tier2"}:
            score += 0.5
            components["scope"] = {"label": scope, "adjustment": 0.5}

        # Downside for low conviction -------------------------------------
        if metadata.get("confidence") is not None:
            try:
                confidence = float(metadata["confidence"])
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = self._clamp(confidence, 0.0, 1.0)
            adjust = (confidence - 0.5) * 2.0  # +/-1 based on confidence
            if adjust:
                score += adjust
                components["confidence"] = {
                    "value": confidence,
                    "adjustment": adjust,
                }

        if metadata.get("is_preliminary") or metadata.get("tentative"):
            score -= 1.0
            components["preliminary_penalty"] = -1.0

        score = self._clamp(score, 1.0, 10.0)
        final_score = int(round(score))
        components["final_score"] = final_score
        return ImpactScoreResult(score=final_score, components=components)

    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def _first_number(data: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
        for key in keys:
            if key not in data:
                continue
            value = data[key]
            if value is None:
                continue
            try:
                if isinstance(value, str):
                    stripped = value.replace("%", "").replace(",", "").strip()
                    if not stripped:
                        continue
                    return float(stripped)
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_flag(data: Dict[str, Any], keys: Iterable[str]) -> bool:
        for key in keys:
            value = data.get(key)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value >= 1:
                    return True
            elif isinstance(value, str):
                if value.lower() in {"y", "yes", "true", "high", "major"}:
                    return True
        return False

    @staticmethod
    def _market_cap_adjustment(market_cap: float) -> Tuple[float, str]:
        if market_cap >= 200e9:
            return 2.0, "mega"
        if market_cap >= 50e9:
            return 1.5, "large"
        if market_cap >= 10e9:
            return 1.0, "mid"
        if market_cap >= 2e9:
            return 0.5, "small"
        return -0.5, "micro"

    @staticmethod
    def _implied_move_adjustment(move_pct: float) -> float:
        if move_pct >= 10:
            return 3.0
        if move_pct >= 7:
            return 2.0
        if move_pct >= 4:
            return 1.0
        if move_pct <= 1.0:
            return -0.5
        return 0.0

    @staticmethod
    def _historical_adjustment(move_pct: float) -> float:
        if move_pct >= 8:
            return 1.5
        if move_pct >= 5:
            return 1.0
        if move_pct <= 1.5:
            return -0.5
        return 0.0

    @staticmethod
    def _adv_adjustment(adv: float) -> float:
        if adv >= 50_000_000:
            return 1.0
        if adv >= 10_000_000:
            return 0.5
        if adv <= 500_000:
            return -0.5
        return 0.0

    @staticmethod
    def _normalize_percent(value: float) -> float:
        if -1.0 < value < 1.0 and value != 0:
            return value * 100
        return value

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))


DEFAULT_SCORER = EventImpactScorer()
