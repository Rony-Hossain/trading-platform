"""Heuristic categorization of market events."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass
class CategorizationResult:
    category: str
    raw_category: Optional[str]
    confidence: float
    tags: List[str]
    matched_keywords: List[str]


class EventCategorizer:
    """Map provider-specific events into canonical categories."""

    DEFAULT_CATEGORY = "other"
    CATEGORY_KEYWORDS: Dict[str, Sequence[str]] = {
        "earnings": (
            "earnings",
            "results",
            "quarterly",
            "q1",
            "q2",
            "q3",
            "q4",
            "annual",
            "call",
            "financial",
            "eps",
            "revenue",
        ),
        "fda_approval": (
            "fda",
            "pdufa",
            "drug application",
            "biologics",
            "nda",
            "anda",
            "ema",
            "clinical",
            "trial data",
            "phase",
            "approval",
        ),
        "mna": (
            "merger",
            "acquisition",
            "takeover",
            "buyout",
            "deal",
            "transaction",
            "strategic review",
            "going private",
        ),
        "regulatory": (
            "sec",
            "regulator",
            "compliance",
            "investigation",
            "lawsuit",
            "settlement",
            "cftc",
            "doj",
            "fine",
        ),
        "product_launch": (
            "launch",
            "product",
            "release",
            "unveil",
            "update",
            "upgrade",
            "rollout",
        ),
        "analyst_day": (
            "analyst day",
            "investor day",
            "capital markets",
            "conference",
            "presentation",
        ),
        "guidance": (
            "guidance",
            "outlook",
            "forecast",
            "update",
        ),
        "dividend": (
            "dividend",
            "distribution",
            "share buyback",
            "capital return",
        ),
        "macro": (
            "cpi",
            "ppi",
            "jobs",
            "employment",
            "fomc",
            "rate",
            "gdp",
            "inflation",
            "retail sales",
        ),
        "earnings_call": (
            "earnings call",
            "conference call",
            "webcast",
        ),
        "shareholder_meeting": (
            "shareholder",
            "agm",
            "annual meeting",
            "proxy",
            "vote",
        ),
        "split": (
            "stock split",
            "split",
            "reverse split",
        ),
    }

    TAG_KEYWORDS: Dict[str, Sequence[str]] = {
        "healthcare": ("biotech", "drug", "fda", "clinical", "pharma"),
        "technology": ("software", "chip", "ai", "cloud", "semiconductor"),
        "energy": ("oil", "gas", "energy", "pipeline", "refinery"),
        "financials": ("bank", "loan", "credit", "asset management"),
    }

    def __init__(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        base = dict(self.CATEGORY_KEYWORDS)
        if overrides:
            for key, patterns in overrides.items():
                if isinstance(patterns, Iterable):
                    base[key.lower()] = tuple(str(p).lower() for p in patterns)
        env_override = os.getenv("EVENT_CATEGORY_OVERRIDES")
        if env_override:
            try:
                payload = json.loads(env_override)
                if isinstance(payload, dict):
                    for key, patterns in payload.items():
                        if isinstance(patterns, Iterable):
                            base[key.lower()] = tuple(str(p).lower() for p in patterns)
            except json.JSONDecodeError:
                pass
        self._category_keywords = base

    def categorize(
        self,
        *,
        raw_category: Optional[str],
        title: Optional[str],
        description: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CategorizationResult:
        haystack_parts: List[str] = []
        if raw_category:
            haystack_parts.append(raw_category)
        if title:
            haystack_parts.append(title)
        if description:
            haystack_parts.append(description)
        if metadata:
            for key in ("notes", "details", "event_type", "event_name"):
                value = metadata.get(key)
                if isinstance(value, str):
                    haystack_parts.append(value)
        haystack = " ".join(haystack_parts).lower()

        matched_keywords: List[str] = []
        best_category = self.DEFAULT_CATEGORY
        best_score = 0
        for category, patterns in self._category_keywords.items():
            score = 0
            matches: List[str] = []
            for token in patterns:
                if token in haystack:
                    score += 1
                    matches.append(token)
            if score > best_score:
                best_score = score
                best_category = category
                matched_keywords = matches

        # fallback heuristics
        if best_score == 0 and raw_category:
            best_category = raw_category.lower().replace(" ", "_")

        confidence = min(1.0, best_score / 3) if best_score else 0.1
        tags = self._extract_tags(haystack)
        return CategorizationResult(
            category=best_category,
            raw_category=raw_category,
            confidence=confidence,
            tags=tags,
            matched_keywords=matched_keywords,
        )

    def categories(self) -> List[str]:
        return sorted(set(self._category_keywords.keys()))

    def _extract_tags(self, haystack: str) -> List[str]:
        tags: List[str] = []
        for tag, tokens in self.TAG_KEYWORDS.items():
            if any(token in haystack for token in tokens):
                tags.append(tag)
        return tags


def build_default_categorizer() -> EventCategorizer:
    return EventCategorizer()
