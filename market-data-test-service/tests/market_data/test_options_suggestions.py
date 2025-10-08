from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.asyncio


async def test_options_suggestions_sorted_by_score(
    api_client, service_availability, tracked_symbols
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    response = await api_client.get(f"/options/{tracked_symbols['liquid'][0]}/suggestions")
    assert response.status_code == 200, response.text
    payload = response.json()
    suggestions = payload.get("suggestions") if isinstance(payload, dict) else payload
    assert isinstance(suggestions, list) and suggestions, f"Suggestions missing: {payload}"
    scores = [item.get("score") for item in suggestions if item.get("score") is not None]
    assert scores, "Suggestions missing score field"
    assert all(isinstance(score, (int, float)) and not math.isnan(score) for score in scores)
    assert scores == sorted(scores, reverse=True), f"Scores not sorted descending: {scores}"
