from __future__ import annotations

from pathlib import Path
from typing import Dict, MutableMapping

import httpx
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.conftest import ServiceAvailability  # pragma: no cover

scenarios(str(Path(__file__).resolve().parent.parent / "market_data_price.feature"))


@pytest.fixture
def bdd_context() -> MutableMapping[str, object]:
    return {}


@given(parsers.cfparse('the Market Data API base URL "{url}"'))
def base_url_step(url: str, bdd_context: MutableMapping[str, object], base_url: str) -> None:
    bdd_context["base_url"] = base_url if url.startswith("http://localhost") else url


@when(parsers.cfparse('I GET "{path}"'))
@pytest.mark.asyncio
async def get_step(
    path: str,
    bdd_context: MutableMapping[str, object],
    service_availability: ServiceAvailability,
) -> None:
    if not service_availability.available:
        pytest.skip(service_availability.reason)
    async with httpx.AsyncClient(base_url=str(bdd_context["base_url"]), timeout=10.0) as client:
        try:
            response = await client.get(path)
        except httpx.RequestError as exc:  # pragma: no cover - external failure
            pytest.skip(f"Request to {path} failed: {exc}")  # pragma: no cover
    bdd_context["response"] = response


@then(parsers.cfparse("the response status is {status:d}"))
def status_step(status: int, bdd_context: MutableMapping[str, object]) -> None:
    response = bdd_context.get("response")
    if response is None:
        pytest.skip("HTTP response not captured; service likely unavailable.")
    assert isinstance(response, httpx.Response)
    assert response.status_code == status, response.text


@then(parsers.cfparse('the json has a numeric field "{field}"'))
def json_numeric_field(field: str, bdd_context: MutableMapping[str, object]) -> None:
    response: httpx.Response = bdd_context["response"]  # type: ignore[assignment]
    payload = response.json()
    value = payload.get(field) if isinstance(payload, Dict) else None
    assert isinstance(value, (int, float)), f"{field} must be numeric (payload: {payload})"
    assert value > 0, f"{field} must be positive (payload: {payload})"
