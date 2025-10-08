from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Generator

import httpx
import pytest
import pytest_asyncio
import respx
from dotenv import load_dotenv
from freezegun import freeze_time as _freeze_time
from viztracer import VizTracer

REQUIRE_OPTIONALS = os.getenv("REQUIRE_OPTIONAL_ENDPOINTS", "0") == "1"
REQUIRE_DEEP_HISTORY = os.getenv("REQUIRE_DEEP_HISTORY", "0") == "1"
REQUIRE_STRICT_VALIDATION = os.getenv("REQUIRE_STRICT_VALIDATION", "0") == "1"
REQUIRE_OPTIONS_DATA = os.getenv("REQUIRE_OPTIONS_DATA", "0") == "1"

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

ARTIFACT_ROOT = Path("artifacts")
VIZTRACE_DIR = ARTIFACT_ROOT / "viztraces"


@dataclass
class ServiceAvailability:
    available: bool
    reason: str = ""
    details: Dict[str, Any] | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live-tests",
        action="store_true",
        default=False,
        help="Run tests even if /health probe fails.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "profile: capture VizTracer traces for the marked test.")


def pytest_sessionstart(session: pytest.Session) -> None:  # pragma: no cover - session hook
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    VIZTRACE_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_nodeid(nodeid: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    marker = item.get_closest_marker("profile")
    tracer: VizTracer | None = None
    if marker:
        tracer = VizTracer(tracer_entries=500_000)
        tracer.start()
    outcome = yield
    if tracer:
        tracer.stop()
        output_file = VIZTRACE_DIR / f"{sanitize_nodeid(item.nodeid)}.json"
        try:
            tracer.save(str(output_file))
        except Exception as exc:  # pragma: no cover - filesystem issues
            item.warn(pytest.PytestWarning(f"Failed to save VizTrace: {exc}"))  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.getenv("BASE_URL", "http://localhost:8002")


@pytest.fixture(scope="session")
def ws_url() -> str:
    return os.getenv("WS_URL", "ws://localhost:8002/ws")


@pytest_asyncio.fixture(scope="session")
async def service_availability(
    base_url: str, pytestconfig: pytest.Config
) -> ServiceAvailability:
    health_url = f"{base_url.rstrip('/')}/health"
    timeout = float(os.getenv("SERVICE_HEALTH_TIMEOUT", "5"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(health_url)
        except Exception as exc:
            available = pytestconfig.getoption("--live-tests") or os.getenv("LIVE_TESTS") == "1"
            reason = f"Market Data service unreachable ({exc})"
            return ServiceAvailability(available=available, reason=reason)
    details: Dict[str, Any] = {"status_code": response.status_code}
    try:
        details["body"] = response.json()
    except ValueError:
        details["body"] = response.text
    if response.status_code != 200:
        available = pytestconfig.getoption("--live-tests") or os.getenv("LIVE_TESTS") == "1"
        reason = f"/health returned {response.status_code}"
        return ServiceAvailability(available=available, reason=reason, details=details)
    return ServiceAvailability(available=True, details=details)


@pytest_asyncio.fixture
async def api_client(base_url: str) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0, follow_redirects=True) as client:
        yield client


@pytest.fixture(scope="session")
def tracked_symbols() -> Dict[str, Any]:
    return {
        "liquid": ["AAPL", "MSFT", "INTC"],
        "etf": "SPY",
        "invalid": "ZZZINVALID",
        "bad_format": "$BAD",
        "non_option": "NONOPT",
    }


@pytest.fixture(scope="session")
def iso_parser() -> Callable[[str], datetime]:
    def _parse(ts: str | None) -> datetime:
        if not ts:
            raise AssertionError("Timestamp missing")
        candidate = str(ts)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", candidate):
            candidate = f"{candidate}T00:00:00"
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        if re.match(r".*[+-]\d{2}:\d{2}$", candidate) is None:
            candidate = candidate + "+00:00"
        return datetime.fromisoformat(candidate)

    return _parse


@pytest.fixture(scope="session")
def history_expectations() -> Dict[str, Any]:
    expectation_path = Path(__file__).resolve().parent / "data_quality" / "expectations_history.json"
    with expectation_path.open("r", encoding="utf-8") as handle:
        import json

        return json.load(handle)


@pytest_asyncio.fixture(scope="session")
async def event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def respx_mock() -> Generator[respx.Router, None, None]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def freeze_time() -> Generator[Callable[..., None], None, None]:
    active = []

    def _activate(target: str, tz_offset: int = -4) -> None:
        ctx = _freeze_time(target, tz_offset=tz_offset)
        ctx.start()
        active.append(ctx)

    yield _activate
    for ctx in active:
        ctx.stop()


@pytest.fixture(scope="session")
def optional_endpoint_guard() -> Callable[[httpx.Response, str], None]:
    def _guard(response: httpx.Response, endpoint: str) -> None:
        if response.status_code == 404:
            message = f"{endpoint} not enabled in this deployment."
            if REQUIRE_OPTIONALS:
                pytest.fail(message)
            pytest.skip(message)

    return _guard


@pytest.fixture(scope="session")
def require_deep_history() -> bool:
    return REQUIRE_DEEP_HISTORY


@pytest.fixture(scope="session")
def require_strict_validation() -> bool:
    return REQUIRE_STRICT_VALIDATION


@pytest.fixture(scope="session")
def require_options_data() -> bool:
    return REQUIRE_OPTIONS_DATA
