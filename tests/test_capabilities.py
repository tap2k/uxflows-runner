"""Capability dispatch — input resolution, sync await, output binding, error surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from uxflows_runner.dispatcher import capabilities as caps
from uxflows_runner.spec.loader import load_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
COFFEE = REPO_ROOT / "examples" / "coffee.json"


@pytest.fixture(scope="module")
def coffee():
    return load_spec(COFFEE)


def test_resolve_inputs_pulls_declared_only(coffee):
    cap = coffee.capabilities_by_name["place_order"]
    bag = {
        "drink_type": "coffee",
        "drink_style": "latte",
        "size": "large",
        "milk": "oat",
        "syrup": "vanilla",
        "honey": False,
        "irrelevant_var": "should_not_appear",
    }
    args = caps.resolve_inputs(cap, bag)
    assert "irrelevant_var" not in args
    assert args["drink_style"] == "latte"
    assert args["honey"] is False


def test_resolve_inputs_skips_missing(coffee):
    cap = coffee.capabilities_by_name["place_order"]
    args = caps.resolve_inputs(cap, {"drink_style": "drip"})
    assert args == {"drink_style": "drip"}


def test_load_execution_config_missing_file_returns_empty(tmp_path):
    assert caps.load_execution_config(tmp_path / "nope.json") == {}


def test_load_execution_config_parses(tmp_path):
    cfg = tmp_path / "execution.json"
    cfg.write_text(json.dumps({
        "capabilities": {
            "place_order": {"url": "https://example.test/place", "headers": {"X": "1"}}
        }
    }))
    out = caps.load_execution_config(cfg)
    assert out["place_order"].url == "https://example.test/place"
    assert out["place_order"].headers == {"X": "1"}


def _mock_transport(handler):
    async def _h(request: httpx.Request) -> httpx.Response:
        return handler(request)

    return httpx.MockTransport(_h)


@pytest.mark.asyncio
async def test_invoke_function_capability_posts_resolved_args(coffee):
    received = {}

    def handler(request: httpx.Request):
        received["url"] = str(request.url)
        received["body"] = json.loads(request.content)
        received["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"order_id": "abc"})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={
            "place_order": caps.CapabilityEndpoint(
                url="https://example.test/place", headers={"Authorization": "Bearer x"}
            )
        },
        client=client,
    )

    bag = {
        "drink_type": "coffee", "drink_style": "latte", "size": "large",
        "milk": "oat", "syrup": "none", "honey": False,
    }
    invocation, result = await dispatcher.invoke("cap_place_order", bag)
    assert invocation.capability_name == "place_order"
    assert invocation.args["drink_style"] == "latte"

    await dispatcher.aclose()
    assert received["url"] == "https://example.test/place"
    assert received["body"]["drink_style"] == "latte"
    assert received["auth"] == "Bearer x"
    assert result.error is None
    assert result.result == {"order_id": "abc"}


@pytest.mark.asyncio
async def test_invoke_missing_endpoint_surfaces_error(coffee):
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},  # no endpoint configured
    )
    _, result = await dispatcher.invoke("cap_log_walkaway", {})
    await dispatcher.aclose()
    assert result.result is None
    assert "no endpoint configured" in result.error


@pytest.mark.asyncio
async def test_invoke_http_error_surfaces(coffee):
    def handler(_request):
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={
            "log_walkaway": caps.CapabilityEndpoint(url="https://example.test/log")
        },
        client=client,
    )
    _, result = await dispatcher.invoke("cap_log_walkaway", {})
    await dispatcher.aclose()
    assert result.error is not None
    assert "500" in result.error


@pytest.mark.asyncio
async def test_retrieval_capability_returns_empty_context(coffee):
    # The coffee spec doesn't have a retrieval capability declared, so we
    # exercise the stub via a manual spec doctored at the dataclass level.
    dispatcher = caps.CapabilityDispatcher(spec=coffee, endpoints={})
    # `cap_log_walkaway` is function-kind in the coffee spec; flip its kind in
    # the loaded copy for this test. Cheaper than authoring a fixture.
    cap = coffee.capabilities_by_id["cap_log_walkaway"]
    object.__setattr__(cap, "kind", "retrieval")
    try:
        _, result = await dispatcher.invoke("cap_log_walkaway", {})
        assert result.error is None
        assert result.result == {"context": []}
    finally:
        object.__setattr__(cap, "kind", "function")
        await dispatcher.aclose()


@pytest.mark.asyncio
async def test_unknown_capability_id_surfaces_error(coffee):
    dispatcher = caps.CapabilityDispatcher(spec=coffee, endpoints={})
    invocation, result = await dispatcher.invoke("cap_does_not_exist", {})
    await dispatcher.aclose()
    assert invocation.args == {}
    assert "unknown capability_id" in result.error


@pytest.mark.asyncio
async def test_mock_returns_shadows_no_endpoint(coffee):
    """Designer-supplied mock_returns lands as a CapabilityResult.result
    even with no endpoint configured — no HTTP, no error."""
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},
        mock_returns={"place_order": {"order_id": "MOCK-001", "eta_minutes": 4}},
    )
    _, result = await dispatcher.invoke("cap_place_order", {"drink_style": "latte"})
    await dispatcher.aclose()
    assert result.error is None
    assert result.result == {"order_id": "MOCK-001", "eta_minutes": 4}


@pytest.mark.asyncio
async def test_mock_returns_shadows_real_endpoint(coffee):
    """When both a mock and an endpoint are configured for the same capability,
    the mock wins — SimulatePanel always wants deterministic returns."""
    posted = []

    def handler(request: httpx.Request):
        posted.append(str(request.url))
        return httpx.Response(200, json={"order_id": "REAL-999"})

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={"place_order": caps.CapabilityEndpoint(url="https://example.test/x")},
        mock_returns={"place_order": {"order_id": "MOCK-001"}},
        client=client,
    )
    _, result = await dispatcher.invoke("cap_place_order", {})
    await dispatcher.aclose()
    assert posted == [], "HTTP should be skipped when a mock is set"
    assert result.result == {"order_id": "MOCK-001"}


@pytest.mark.asyncio
async def test_mock_returns_keyed_by_name_not_id(coffee):
    """Mocks key by capability NAME (snake_case dispatch identifier), not by
    the spec's stable id. Matches how execution.json keys endpoints."""
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},
        # Using the id would silently no-op; only name matches.
        mock_returns={"cap_place_order": {"order_id": "WRONG"}},
    )
    _, result = await dispatcher.invoke("cap_place_order", {})
    await dispatcher.aclose()
    assert result.error is not None
    assert "no endpoint configured" in result.error


@pytest.fixture
def loguru_capture():
    """Capture loguru WARNING-level messages into an in-memory buffer.

    loguru doesn't propagate through stdlib logging by default, so caplog
    doesn't see runner warnings. This fixture installs a temporary sink.
    """
    from io import StringIO

    from loguru import logger

    buf = StringIO()
    handler_id = logger.add(buf, level="WARNING", format="{message}")
    try:
        yield buf
    finally:
        logger.remove(handler_id)


def test_mock_returns_unknown_name_warns(coffee, loguru_capture):
    """A mock_returns key that doesn't match any capability name is almost
    certainly a typo (or someone using the id by mistake). Warn at construction
    so the designer sees the mismatch before the simulation runs."""
    caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},
        mock_returns={
            "place_order": {"order_id": "ok"},      # valid
            "place_ordr": {"order_id": "typo"},     # typo
            "cap_place_order": {"order_id": "id"},  # id, not name
        },
    )
    out = loguru_capture.getvalue()
    # Extract the warning-subject (name immediately after "unknown capability name ").
    import re
    subjects = re.findall(r"unknown capability name '([^']+)'", out)
    assert "place_ordr" in subjects
    assert "cap_place_order" in subjects
    # The valid name "place_order" must not be a warning subject.
    assert "place_order" not in subjects


@pytest.mark.asyncio
async def test_mock_returns_stray_keys_warn(coffee, loguru_capture):
    """If a mock dict has keys not in cap.outputs, those values won't bind to
    anything — warn so the designer notices instead of debugging why a value
    'didn't land'."""
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},
        # place_order has no declared outputs in coffee.json — every key here
        # is stray.
        mock_returns={"place_order": {"order_id": "x", "eta_minutes": 5}},
    )
    _, result = await dispatcher.invoke("cap_place_order", {})
    await dispatcher.aclose()
    assert result.result == {"order_id": "x", "eta_minutes": 5}  # still returned
    out = loguru_capture.getvalue()
    assert "not in declared outputs" in out
    assert "order_id" in out
    assert "eta_minutes" in out


def test_make_invocation_helper(coffee):
    inv = caps.make_invocation(
        coffee, "cap_place_order", {"drink_type": "coffee", "drink_style": "latte"}
    )
    assert inv.capability_name == "place_order"
    assert inv.args == {"drink_type": "coffee", "drink_style": "latte"}
