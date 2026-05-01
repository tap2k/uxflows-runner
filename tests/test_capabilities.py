"""Capability dispatch — input resolution, fire-and-forget, error surfaces."""

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
    results: list[caps.CapabilityResult] = []
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={
            "place_order": caps.CapabilityEndpoint(
                url="https://example.test/place", headers={"Authorization": "Bearer x"}
            )
        },
        on_result=results.append,
        client=client,
    )

    bag = {
        "drink_type": "coffee", "drink_style": "latte", "size": "large",
        "milk": "oat", "syrup": "none", "honey": False,
    }
    invocation = dispatcher.invoke("cap_place_order", bag)
    assert invocation.capability_name == "place_order"
    assert invocation.args["drink_style"] == "latte"

    await dispatcher.aclose()
    assert received["url"] == "https://example.test/place"
    assert received["body"]["drink_style"] == "latte"
    assert received["auth"] == "Bearer x"
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].result == {"order_id": "abc"}


@pytest.mark.asyncio
async def test_invoke_missing_endpoint_emits_error(coffee):
    results: list[caps.CapabilityResult] = []
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={},  # no endpoint configured
        on_result=results.append,
    )
    dispatcher.invoke("cap_log_walkaway", {})
    await dispatcher.aclose()
    assert len(results) == 1
    assert "no endpoint configured" in results[0].error


@pytest.mark.asyncio
async def test_invoke_http_error_surfaces(coffee):
    def handler(_request):
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=_mock_transport(handler))
    results: list[caps.CapabilityResult] = []
    dispatcher = caps.CapabilityDispatcher(
        spec=coffee,
        endpoints={
            "log_walkaway": caps.CapabilityEndpoint(url="https://example.test/log")
        },
        on_result=results.append,
        client=client,
    )
    dispatcher.invoke("cap_log_walkaway", {})
    await dispatcher.aclose()
    assert results[0].error is not None
    assert "500" in results[0].error


def test_make_invocation_helper(coffee):
    inv = caps.make_invocation(
        coffee, "cap_place_order", {"drink_type": "coffee", "drink_style": "latte"}
    )
    assert inv.capability_name == "place_order"
    assert inv.args == {"drink_type": "coffee", "drink_style": "latte"}
