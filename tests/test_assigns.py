"""Assigns — variable bag mutation when an exit fires."""

from __future__ import annotations

from pathlib import Path

import pytest

from uxflows_runner.dispatcher import assigns
from uxflows_runner.spec.loader import load_spec
from uxflows_runner.spec.types import Assign, ExitPath

REPO_ROOT = Path(__file__).resolve().parent.parent
COFFEE = REPO_ROOT / "examples" / "coffee.json"


@pytest.fixture(scope="module")
def coffee():
    return load_spec(COFFEE)


def test_direct_assign_fires(coffee):
    flow_confirm = coffee.flows_by_id["flow_confirm"]
    placed = next(ep for ep in flow_confirm.routing.exit_paths if ep.id == "xp_cf_placed")
    bag = {"drink_type": "coffee"}
    results = assigns.fire(placed, bag, llm_results={})
    assert bag["order_status"] == "confirmed"
    assert len(results) == 1
    r = results[0]
    assert r.variable == "order_status"
    assert r.value == "confirmed"
    assert r.method == "direct"
    assert r.skipped is False


def test_llm_assign_pulls_from_take_exit_args():
    """A flow whose exit path harvests slots via llm-method assigns reads them
    from the take_exit_path tool args."""
    ep = ExitPath(
        id="xp_x",
        type="happy",
        condition=None,
        next_flow_id="flow_next",
        assigns={
            "drink_style": Assign(method="llm", value=None),
            "size": Assign(method="llm", value=None),
        },
    )
    bag: dict = {}
    results = assigns.fire(
        ep,
        bag,
        llm_results={"take_exit_path": {"exit_path_id": "xp_x", "drink_style": "latte", "size": "large"}},
    )
    assert bag == {"drink_style": "latte", "size": "large"}
    assert {r.variable for r in results} == {"drink_style", "size"}
    assert all(not r.skipped for r in results)


def test_llm_assign_missing_value_skipped():
    ep = ExitPath(
        id="xp_x",
        type="happy",
        condition=None,
        next_flow_id="flow_next",
        assigns={"drink_style": Assign(method="llm", value=None)},
    )
    bag: dict = {}
    results = assigns.fire(ep, bag, llm_results={"take_exit_path": {"exit_path_id": "xp_x"}})
    assert "drink_style" not in bag
    assert results[0].skipped is True
    assert results[0].method == "llm"


def test_calculation_assign_evaluates():
    ep = ExitPath(
        id="xp_x",
        type="happy",
        condition=None,
        next_flow_id="flow_next",
        assigns={"big_loan": Assign(method="calculation", value="loan_amount > 50000")},
    )
    bag = {"loan_amount": 75000}
    results = assigns.fire(ep, bag, llm_results={})
    assert bag["big_loan"] is True
    assert results[0].method == "calculation"
