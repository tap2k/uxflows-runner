"""Three-method evaluator — thin layer, but the contract here is what
routing.py and assigns.py both lean on."""

from __future__ import annotations

import pytest

from uxflows_runner.dispatcher import methods
from uxflows_runner.spec.types import Assign, Condition


def cond(method, expression, pattern=None):
    return Condition(method=method, expression=expression, pattern=pattern)


def test_direct_condition_always_true():
    assert methods.evaluate_condition(cond("direct", "noop"), {}, {}, llm_key="x") is True


def test_calculation_condition():
    assert methods.evaluate_condition(
        cond("calculation", 'drink_type == "coffee"'),
        {"drink_type": "coffee"},
        {},
        llm_key="ep1",
    ) is True
    assert methods.evaluate_condition(
        cond("calculation", 'drink_type == "coffee"'),
        {"drink_type": "tea"},
        {},
        llm_key="ep1",
    ) is False


def test_calculation_pattern_subtype():
    c = cond("calculation", "phone", pattern=r"^\d{3}-\d{4}$")
    assert methods.evaluate_condition(c, {"phone": "555-1234"}, {}, llm_key="ep1") is True
    assert methods.evaluate_condition(c, {"phone": "abc"}, {}, llm_key="ep1") is False


def test_llm_condition_present_in_results():
    c = cond("llm", "Patron is walking away.")
    assert methods.evaluate_condition(c, {}, {"xp_walkaway": {}}, llm_key="xp_walkaway") is True
    assert methods.evaluate_condition(c, {}, {}, llm_key="xp_walkaway") is False


def test_unknown_method_raises():
    with pytest.raises(methods.MethodError):
        methods.evaluate_condition(
            Condition.model_construct(method="bogus", expression="x"),
            {}, {}, llm_key="x",
        )


def test_direct_assign_value():
    a = Assign(method="direct", value="confirmed")
    resolved, value = methods.evaluate_assign(a, {}, {}, llm_key="status")
    assert resolved is True
    assert value == "confirmed"


def test_direct_assign_value_can_be_none():
    a = Assign(method="direct", value=None)
    resolved, value = methods.evaluate_assign(a, {}, {}, llm_key="x")
    assert resolved is True
    assert value is None


def test_calculation_assign():
    a = Assign(method="calculation", value="loan_amount > 50000")
    resolved, value = methods.evaluate_assign(
        a, {"loan_amount": 75000}, {}, llm_key="big_loan"
    )
    assert resolved is True
    assert value is True


def test_llm_assign_present():
    a = Assign(method="llm", value=None)
    resolved, value = methods.evaluate_assign(
        a, {}, {"drink_style": "latte"}, llm_key="drink_style"
    )
    assert resolved is True
    assert value == "latte"


def test_llm_assign_missing_returns_unresolved():
    a = Assign(method="llm", value=None)
    resolved, value = methods.evaluate_assign(a, {}, {}, llm_key="drink_style")
    assert resolved is False
    assert value is None


def test_calculation_assign_value_must_be_string():
    a = Assign(method="calculation", value=123)
    with pytest.raises(methods.MethodError):
        methods.evaluate_assign(a, {}, {}, llm_key="x")
