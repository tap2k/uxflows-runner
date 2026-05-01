"""Calculation-method evaluator — the schema's deterministic substrate."""

from __future__ import annotations

import pytest

from uxflows_runner.dispatcher.expressions import (
    ExpressionError,
    evaluate,
    is_truthy,
    match_pattern,
)


def test_string_equality():
    assert evaluate('drink_type == "coffee"', {"drink_type": "coffee"}) is True
    assert evaluate('drink_type == "coffee"', {"drink_type": "tea"}) is False


def test_number_comparison():
    assert evaluate("loan_amount > 50000", {"loan_amount": 75000}) is True
    assert evaluate("loan_amount > 50000", {"loan_amount": 25000}) is False


def test_boolean_and_or_not():
    bag = {"verified": True, "amount": 60000}
    assert evaluate("verified == True and amount > 50000", bag) is True
    assert evaluate("verified == False or amount > 50000", bag) is True
    assert evaluate("not verified", bag) is False


def test_none_literal():
    assert evaluate("payment_received == None", {"payment_received": None}) is True
    assert evaluate("payment_received == None", {"payment_received": 100}) is False


def test_missing_variable_treated_as_none():
    # A condition referencing an unset variable should resolve cleanly to
    # "not satisfied" rather than blowing up — the runner needs to evaluate
    # exit-paths on a partially-populated bag.
    assert evaluate('drink_type == "coffee"', {}) is False
    assert evaluate("verified == True", {}) is False
    assert evaluate("payment_received == None", {}) is True


def test_is_truthy_coerces():
    assert is_truthy("verified", {"verified": True}) is True
    assert is_truthy("verified", {"verified": False}) is False
    assert is_truthy("verified", {}) is False  # missing => None => False


def test_no_function_calls_allowed():
    with pytest.raises(ExpressionError):
        evaluate('len("hi") == 2', {})


def test_no_attribute_access():
    # Access to attributes / methods should be blocked.
    with pytest.raises(ExpressionError):
        evaluate("foo.bar", {"foo": "hi"})


def test_pattern_match():
    bag = {"phone": "415-555-1234", "name": "skye"}
    assert match_pattern("phone", r"^\d{3}-\d{3}-\d{4}$", bag) is True
    assert match_pattern("phone", r"^\d{10}$", bag) is False


def test_pattern_match_missing_variable():
    assert match_pattern("phone", r".+", {}) is False


def test_pattern_match_non_string_value():
    assert match_pattern("amount", r"\d+", {"amount": 100}) is False


def test_pattern_match_bad_regex_raises():
    with pytest.raises(ExpressionError):
        match_pattern("name", r"[unclosed", {"name": "skye"})
