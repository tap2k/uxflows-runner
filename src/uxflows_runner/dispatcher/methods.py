"""Three-method evaluator — the thin layer that fans out to expressions /
literals / pre-collected LLM tool-call results.

The runner never makes ad-hoc LLM calls inside this module. The "one LLM call
per turn" rule means LLM-method work is *collected* during routing (`pending`
slots in the per-turn tool schema) and then *consumed* here from the LLM
response payload. So `evaluate_*` takes an `llm_results` dict — the parsed
tool-call args from this turn's response — and reads pre-resolved values out
of it. If the LLM didn't supply a value for an LLM-method node, callers treat
that as "condition not satisfied" / "assign skipped".
"""

from __future__ import annotations

from typing import Any

from uxflows_runner.spec.types import Assign, Condition

from . import expressions


class MethodError(ValueError):
    """Malformed method spec — e.g. method=='direct' with no `value`."""


def evaluate_condition(
    condition: Condition,
    variables: dict[str, Any],
    llm_results: dict[str, Any],
    *,
    llm_key: str,
) -> bool:
    """True if the condition is satisfied. `llm_key` identifies this condition
    in the LLM response payload (e.g. exit_path id, interrupt flow id) so
    multiple LLM-method conditions on a single turn can be disambiguated."""
    if condition.method == "direct":
        # `direct` on a condition is unconditional — the schema uses it for
        # always-takeable exits. The expression is annotation-only.
        return True

    if condition.method == "calculation":
        if condition.pattern is not None:
            # Pattern subtype: expression is a bare variable name.
            return expressions.match_pattern(condition.expression, condition.pattern, variables)
        return expressions.is_truthy(condition.expression, variables)

    if condition.method == "llm":
        # LLM picked this key (e.g. via `take_exit_path` tool args) iff the key
        # appears in the results bag.
        return llm_key in llm_results

    raise MethodError(f"unknown method: {condition.method!r}")


def evaluate_assign(
    assign: Assign,
    variables: dict[str, Any],
    llm_results: dict[str, Any],
    *,
    llm_key: str,
) -> tuple[bool, Any]:
    """Resolve an assign's value. Returns (resolved, value).
    `resolved=False` means the value is unavailable (e.g. LLM didn't supply
    one) — caller skips the variable_set rather than writing None."""
    if assign.method == "direct":
        if assign.value is None:
            # value=None is meaningful (the schema permits it), but we still
            # require the field to be *present*. pydantic gives us None for
            # both "field absent" and "field set to null" — accept both.
            pass
        return True, assign.value

    if assign.method == "calculation":
        if assign.pattern is not None:
            # Pattern subtype on an assign: same shape as on a condition —
            # `value` is the variable name to test, `pattern` is the regex,
            # result is bool.
            if not isinstance(assign.value, str):
                raise MethodError(
                    "calculation assign with `pattern` requires `value` to be a variable name string"
                )
            return True, expressions.match_pattern(assign.value, assign.pattern, variables)
        if not isinstance(assign.value, str):
            raise MethodError(
                "calculation assign requires `value` to be an expression string"
            )
        return True, expressions.evaluate(assign.value, variables)

    if assign.method == "llm":
        if llm_key in llm_results:
            return True, llm_results[llm_key]
        return False, None

    raise MethodError(f"unknown method: {assign.method!r}")
