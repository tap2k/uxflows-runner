"""Calculation-method expression engine.

Stays tiny on purpose (RUNNER-PLAN §Risks: "expressions.py scope creep").
Supported per SCHEMA.md §"Calculation Expression Syntax":
  - Variable references (bare names) resolved from the variable bag
  - Comparison: == != > >= < <=
  - Logical: and / or / not
  - Literals: strings (double-quoted), numbers, True / False, None

Pattern-match subtype: when a node has method=="calculation" and a `pattern`
field, the expression is treated as a variable name and the pattern is a
regex applied to its current value. `match_pattern` covers that case.

`simpleeval` does the heavy lifting; we just configure it tightly. No imports,
no functions, no attribute access, no comprehensions — pure scalar math over
the variable bag.
"""

from __future__ import annotations

import re
from typing import Any

from simpleeval import (
    AttributeDoesNotExist,
    NameNotDefined,
    SimpleEval,
)


class ExpressionError(ValueError):
    """Raised when a calculation expression can't be evaluated. Distinct from
    'evaluates to False' — the dispatcher distinguishes 'not takeable' from
    'malformed' for diagnostics."""


_NAMES = {"True": True, "False": False, "None": None}


def evaluate(expression: str, variables: dict[str, Any]) -> Any:
    """Evaluate a calculation expression against the variable bag. Returns the
    raw value (caller decides truthiness). Raises ExpressionError on failure."""
    names = {**_NAMES, **variables}
    evaluator = SimpleEval(names=names)
    # Lock down the surface — no function calls, no attr access.
    evaluator.functions = {}
    try:
        return evaluator.eval(expression)
    except NameNotDefined as exc:
        # An unset variable evaluates to "no value" — surfaces as None so a
        # condition like `verified == True` cleanly returns False on a bag
        # missing `verified`. simpleeval raises before we see the comparison,
        # so we re-evaluate with the missing names defaulted to None.
        missing = exc.name
        return evaluate(expression, {**variables, missing: None})
    except AttributeDoesNotExist as exc:
        raise ExpressionError(f"attribute access not allowed: {exc}") from exc
    except SyntaxError as exc:
        raise ExpressionError(f"bad expression {expression!r}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 — simpleeval raises a few subclasses; normalize.
        raise ExpressionError(f"failed to evaluate {expression!r}: {exc}") from exc


def is_truthy(expression: str, variables: dict[str, Any]) -> bool:
    """Convenience: evaluate and coerce to bool. Used by routing.py for
    condition checks. Missing variables → falsy (handled by `evaluate`)."""
    return bool(evaluate(expression, variables))


def match_pattern(variable_name: str, pattern: str, variables: dict[str, Any]) -> bool:
    """Pattern-matching subtype of `calculation`. The 'expression' is the bare
    variable name; `pattern` is a regex applied via re.search to the current
    string value. Missing or non-string values match nothing."""
    value = variables.get(variable_name)
    if not isinstance(value, str):
        return False
    try:
        return re.search(pattern, value) is not None
    except re.error as exc:
        raise ExpressionError(f"bad regex pattern {pattern!r}: {exc}") from exc
