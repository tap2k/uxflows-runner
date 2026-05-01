"""Fire an exit path's assigns block.

Called by the processor when an exit fires (NOT per-turn — variables in v0 are
only assigned on exit-path firing). Mutates the variable bag in place; returns
a list of (variable_name, value, method) triples the processor turns into
`variable_set` events.

LLM-method assigns read pre-computed values out of `llm_results` — the LLM
delivered them as parameters on the chosen `take_exit_path` tool call. If a
value is missing from the payload, the assign is skipped (logged elsewhere)
rather than written as None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from uxflows_runner.spec.types import ExitPath

from . import methods


@dataclass(frozen=True)
class AssignResult:
    variable: str
    value: Any
    method: str  # "llm" | "calculation" | "direct"
    skipped: bool = False  # true when llm-method assign came back empty


def fire(
    exit_path: ExitPath,
    variables: dict[str, Any],
    llm_results: dict[str, Any],
) -> list[AssignResult]:
    """Apply each assign on the exit path, mutating `variables` in place."""
    results: list[AssignResult] = []
    take_args = (llm_results.get("take_exit_path") or {})
    for var_name, assign in exit_path.assigns.items():
        # LLM assigns are nested as named parameters on take_exit_path —
        # the LLM emitted them when it chose this exit_path_id.
        per_var_llm = {var_name: take_args[var_name]} if var_name in take_args else {}
        resolved, value = methods.evaluate_assign(
            assign, variables, per_var_llm, llm_key=var_name
        )
        if not resolved:
            results.append(
                AssignResult(variable=var_name, value=None, method=assign.method, skipped=True)
            )
            continue
        variables[var_name] = value
        results.append(AssignResult(variable=var_name, value=value, method=assign.method))
    return results
