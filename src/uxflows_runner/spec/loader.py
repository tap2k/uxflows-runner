"""Spec loading + indexing. Validate JSON, build lookup tables.

The dispatcher reads through `LoadedSpec`, never the raw JSON. Anything the
dispatcher needs O(1) access to during a turn (flow by id, capability by name,
interrupts by scope) is precomputed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import Agent, Capability, Flow, Spec


@dataclass(frozen=True)
class LoadedSpec:
    agent: Agent
    flows_by_id: dict[str, Flow]
    capabilities_by_name: dict[str, Capability]
    capabilities_by_id: dict[str, Capability]
    interrupts_by_scope: dict[str, list[Flow]]  # flow_id -> applicable interrupts; "__global__" -> globals
    spec_hash: str = ""

    @property
    def entry_flow(self) -> Flow:
        return self.flows_by_id[self.agent.entry_flow_id]


GLOBAL_SCOPE_KEY = "__global__"


def load_spec(path: str | Path) -> LoadedSpec:
    raw = Path(path).read_text()
    return parse_spec(raw)


def parse_spec(raw: str) -> LoadedSpec:
    data = json.loads(raw)
    spec = Spec.model_validate(data)
    return _index(spec, raw)


def _index(spec: Spec, raw: str) -> LoadedSpec:
    flows_by_id: dict[str, Flow] = {}
    for flow in spec.flows:
        if flow.id in flows_by_id:
            raise ValueError(f"duplicate flow id: {flow.id}")
        flows_by_id[flow.id] = flow

    if spec.agent.entry_flow_id not in flows_by_id:
        raise ValueError(
            f"agent.entry_flow_id={spec.agent.entry_flow_id!r} not present in flows[]"
        )

    capabilities_by_name: dict[str, Capability] = {}
    capabilities_by_id: dict[str, Capability] = {}
    for cap in spec.agent.capabilities:
        if cap.name in capabilities_by_name:
            raise ValueError(f"duplicate capability name: {cap.name}")
        if cap.id in capabilities_by_id:
            raise ValueError(f"duplicate capability id: {cap.id}")
        capabilities_by_name[cap.name] = cap
        capabilities_by_id[cap.id] = cap

    # Validate referential integrity: actions reference capabilities by id.
    for flow in spec.flows:
        for ep in flow.routing.exit_paths:
            if ep.next_flow_id is not None and ep.next_flow_id not in flows_by_id:
                raise ValueError(
                    f"flow {flow.id} exit_path {ep.id}: next_flow_id={ep.next_flow_id!r} not found"
                )
            for action in ep.actions:
                if action.capability_id not in capabilities_by_id:
                    raise ValueError(
                        f"flow {flow.id} exit_path {ep.id}: capability_id={action.capability_id!r} not in catalog"
                    )

    interrupts_by_scope: dict[str, list[Flow]] = {GLOBAL_SCOPE_KEY: []}
    for flow in spec.flows:
        if flow.type != "interrupt":
            continue
        scope = flow.scope or []
        if scope == ["global"]:
            interrupts_by_scope[GLOBAL_SCOPE_KEY].append(flow)
            continue
        for caller_id in scope:
            if caller_id not in flows_by_id:
                raise ValueError(
                    f"interrupt flow {flow.id} scope references unknown flow {caller_id!r}"
                )
            interrupts_by_scope.setdefault(caller_id, []).append(flow)

    import hashlib

    spec_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    return LoadedSpec(
        agent=spec.agent,
        flows_by_id=flows_by_id,
        capabilities_by_name=capabilities_by_name,
        capabilities_by_id=capabilities_by_id,
        interrupts_by_scope=interrupts_by_scope,
        spec_hash=spec_hash,
    )


def applicable_interrupts(loaded: LoadedSpec, active_flow_id: str) -> list[Flow]:
    """Interrupts whose scope matches the active flow OR is global. Per
    RUNNER-PLAN: scope matches against top-of-stack flow only."""
    return [
        *loaded.interrupts_by_scope.get(GLOBAL_SCOPE_KEY, []),
        *loaded.interrupts_by_scope.get(active_flow_id, []),
    ]
