"""Spec loading + indexing. Validate JSON, build lookup tables.

The dispatcher reads through `LoadedSpec`, never the raw JSON. Anything the
dispatcher needs O(1) access to during a turn (flow by id, capability by name,
applicable interrupts) is precomputed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import (
    GOTO_END,
    GOTO_RETURN,
    Agent,
    Capability,
    Flow,
    Spec,
    is_end_goto,
    is_flow_goto,
    is_return_goto,
)


@dataclass(frozen=True)
class LoadedSpec:
    agent: Agent
    flows_by_id: dict[str, Flow]
    capabilities_by_name: dict[str, Capability]
    capabilities_by_id: dict[str, Capability]
    # Interrupts (`type: "interrupt"`) are implicitly globally callable per the
    # new schema. Kept as a single list for O(1) per-turn iteration.
    global_interrupts: list[Flow]
    spec_hash: str = ""

    @property
    def entry_flow(self) -> Flow:
        return self.flows_by_id[self.agent.entry_flow_id]


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
        if flow.id in (GOTO_END, GOTO_RETURN):
            raise ValueError(
                f"flow id {flow.id!r} shadows a reserved goto keyword"
            )
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

    # Validate referential integrity: goto must point at END / RETURN / known
    # flow id; actions reference capabilities by id.
    for flow in spec.flows:
        for ep in flow.exit_paths:
            goto = ep.goto
            if is_end_goto(goto) or is_return_goto(goto):
                pass
            elif is_flow_goto(goto):
                if goto not in flows_by_id:
                    raise ValueError(
                        f"flow {flow.id} exit_path {ep.id}: goto={goto!r} not found"
                    )
            for action in ep.actions:
                if action.capability_id not in capabilities_by_id:
                    raise ValueError(
                        f"flow {flow.id} exit_path {ep.id}: capability_id={action.capability_id!r} not in catalog"
                    )

    # Interrupts are implicitly globally callable.
    global_interrupts: list[Flow] = [
        flow for flow in spec.flows if flow.type == "interrupt"
    ]

    import hashlib

    spec_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    return LoadedSpec(
        agent=spec.agent,
        flows_by_id=flows_by_id,
        capabilities_by_name=capabilities_by_name,
        capabilities_by_id=capabilities_by_id,
        global_interrupts=global_interrupts,
        spec_hash=spec_hash,
    )


def applicable_interrupts(loaded: LoadedSpec) -> list[Flow]:
    """Interrupts available at the current turn. All interrupts are implicitly
    globally callable; the active flow id does not narrow the list."""
    return list(loaded.global_interrupts)
