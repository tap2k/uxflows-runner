"""Capability dispatch — fire post-exit actions.

v0 surface:
  - kind: "function" → HTTP POST with implicit input resolution: read
    capabilities[name].inputs and pull those variables from the bag at fire
    time. JSON body, optional headers from execution config.
  - kind: "retrieval" → stub returning an empty context.

Execution config is *not* part of the spec (RUNNER-PLAN: "API keys, endpoints,
voice IDs → execution config, sibling to spec, never inside"). For v0 it's a
JSON file alongside the spec, structure:

  {
    "capabilities": {
      "place_order": {"url": "https://...", "headers": {"Authorization": "..."}}
    }
  }

Missing entry = the capability isn't actually wired up; we log a warning,
emit `capability_invoked` + `capability_returned{error}` for the event stream,
and move on. The conversation does not block.

Fire-and-forget: invoke() returns immediately after scheduling the call. The
`capability_returned` event arrives later. v0 ordering across multiple
actions on a single exit is best-effort (RUNNER-PLAN line 281).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from loguru import logger

from uxflows_runner.spec.loader import LoadedSpec
from uxflows_runner.spec.types import Capability


@dataclass(frozen=True)
class CapabilityEndpoint:
    url: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityInvocation:
    """Argument record for `capability_invoked` events."""

    capability_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class CapabilityResult:
    """Outcome record for `capability_returned` events."""

    capability_name: str
    result: Any | None = None
    error: str | None = None


def load_execution_config(path: str | Path) -> dict[str, CapabilityEndpoint]:
    """Read a sibling execution config; return name -> endpoint. Missing file
    is fine (returns {}); the runner just warns and proceeds with stubs."""
    p = Path(path)
    if not p.exists():
        return {}
    import json

    raw = json.loads(p.read_text())
    out: dict[str, CapabilityEndpoint] = {}
    for name, entry in (raw.get("capabilities") or {}).items():
        out[name] = CapabilityEndpoint(
            url=entry["url"],
            headers=entry.get("headers", {}),
        )
    return out


def resolve_inputs(capability: Capability, variables: dict[str, Any]) -> dict[str, Any]:
    """Pull declared inputs from the variable bag. Missing values are simply
    omitted (the receiving service can validate). RUNNER-PLAN §"Schema coverage":
    'no explicit input binding syntax'."""
    return {name: variables[name] for name in capability.inputs if name in variables}


# Type for the on-result callback the processor passes in to receive
# capability_returned events. Sync to keep the call site simple.
ResultCallback = Callable[[CapabilityResult], None]


class CapabilityDispatcher:
    """Owns a per-session httpx client. Schedule fire-and-forget invocations
    that emit results back via callback when they complete."""

    def __init__(
        self,
        spec: LoadedSpec,
        endpoints: dict[str, CapabilityEndpoint],
        on_result: ResultCallback | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._spec = spec
        self._endpoints = endpoints
        self._on_result = on_result or (lambda _result: None)
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._tasks: set[asyncio.Task[None]] = set()

    async def aclose(self) -> None:
        # Wait for in-flight tasks so capability_returned events still fire,
        # then close the client.
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    def invoke(
        self,
        capability_id: str,
        variables: dict[str, Any],
    ) -> CapabilityInvocation:
        """Look up capability by id (the schema's reference key on actions),
        resolve inputs, schedule the call. Returns the invocation record so
        the processor can emit `capability_invoked` immediately."""
        cap = self._spec.capabilities_by_id.get(capability_id)
        if cap is None:
            logger.warning(f"capability_id={capability_id!r} not in catalog; skipping")
            return CapabilityInvocation(capability_name=capability_id, args={})

        args = resolve_inputs(cap, variables)
        invocation = CapabilityInvocation(capability_name=cap.name, args=args)

        task = asyncio.create_task(self._run(cap, args))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return invocation

    async def _run(self, cap: Capability, args: dict[str, Any]) -> None:
        try:
            if cap.kind == "retrieval":
                # v0 stub — RUNNER-PLAN §"Schema coverage": "kind: retrieval →
                # stub returning empty context."
                self._on_result(CapabilityResult(capability_name=cap.name, result={"context": []}))
                return

            endpoint = self._endpoints.get(cap.name)
            if endpoint is None:
                self._on_result(
                    CapabilityResult(
                        capability_name=cap.name,
                        error=f"no endpoint configured for {cap.name!r}",
                    )
                )
                return

            response = await self._client.post(
                endpoint.url, json=args, headers=endpoint.headers
            )
            response.raise_for_status()
            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            self._on_result(CapabilityResult(capability_name=cap.name, result=payload))
        except Exception as exc:  # noqa: BLE001 — capture and surface as event
            logger.exception(f"capability {cap.name} dispatch failed")
            self._on_result(CapabilityResult(capability_name=cap.name, error=str(exc)))


# Sync helper for tests / non-async contexts.
def make_invocation(
    spec: LoadedSpec, capability_id: str, variables: dict[str, Any]
) -> CapabilityInvocation:
    cap = spec.capabilities_by_id[capability_id]
    return CapabilityInvocation(capability_name=cap.name, args=resolve_inputs(cap, variables))
