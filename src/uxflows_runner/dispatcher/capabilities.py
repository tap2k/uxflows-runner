"""Capability dispatch — fire post-exit actions.

v0 surface:
  - kind: "function" → HTTP POST with implicit input resolution: read
    capabilities[name].inputs and pull those variables from the bag at fire
    time. JSON body, optional headers from execution config.
  - kind: "retrieval" → stub returning an empty context.

Three sources of capability results, checked in order on each invoke:

  1. **Session mock_returns** — a per-session dict keyed by `capability.name`
     mapping to an output dict. Shadows endpoints when present. Mirrors the
     `context_vars` pattern: design-time scaffolding for simulation, NOT in
     the spec; threaded through the session-start payload from the editor's
     SimulatePanel. Lets a designer probe happy / sad / unusual capability
     returns without standing up a mock server.
  2. **Execution-config endpoints** — sibling JSON, keyed by capability name.
     For real backends in production.
  3. **No source configured** — return an error result; outputs simply don't
     land in variable scope.

Execution config is *not* part of the spec (RUNNER-PLAN: "API keys, endpoints,
voice IDs → execution config, sibling to spec, never inside"). For v0 it's a
JSON file alongside the spec, structure:

  {
    "capabilities": {
      "place_order": {"url": "https://...", "headers": {"Authorization": "..."}}
    }
  }

Dispatch is synchronous: `invoke` awaits the HTTP call (or returns the mock)
and returns the result so the caller can bind declared `outputs` into variable
scope before the flow transitions (RUNNER-PLAN §"Capability outputs bind to
variable scope"). The caller is responsible for emitting `capability_invoked`
/ `capability_returned` / `variable_set` events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


class CapabilityDispatcher:
    """Owns a per-session httpx client. Synchronously dispatches capabilities
    on exit-path fire so callers can bind declared outputs before the
    transition completes."""

    def __init__(
        self,
        spec: LoadedSpec,
        endpoints: dict[str, CapabilityEndpoint],
        *,
        mock_returns: dict[str, dict[str, Any]] | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._spec = spec
        self._endpoints = endpoints
        # Per-session simulation fixtures, keyed by capability NAME (not id).
        # Mirrors the context_vars pattern — sent via session-start payload,
        # never lives in the spec.
        self._mock_returns = mock_returns or {}
        # Catch designer typos early — a mock keyed by id (instead of name) or
        # a misspelled name would silently fall through to "no endpoint
        # configured" with no hint that the mock didn't apply.
        known_names = set(spec.capabilities_by_name.keys())
        for cap_name in self._mock_returns:
            if cap_name not in known_names:
                logger.warning(
                    f"mock_returns has unknown capability name {cap_name!r}; "
                    f"valid names: {sorted(known_names)}"
                )
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def invoke(
        self,
        capability_id: str,
        variables: dict[str, Any],
    ) -> tuple[CapabilityInvocation, CapabilityResult]:
        """Look up capability by id (the schema's reference key on actions),
        resolve inputs, await dispatch. Returns (invocation, result) so the
        caller can emit `capability_invoked` / `capability_returned` events
        and bind `outputs` into variable scope."""
        cap = self._spec.capabilities_by_id.get(capability_id)
        if cap is None:
            logger.warning(f"capability_id={capability_id!r} not in catalog; skipping")
            invocation = CapabilityInvocation(capability_name=capability_id, args={})
            return invocation, CapabilityResult(
                capability_name=capability_id, error=f"unknown capability_id {capability_id!r}"
            )

        args = resolve_inputs(cap, variables)
        invocation = CapabilityInvocation(capability_name=cap.name, args=args)
        result = await self._run(cap, args)
        return invocation, result

    async def _run(self, cap: Capability, args: dict[str, Any]) -> CapabilityResult:
        # Session-mock fixture shadows everything else. Designer's SimulatePanel
        # sets these; production deployments don't.
        mock = self._mock_returns.get(cap.name)
        if mock is not None:
            # Catch the "mock value didn't land" footgun: warn when the mock
            # dict has keys not in the capability's declared outputs (those
            # keys won't bind to anything downstream). Don't strip them —
            # surface as a warning so the designer sees the mismatch.
            stray = set(mock) - set(cap.outputs)
            if stray:
                logger.warning(
                    f"mock_returns[{cap.name!r}] has keys not in declared "
                    f"outputs: {sorted(stray)} (declared: {list(cap.outputs)})"
                )
            return CapabilityResult(capability_name=cap.name, result=dict(mock))

        try:
            if cap.kind == "retrieval":
                # v0 stub — real retrieval lands with knowledge tables in v0.5.
                return CapabilityResult(capability_name=cap.name, result={"context": []})

            endpoint = self._endpoints.get(cap.name)
            if endpoint is None:
                return CapabilityResult(
                    capability_name=cap.name,
                    error=f"no endpoint configured for {cap.name!r}",
                )

            response = await self._client.post(
                endpoint.url, json=args, headers=endpoint.headers
            )
            response.raise_for_status()
            payload: Any
            try:
                payload = response.json()
            except ValueError:
                payload = response.text
            return CapabilityResult(capability_name=cap.name, result=payload)
        except Exception as exc:  # noqa: BLE001 — capture and surface to caller
            logger.exception(f"capability {cap.name} dispatch failed")
            return CapabilityResult(capability_name=cap.name, error=str(exc))


# Sync helper for tests / non-async contexts.
def make_invocation(
    spec: LoadedSpec, capability_id: str, variables: dict[str, Any]
) -> CapabilityInvocation:
    cap = spec.capabilities_by_id[capability_id]
    return CapabilityInvocation(capability_name=cap.name, args=resolve_inputs(cap, variables))
