"""End-to-end smoke for the FNOL example — locks in the demo path that
drove the capability-output-binding + mock_returns design.

Drives the real dispatcher (Session, FlowState, routing, capabilities,
processor.apply_route) against ../uxflows/public/fnol.json without the
LLM in the loop. Two paths exercised:

  - Happy: mock_returns gives verify_policy a positive policy. Firing
    flow_identify's xp_id_verify exit lands three variables in scope and
    the next plan's calc shortcut routes through flow_route_verified into
    flow_incident_details.
  - Sad: mock_returns flips policy_active false. Same exit + same plan
    shortcuts the other way, into flow_policy_not_found.

If either of these regresses, the FNOL demo silently breaks. This is the
backstop.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pipecat.processors.aggregators.llm_context import LLMContext

from uxflows_runner.dispatcher.capabilities import CapabilityDispatcher
from uxflows_runner.dispatcher.processor import apply_route, plan_for_active_flow
from uxflows_runner.dispatcher.routing_protocol import RouteTag
from uxflows_runner.dispatcher.session import Session
from uxflows_runner.events.emitter import BufferingEventEmitter
from uxflows_runner.events.schema import (
    CapabilityInvoked,
    CapabilityReturned,
    ExitPathTaken,
    FlowEntered,
    VariableSet,
)
from uxflows_runner.spec.loader import load_spec

FNOL = Path(__file__).resolve().parents[2] / "uxflows" / "public" / "fnol.json"


@pytest.fixture(scope="module")
def fnol_spec():
    return load_spec(FNOL)


def _build_session(
    fnol_spec, mock_returns: dict[str, dict] | None = None
) -> tuple[Session, BufferingEventEmitter, CapabilityDispatcher]:
    """Construct a Session positioned at flow_identify with caller info seeded
    and a current_plan ready, so a route tag can fire."""
    dispatcher = CapabilityDispatcher(
        spec=fnol_spec, endpoints={}, mock_returns=mock_returns
    )
    events = BufferingEventEmitter()
    context = LLMContext(messages=[{"role": "system", "content": ""}])
    session = Session.start(
        spec=fnol_spec,
        llm_context=context,
        events=events,
        capabilities=dispatcher,
    )
    session.state.transition("flow_identify")
    session.state.variables["caller_name"] = "Tapan Parikh"
    session.state.variables["policy_number"] = "NW123456"
    plan_for_active_flow(session)
    events.drain()  # discard setup events; tests inspect only post-route events
    return session, events, dispatcher


@pytest.mark.asyncio
async def test_fnol_happy_path_binds_outputs_and_shortcuts_to_incident_details(
    fnol_spec,
):
    mock_returns = {
        "verify_policy": {
            "policy_active": True,
            "deductible_amount": 500,
            "named_drivers": "Casey Lin, Pat Lin",
        }
    }
    session, events, dispatcher = _build_session(fnol_spec, mock_returns)

    await apply_route(session, RouteTag(exit="xp_id_verify"))

    # Outputs bound into scope on the action fire.
    vars_ = session.state.variables
    assert vars_["policy_active"] is True
    assert vars_["deductible_amount"] == 500
    assert vars_["named_drivers"] == "Casey Lin, Pat Lin"

    # We're now in the utility flow that calc-routes on policy_active.
    assert session.state.active_flow_id == "flow_route_verified"

    # Three variable_set events with method=capability — one per declared output.
    var_sets = [e for e in events.drain() if isinstance(e, VariableSet)]
    by_name = {e.variable_name: e for e in var_sets}
    assert set(by_name) == {"policy_active", "deductible_amount", "named_drivers"}
    for ev in by_name.values():
        assert ev.method == "capability"
        assert ev.source_flow_id == "flow_identify"
        assert ev.source_exit_path_id == "xp_id_verify"

    # Next plan should resolve a calc shortcut directly into flow_incident_details.
    plan_for_active_flow(session)
    sc = session.current_plan.shortcut
    assert sc is not None, "utility flow should produce a calc shortcut"
    assert sc.target_flow_id == "flow_incident_details"

    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_fnol_sad_path_routes_to_policy_not_found(fnol_spec):
    mock_returns = {"verify_policy": {"policy_active": False}}
    session, _events, dispatcher = _build_session(fnol_spec, mock_returns)

    await apply_route(session, RouteTag(exit="xp_id_verify"))

    assert session.state.variables["policy_active"] is False
    assert session.state.active_flow_id == "flow_route_verified"

    plan_for_active_flow(session)
    sc = session.current_plan.shortcut
    assert sc is not None and sc.target_flow_id == "flow_policy_not_found"

    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_fnol_no_mock_undefined_outputs_also_routes_to_policy_not_found(
    fnol_spec,
):
    """No mock configured → verify_policy is kind=retrieval, so the v0 stub
    returns `{"context": []}` (not an error). None of its declared outputs
    are in that dict, so nothing binds — policy_active stays undefined,
    which the calc condition `policy_active != True` catches alongside False."""
    session, events, dispatcher = _build_session(fnol_spec, mock_returns=None)

    await apply_route(session, RouteTag(exit="xp_id_verify"))

    assert "policy_active" not in session.state.variables

    # No variable_set events fired — outputs didn't land because the retrieval
    # stub's payload has no matching keys.
    drained = events.drain()
    assert not [e for e in drained if isinstance(e, VariableSet)]

    plan_for_active_flow(session)
    sc = session.current_plan.shortcut
    assert sc is not None and sc.target_flow_id == "flow_policy_not_found"

    await dispatcher.aclose()


@pytest.mark.asyncio
async def test_fnol_event_ordering_through_one_exit(fnol_spec):
    """Spot-check that events fire in the right order on a single exit:
    capability_invoked → capability_returned → variable_set(s) → exit_path_taken
    → flow_exited → flow_entered. Regression catch for the dispatch loop."""
    mock_returns = {
        "verify_policy": {"policy_active": True, "deductible_amount": 1000}
    }
    session, events, dispatcher = _build_session(fnol_spec, mock_returns)

    await apply_route(session, RouteTag(exit="xp_id_verify"))

    types = [type(e).__name__ for e in events.drain()]
    # capability_invoked must precede capability_returned must precede any
    # variable_set bindings (you can't bind until you've returned).
    assert types.index("CapabilityInvoked") < types.index("CapabilityReturned")
    assert types.index("CapabilityReturned") < types.index("VariableSet")
    # variable_set events must precede exit_path_taken (binding writes
    # contribute to the trace before we declare the exit taken).
    assert types.index("VariableSet") < types.index("ExitPathTaken")
    # flow_entered fires last — the new flow is active.
    assert types.index("ExitPathTaken") < types.index("FlowEntered")

    await dispatcher.aclose()
