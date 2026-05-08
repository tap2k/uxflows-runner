"""Unit tests for event emitters — JsonlEventEmitter + MultiEventEmitter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uxflows_runner.events.emitter import (
    BufferingEventEmitter,
    JsonlEventEmitter,
    LoggingEventEmitter,
    MultiEventEmitter,
)
from uxflows_runner.events.schema import (
    FlowEntered,
    SessionEnded,
    SessionStarted,
    VariableSet,
)


def _ev_session_started(sid: str = "sess1") -> SessionStarted:
    return SessionStarted(
        session_id=sid, agent_id="agent_x", lang="en-US", spec_hash="h1"
    )


def _ev_flow_entered(sid: str = "sess1", flow: str = "flow_greet") -> FlowEntered:
    return FlowEntered(session_id=sid, flow_id=flow, via="entry")


def _ev_variable_set(sid: str = "sess1") -> VariableSet:
    return VariableSet(
        session_id=sid,
        variable_name="drink_type",
        value="coffee",
        method="direct",
        source_flow_id="flow_greet",
        source_exit_path_id="xp_greet_to_coffee",
    )


def _ev_session_ended(sid: str = "sess1") -> SessionEnded:
    return SessionEnded(session_id=sid, reason="user_stop")


# --- JsonlEventEmitter ---------------------------------------------------


def test_jsonl_emitter_writes_one_line_per_event(tmp_path: Path):
    path = tmp_path / "sess.jsonl"
    em = JsonlEventEmitter(path)

    em.emit(_ev_session_started())
    em.emit(_ev_flow_entered())
    em.emit(_ev_variable_set())
    em.emit(_ev_session_ended())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4

    parsed = [json.loads(line) for line in lines]
    assert [p["type"] for p in parsed] == [
        "session_started",
        "flow_entered",
        "variable_set",
        "session_ended",
    ]
    assert parsed[2]["variable_name"] == "drink_type"
    assert parsed[2]["value"] == "coffee"
    # Every record carries session_id + ts
    assert all("session_id" in p and "ts" in p for p in parsed)


def test_jsonl_emitter_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "sess.jsonl"
    em = JsonlEventEmitter(nested)
    em.emit(_ev_session_started())
    assert nested.is_file()
    assert nested.parent.is_dir()


def test_jsonl_emitter_closes_on_session_ended(tmp_path: Path):
    path = tmp_path / "sess.jsonl"
    em = JsonlEventEmitter(path)
    em.emit(_ev_session_started())
    assert em._fh is not None  # noqa: SLF001
    em.emit(_ev_session_ended())
    assert em._fh is None  # noqa: SLF001
    # File still readable after close — buffered content was flushed.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_jsonl_emitter_disables_silently_on_unwritable_path(tmp_path: Path, caplog):
    # Point at a path whose parent is a regular file → mkdir(parents) fails.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("hi")
    bad_path = blocker / "sess.jsonl"

    em = JsonlEventEmitter(bad_path)
    # Should NOT raise — degrades to no-op.
    em.emit(_ev_session_started())
    em.emit(_ev_flow_entered())
    assert em._broken is True  # noqa: SLF001
    assert em._fh is None  # noqa: SLF001


# --- MultiEventEmitter ---------------------------------------------------


def test_multi_emitter_fans_out_to_all_children(tmp_path: Path):
    buffer = BufferingEventEmitter()
    jsonl = JsonlEventEmitter(tmp_path / "sess.jsonl")
    multi = MultiEventEmitter([buffer, jsonl])

    multi.emit(_ev_session_started())
    multi.emit(_ev_flow_entered())

    # Buffering side
    drained = buffer.drain()
    assert len(drained) == 2
    assert drained[0].type == "session_started"

    # JSONL side
    lines = (tmp_path / "sess.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_multi_emitter_isolates_child_failures():
    """One child raising must not stop the others from receiving the event."""

    class Bomb:
        def emit(self, event):
            raise RuntimeError("boom")

    buffer = BufferingEventEmitter()
    multi = MultiEventEmitter([Bomb(), buffer])

    multi.emit(_ev_session_started())
    drained = buffer.drain()
    assert len(drained) == 1


def test_multi_emitter_with_logging_does_not_raise(tmp_path: Path):
    """Smoke test — the production combo (Logging + Jsonl) wires up."""
    multi = MultiEventEmitter(
        [LoggingEventEmitter(), JsonlEventEmitter(tmp_path / "sess.jsonl")]
    )
    multi.emit(_ev_session_started())
    multi.emit(_ev_session_ended())
    lines = (tmp_path / "sess.jsonl").read_text().splitlines()
    assert len(lines) == 2
