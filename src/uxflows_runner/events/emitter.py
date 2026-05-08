"""Event emitter — buffers events on a queue, fans out to subscribers.

Phase 1: in-process queue, single subscriber pattern (CLI logs). Phase 2
adds the SSE broker that turns this into a multi-consumer stream.

The dispatcher only ever calls `emit(event)` — no awareness of who's listening.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import IO, Protocol

from loguru import logger

from .schema import Event


class EventEmitter(Protocol):
    """Anything that accepts dispatcher events. Concrete impls: log to CLI,
    push to SSE, append to JSONL, etc."""

    def emit(self, event: Event) -> None: ...


class NullEventEmitter:
    """No-op emitter for tests + early dev."""

    def emit(self, event: Event) -> None:  # noqa: D401
        return


class LoggingEventEmitter:
    """Phase-1 default — logs events at INFO level. Easy to grep, no stream
    machinery needed yet."""

    def emit(self, event: Event) -> None:
        logger.info("[event] {} {}", event.type, event.model_dump_json(exclude={"session_id", "ts"}))


class QueueEventEmitter:
    """Phase-2 ready: pushes events onto an asyncio.Queue. The /events SSE
    handler will pop them and stream them to subscribers. Multiple subscribers
    later get their own queue via a small fan-out broker."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Event] = asyncio.Queue()

    def emit(self, event: Event) -> None:
        self.queue.put_nowait(event)


class BufferingEventEmitter:
    """Phase-1.5 (text adapter): collect events in a list, drain on demand.

    Used by TextSession to return the events fired during a turn alongside
    the agent's reply in a single JSON response. `drain()` swaps the buffer
    for a fresh list and returns the old one — caller-owned, atomic.
    """

    def __init__(self) -> None:
        self._buffer: list[Event] = []

    def emit(self, event: Event) -> None:
        self._buffer.append(event)

    def drain(self) -> list[Event]:
        out, self._buffer = self._buffer, []
        return out


class JsonlEventEmitter:
    """v0.5 — appends events to a JSONL file, one record per line.

    File is opened lazily on the first `emit()` and closed on the first
    `session_ended` event (which is always the last event in a session).
    Synchronous append; events are small + low rate, so the cost is
    negligible vs. queuing/aiofiles.

    Caller owns directory creation only via the path's parent. Bad-path
    failures (permission, disk full) are logged and the emitter degrades
    to a no-op — never raises into the dispatcher.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: IO[str] | None = None
        self._broken = False

    def emit(self, event: Event) -> None:
        if self._broken:
            return
        try:
            if self._fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(event.model_dump_json() + "\n")
            self._fh.flush()
        except OSError as exc:
            logger.warning("JsonlEventEmitter disabled for {}: {}", self.path, exc)
            self._broken = True
            self._safe_close()
            return

        if event.type == "session_ended":
            self._safe_close()

    def _safe_close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


class MultiEventEmitter:
    """Fan-out wrapper: forwards each `emit()` to a list of child emitters.

    Used to tee events to multiple sinks (e.g. Logging + Jsonl, or
    Buffering + Jsonl). Children are invoked in order; one child raising
    does not prevent the others from receiving the event.
    """

    def __init__(self, emitters: list[EventEmitter]) -> None:
        self.emitters = emitters

    def emit(self, event: Event) -> None:
        for em in self.emitters:
            try:
                em.emit(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("emitter {} failed: {}", type(em).__name__, exc)
