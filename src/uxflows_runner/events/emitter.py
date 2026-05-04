"""Event emitter — buffers events on a queue, fans out to subscribers.

Phase 1: in-process queue, single subscriber pattern (CLI logs). Phase 2
adds the SSE broker that turns this into a multi-consumer stream.

The dispatcher only ever calls `emit(event)` — no awareness of who's listening.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

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
