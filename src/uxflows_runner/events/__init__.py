"""Event schema + emitter — the runner-side contract with the editor canvas."""

from .emitter import EventEmitter, NullEventEmitter
from .schema import Event, EventEnvelope

__all__ = ["Event", "EventEnvelope", "EventEmitter", "NullEventEmitter"]
