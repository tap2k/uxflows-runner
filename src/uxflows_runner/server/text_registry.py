"""In-memory TextSession registry with idle GC.

Single-process, single-user deployment model — no Redis, no DB. If two browser
tabs open simultaneously they get distinct session_ids and don't share state.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from .text_session import TextSession


IDLE_TIMEOUT_SEC = 30 * 60  # drop sessions inactive >30 min
SWEEP_INTERVAL_SEC = 5 * 60  # check every 5 min


class TextSessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, TextSession] = {}
        self._sweep_task: asyncio.Task | None = None

    def register(self, ts: TextSession) -> None:
        self._sessions[ts.session_id] = ts

    def get(self, session_id: str) -> TextSession | None:
        return self._sessions.get(session_id)

    async def drop(self, session_id: str) -> None:
        ts = self._sessions.pop(session_id, None)
        if ts is not None:
            await ts.end()

    async def drop_all(self) -> None:
        for sid in list(self._sessions.keys()):
            await self.drop(sid)

    def start_sweeper(self) -> None:
        if self._sweep_task is None:
            self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def stop_sweeper(self) -> None:
        if self._sweep_task is not None:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except (asyncio.CancelledError, Exception):
                pass
            self._sweep_task = None

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL_SEC)
                now = time.monotonic()
                stale = [
                    sid
                    for sid, ts in self._sessions.items()
                    if now - ts.last_active_at > IDLE_TIMEOUT_SEC
                ]
                for sid in stale:
                    logger.info("dropping idle text session {}", sid)
                    await self.drop(sid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("text session sweeper error: {}", exc)
