from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import RLock

from .repository import DatasetContext


@dataclass
class PendingDataset:
    token: str
    context: DatasetContext
    expires_at: float


@dataclass
class SessionState:
    context: DatasetContext
    touched_at: float
    pending: PendingDataset | None = None
    query_times: deque[float] = field(default_factory=deque)


class SessionStore:
    def __init__(self, default_context: DatasetContext, ttl_seconds: int = 1800, max_sessions: int = 64):
        self.default_context = default_context
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._states: dict[str, SessionState] = {}
        self._lock = RLock()

    def _cleanup(self, now: float) -> None:
        expired = [session_id for session_id, state in self._states.items() if now - state.touched_at > self.ttl_seconds]
        for session_id in expired:
            self._states.pop(session_id, None)

    def get(self, session_id: str) -> DatasetContext:
        now = time.monotonic()
        with self._lock:
            self._cleanup(now)
            state = self._states.get(session_id)
            if state is None:
                while len(self._states) >= self.max_sessions:
                    oldest = min(self._states, key=lambda key: self._states[key].touched_at)
                    self._states.pop(oldest, None)
                state = SessionState(context=self.default_context, touched_at=now)
                self._states[session_id] = state
            state.touched_at = now
            if state.pending and state.pending.expires_at <= now:
                state.pending = None
            return state.context

    def stage(self, session_id: str, context: DatasetContext, lifetime_seconds: int = 600) -> str:
        token = uuid.uuid4().hex
        now = time.monotonic()
        with self._lock:
            self.get(session_id)
            self._states[session_id].pending = PendingDataset(token, context, now + lifetime_seconds)
        return token

    def commit(self, session_id: str, token: str) -> DatasetContext:
        now = time.monotonic()
        with self._lock:
            self.get(session_id)
            pending = self._states[session_id].pending
            if pending is None or pending.token != token or pending.expires_at <= now:
                raise KeyError("Preview token is invalid or expired")
            self._states[session_id].context = pending.context
            self._states[session_id].pending = None
            self._states[session_id].touched_at = now
            return self._states[session_id].context

    def reset(self, session_id: str) -> DatasetContext:
        with self._lock:
            self.get(session_id)
            self._states[session_id].context = self.default_context
            self._states[session_id].pending = None
            return self.default_context

    def allow_query(self, session_id: str, limit: int = 20, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        with self._lock:
            self.get(session_id)
            times = self._states[session_id].query_times
            while times and now - times[0] > window_seconds:
                times.popleft()
            if len(times) >= limit:
                return False
            times.append(now)
            return True


def valid_session_id(value: str | None) -> bool:
    if not value or len(value) != 32:
        return False
    try:
        uuid.UUID(hex=value)
        return True
    except ValueError:
        return False
