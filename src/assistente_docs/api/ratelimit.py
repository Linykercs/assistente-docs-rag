"""Request limits for the public demo: per visitor per hour, and a total per day.

Every question spends tokens of the LLM provider, and the free tier has a daily quota, so a
public demo needs a ceiling. The state lives in memory: one process, no extra service, and a
restart forgets it, which is acceptable for a demo.

Behind a Cloudflare tunnel every request reaches the app from 127.0.0.1, so the visitor is
identified by the `CF-Connecting-IP` header the tunnel adds. Without the header (local runs,
tests) the socket address is used.
"""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from fastapi import Request

HOUR_SECONDS = 3600


@dataclass
class RateLimiter:
    per_hour: int
    per_day: int
    clock: Callable[[], float] = time.time
    _by_client: dict[str, deque[float]] = field(default_factory=dict)
    _day: str = ""
    _day_count: int = 0

    def check(self, client: str) -> str | None:
        """Register a request; return why it was refused, or None when it may proceed."""
        now = self.clock()
        today = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
        if today != self._day:
            self._day, self._day_count = today, 0

        if self.per_day and self._day_count >= self.per_day:
            return "A demo atingiu o limite de perguntas de hoje. Volte amanhã."

        if self.per_hour:
            recent = self._by_client.setdefault(client, deque())
            while recent and now - recent[0] >= HOUR_SECONDS:
                recent.popleft()
            if len(recent) >= self.per_hour:
                return (
                    f"Limite de {self.per_hour} perguntas por hora atingido. "
                    "Tente de novo daqui a pouco."
                )
            recent.append(now)

        self._day_count += 1
        return None


def client_key(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip", "").strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "desconhecido"
