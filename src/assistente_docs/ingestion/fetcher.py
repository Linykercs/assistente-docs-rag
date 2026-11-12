"""Polite HTTP fetching: one identified client, a delay between requests and an on-disk cache."""

import json
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import httpx


@dataclass(frozen=True, slots=True)
class RawPage:
    url: str
    content_type: str
    body: str

    @property
    def is_html(self) -> bool:
        return "html" in self.content_type.lower()


class PageFetcher:
    """Fetches pages, caching raw responses so re-runs don't hit the docs site again."""

    def __init__(
        self,
        client: httpx.Client,
        cache_dir: Path,
        *,
        delay_seconds: float,
        refresh: bool = False,
    ) -> None:
        self._client = client
        self._cache_dir = cache_dir
        self._delay_seconds = delay_seconds
        self._refresh = refresh
        self._last_request_at: float | None = None

    def fetch(self, url: str) -> RawPage:
        cache_file = self._cache_dir / f"{sha256(url.encode()).hexdigest()[:16]}.json"
        if cache_file.exists() and not self._refresh:
            return RawPage(**json.loads(cache_file.read_text(encoding="utf-8")))

        self._wait_for_turn()
        response = self._client.get(url)
        response.raise_for_status()
        page = RawPage(
            url=url,
            content_type=response.headers.get("content-type", ""),
            body=response.text,
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(asdict(page), ensure_ascii=False), encoding="utf-8")
        return page

    def _wait_for_turn(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            time.sleep(max(0.0, self._delay_seconds - elapsed))
        self._last_request_at = time.monotonic()
