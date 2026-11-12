"""Loads the list of documentation pages to index."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class _SourcesFile(BaseModel):
    urls: list[str]


def load_source_urls(path: Path) -> list[str]:
    """Return the URLs listed in the sources file, de-duplicated and in file order."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    urls = (url.strip().rstrip("/") for url in _SourcesFile.model_validate(data).urls)
    return list(dict.fromkeys(urls))
