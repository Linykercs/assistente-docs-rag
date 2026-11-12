"""Turns raw documentation pages into clean Markdown.

The Asaas docs (hosted on ReadMe) serve a Markdown version of every page at `<url>.md`.
That Markdown still carries frontmatter, JSX components and generated sections, which
`clean_markdown` removes. `html_to_markdown` is the fallback for pages only available as HTML.
"""

import re

from bs4 import BeautifulSoup
from markdownify import markdownify

_FRONTMATTER = re.compile(r"\A(?:---[ \t]*\n)+(?:[ \t]*[\w-]+:.*\n|[ \t]*\n)*---[ \t]*\n")
_ANCHOR = re.compile(r'<Anchor\b[^>]*?\bhref="([^"]*)"[^>]*>(.*?)</Anchor>')
_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_COMPONENT_TAG = re.compile(r"</?[A-Z][A-Za-z]*\b[^>]*>")
_H1 = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")

# Generated H1 sections with no value for Q&A: the raw OpenAPI JSON and navigation links.
_DROPPED_SECTIONS = frozenset({"openapi definition", "sibling pages"})
_DIAGRAM_FENCE = "```mermaid"
_HTML_NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form"]


def clean_markdown(text: str) -> str:
    """Strip ReadMe-specific markup, keeping prose, tables and code blocks intact."""
    text = _FRONTMATTER.sub("", text.replace("\r\n", "\n"), count=1)
    lines: list[str] = []
    in_code = in_diagram = in_dropped_section = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("```"):
            opening = not in_code
            if opening:
                in_diagram = stripped.startswith(_DIAGRAM_FENCE)
            if not (in_diagram or in_dropped_section):
                lines.append(line)
            if not opening:
                in_diagram = False
            in_code = opening
            continue

        if in_code:
            if not (in_diagram or in_dropped_section):
                lines.append(line)
            continue

        if heading := _H1.match(stripped):
            in_dropped_section = heading.group(1).lower() in _DROPPED_SECTIONS
        if not in_dropped_section:
            lines.append(_clean_prose_line(line))

    return _normalize_blank_lines("\n".join(lines))


def html_to_markdown(html: str) -> str:
    """Convert the main content of an HTML page to Markdown, dropping navigation chrome."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in root.find_all(_HTML_NOISE_TAGS):
        tag.decompose()
    return _normalize_blank_lines(markdownify(str(root), heading_style="ATX"))


def extract_title(markdown: str, *, fallback: str) -> str:
    """Return the first H1 of the page, or `fallback` when there is none."""
    match = _H1.search(markdown)
    return match.group(1) if match else fallback


def _clean_prose_line(line: str) -> str:
    line = _ANCHOR.sub(r"[\2](\1)", line)
    line = _LINE_BREAK.sub(" ", line)
    line = _COMPONENT_TAG.sub("", line).rstrip()
    # Headings nested in components (e.g. inside a Callout) are indented; restore them.
    return line.lstrip() if line.lstrip().startswith("#") else line


def _normalize_blank_lines(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", text).strip() + "\n"
