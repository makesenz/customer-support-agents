"""Split knowledge-base markdown into one chunk per `##` section."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import config


@dataclass
class Chunk:
    id: str
    source: str  # file name, e.g. refund_policy.md
    section: str  # the ## heading (usually the customer question)
    text: str  # text that is indexed and shown to the agents

    def to_dict(self) -> dict:
        return asdict(self)


_FRONT_MATTER = re.compile(r"^---\n.*?\n---\n", re.S)


def _customer_docs() -> list[Path]:
    """Customer-facing docs only. Internal docs guide agents but are never quoted."""
    return sorted(p for p in config.KB_DIR.glob("*.md"))


def chunk_file(path: Path) -> list[Chunk]:
    raw = _FRONT_MATTER.sub("", path.read_text(encoding="utf-8"), count=1)
    title_match = re.search(r"^# (.+)$", raw, re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    chunks: list[Chunk] = []
    # Split before every "## " heading; the part before the first heading is the intro.
    parts = re.split(r"(?m)^## ", raw)
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i == 0:
            section, body = "Overview", re.sub(r"^# .+\n?", "", part).strip()
        else:
            section, _, body = part.partition("\n")
        body = body.strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                id=f"{path.name}#{i}",
                source=path.name,
                section=section.strip(),
                text=f"{title} — {section.strip()}\n{body}",
            )
        )
    return chunks


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in _customer_docs():
        chunks.extend(chunk_file(path))
    return chunks


def kb_fingerprint() -> str:
    """Changes whenever any customer-facing doc changes, so the index is rebuilt."""
    h = hashlib.sha256()
    for path in _customer_docs():
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(config.EMBED_MODEL.encode())
    return h.hexdigest()[:16]
