"""Loading documents from disk and splitting them into overlapping chunks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".md")


@dataclass(frozen=True)
class Document:
    source: str  # path relative to the data directory
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: int  # position of the chunk within its source document
    text: str


def load_documents(data_dir: Path) -> list[Document]:
    """Read every supported text file under ``data_dir`` (recursively)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            logger.warning("Skipping %s: not valid UTF-8", path)
            continue
        if not text:
            logger.warning("Skipping empty file: %s", path)
            continue
        documents.append(Document(source=path.relative_to(data_dir).as_posix(), text=text))

    if not documents:
        raise ValueError(
            f"No documents found in {data_dir}. Add .txt or .md files "
            f"(supported: {', '.join(SUPPORTED_SUFFIXES)})."
        )
    logger.info("Loaded %d documents from %s", len(documents), data_dir)
    return documents


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of ``size`` words, sharing ``overlap`` words."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):  # this window reached the end of the text
            break
    return chunks


def chunk_documents(documents: Iterable[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc.text, size, overlap)):
            chunks.append(Chunk(source=doc.source, chunk_id=i, text=piece))
    return chunks


def _shared_words(previous: list[str], current: list[str]) -> int:
    """Length of the longest run of words that ends ``previous`` and starts ``current``."""
    for n in range(min(len(previous), len(current)), 0, -1):
        if previous[-n:] == current[:n]:
            return n
    return 0


def summarise_chunks(chunks: list[Chunk], preview_words: int = 4) -> str:
    """Human-readable chunk report: sizes, overlap with the previous chunk, and a preview."""
    if not chunks:
        return "No chunks."
    sizes = [len(c.text.split()) for c in chunks]
    lines = [
        f"{len(chunks)} chunks from {len({c.source for c in chunks})} documents "
        f"(words per chunk: min {min(sizes)}, avg {sum(sizes) / len(sizes):.0f}, max {max(sizes)})",
        "",
    ]
    previous: Chunk | None = None
    for chunk, size in zip(chunks, sizes, strict=True):
        words = chunk.text.split()
        shared = 0
        if previous is not None and previous.source == chunk.source:
            shared = _shared_words(previous.text.split(), words)
        head = " ".join(words[:preview_words])
        tail = " ".join(words[-preview_words:])
        lines.append(
            f"{chunk.source} #{chunk.chunk_id:<2} {size:>3} words, {shared:>2} shared "
            f"with previous | {head} ... {tail}"
        )
        previous = chunk
    return "\n".join(lines)
