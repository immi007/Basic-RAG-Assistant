"""Embeddings and a small in-memory vector index with an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .ingest import Chunk

logger = logging.getLogger(__name__)

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.json"


class Embedder(Protocol):
    """Anything that maps texts to L2-normalised vectors."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model (runs locally)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install -e '.[embeddings]'"
            ) from exc
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is more similar


def fingerprint(chunks: Sequence[Chunk], embed_model: str) -> str:
    """Hash of everything the cached vectors depend on."""
    digest = hashlib.sha256()
    digest.update(embed_model.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x00")
        digest.update(chunk.source.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


class VectorIndex:
    """Exact (brute-force) cosine-similarity search over normalised vectors."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Embedder) -> VectorIndex:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        vectors = embedder.encode([c.text for c in chunks])
        return cls(list(chunks), vectors)

    def search(self, query_vector: np.ndarray, top_k: int, min_score: float = -1.0) -> list[Hit]:
        scores = self.vectors @ query_vector  # dot product == cosine for unit vectors
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score
        ]

    def save(self, directory: Path, fp: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        vectors_tmp = directory / "vectors.tmp.npy"
        chunks_tmp = directory / "chunks.tmp.json"
        np.save(vectors_tmp, self.vectors)
        payload = {
            "fingerprint": fp,
            "chunks": [
                {"source": c.source, "chunk_id": c.chunk_id, "text": c.text} for c in self.chunks
            ],
        }
        chunks_tmp.write_text(json.dumps(payload), encoding="utf-8")
        # Replace the vectors first; the fingerprint lives in the chunks file, so a
        # crash between the two replaces can only cause a rebuild, never wrong results.
        os.replace(vectors_tmp, directory / _VECTORS_FILE)
        os.replace(chunks_tmp, directory / _CHUNKS_FILE)

    @classmethod
    def load(cls, directory: Path, fp: str) -> VectorIndex | None:
        """Return the cached index if it exists and matches ``fp``, else ``None``."""
        vectors_path = directory / _VECTORS_FILE
        chunks_path = directory / _CHUNKS_FILE
        if not (vectors_path.is_file() and chunks_path.is_file()):
            return None
        try:
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            if payload["fingerprint"] != fp:
                logger.info("Cache is stale; rebuilding the index.")
                return None
            chunks = [Chunk(**item) for item in payload["chunks"]]
            vectors = np.load(vectors_path, allow_pickle=False)
            return cls(chunks, vectors)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring unreadable cache in %s (%s)", directory, exc)
            return None
