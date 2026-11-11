"""Shared fixtures: a deterministic fake embedder and fake generators."""

import re
import zlib

import numpy as np
import pytest

from basic_rag.ingest import Chunk


class HashingEmbedder:
    """Bag-of-words hashing embedder: deterministic, offline, L2-normalised."""

    name = "hashing-test"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class RecordingGenerator:
    """Fake generator that records its calls."""

    def __init__(self, reply: str = "fake answer [1]") -> None:
        self.reply = reply
        self.calls = []

    def generate(self, question, hits):
        self.calls.append((question, list(hits)))
        return self.reply


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def recording_generator():
    return RecordingGenerator()


@pytest.fixture
def chunks():
    return [
        Chunk("rag.txt", 0, "retrieval augmented generation gives a model external context"),
        Chunk("vectors.txt", 0, "a vector database stores embeddings and finds nearest neighbours"),
        Chunk("vectors.txt", 1, "approximate nearest neighbour search trades accuracy for speed"),
        Chunk("cooking.txt", 0, "simmer the tomato sauce with garlic and fresh basil"),
    ]
