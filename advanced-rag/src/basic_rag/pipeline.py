"""The RAG pipeline: ingest -> index -> retrieve -> generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .generate import OllamaGenerator
from .index import (
    Embedder,
    Hit,
    SentenceTransformerEmbedder,
    VectorIndex,
    fingerprint,
)
from .ingest import chunk_documents, load_documents

logger = logging.getLogger(__name__)

NOTHING_RELEVANT = "I couldn't find anything relevant in the indexed documents."


class Generator(Protocol):
    def generate(self, question: str, hits: list[Hit]) -> str: ...


@dataclass(frozen=True)
class Answer:
    question: str
    text: str | None  # None in retrieval-only mode
    hits: list[Hit]

    @property
    def sources(self) -> list[str]:
        """Distinct source files, in order of relevance."""
        return list(dict.fromkeys(hit.chunk.source for hit in self.hits))


class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        generator: Generator | None,
        top_k: int = 3,
        min_score: float = 0.15,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        rebuild: bool = False,
        use_llm: bool = True,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ) -> RAGPipeline:
        # Read the documents first so that a bad data directory fails fast,
        # before the (slow) embedding model is loaded.
        documents = load_documents(settings.data_dir)
        chunks = chunk_documents(documents, settings.chunk_words, settings.chunk_overlap)
        if embedder is None:
            embedder = SentenceTransformerEmbedder(settings.embed_model)

        fp = fingerprint(chunks, embedder.name)
        index = None if rebuild else VectorIndex.load(settings.cache_dir, fp)
        if index is None:
            index = VectorIndex.build(chunks, embedder)
            index.save(settings.cache_dir, fp)
            logger.info("Built index: %d chunks from %d documents", len(index), len(documents))
        else:
            logger.info("Loaded cached index: %d chunks", len(index))

        if use_llm and generator is None:
            candidate = OllamaGenerator(settings.llm_model, timeout=settings.llm_timeout)
            if candidate.is_available():
                generator = candidate
            else:
                logger.warning(
                    "Ollama is not reachable; running in retrieval-only mode. "
                    "Start Ollama to get generated answers."
                )
        return cls(index, embedder, generator, settings.top_k, settings.min_score)

    def retrieve(self, question: str) -> list[Hit]:
        question = question.strip()
        if not question:
            raise ValueError("The question must not be empty.")
        query_vector = self.embedder.encode([question])[0]
        return self.index.search(query_vector, self.top_k, self.min_score)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            # Nothing relevant: skip the model entirely instead of letting it guess.
            return Answer(question, NOTHING_RELEVANT, hits)
        if self.generator is None:
            return Answer(question, None, hits)
        return Answer(question, self.generator.generate(question, hits), hits)
