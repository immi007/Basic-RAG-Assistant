import pytest

from basic_rag.config import Settings
from basic_rag.index import VectorIndex
from basic_rag.pipeline import NOTHING_RELEVANT, RAGPipeline


def make_pipeline(chunks, embedder, generator, min_score=0.15, top_k=3):
    index = VectorIndex.build(chunks, embedder)
    return RAGPipeline(index, embedder, generator, top_k=top_k, min_score=min_score)


def test_ask_passes_hits_to_the_generator(chunks, embedder, recording_generator):
    pipeline = make_pipeline(chunks, embedder, recording_generator)

    answer = pipeline.ask("how does a vector database find nearest neighbours")

    assert answer.text == "fake answer [1]"
    assert len(recording_generator.calls) == 1
    question, hits = recording_generator.calls[0]
    assert hits == answer.hits
    assert answer.hits[0].chunk.source == "vectors.txt"


def test_sources_are_distinct_and_ordered_by_relevance(chunks, embedder, recording_generator):
    pipeline = make_pipeline(chunks, embedder, recording_generator, min_score=0.0)

    answer = pipeline.ask("nearest neighbour search in a vector database")

    assert answer.sources[0] == "vectors.txt"
    assert len(answer.sources) == len(set(answer.sources))


def test_no_relevant_hits_skips_the_model(chunks, embedder, recording_generator):
    pipeline = make_pipeline(chunks, embedder, recording_generator, min_score=0.3)

    answer = pipeline.ask("who won the world cup")

    assert answer.text == NOTHING_RELEVANT
    assert answer.hits == []
    assert recording_generator.calls == []


def test_retrieval_only_mode_has_no_text(chunks, embedder):
    pipeline = make_pipeline(chunks, embedder, generator=None)

    answer = pipeline.ask("vector database embeddings")

    assert answer.text is None
    assert answer.hits


def test_blank_question_is_rejected(chunks, embedder, recording_generator):
    pipeline = make_pipeline(chunks, embedder, recording_generator)
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_from_settings_builds_then_reuses_the_cache(tmp_path, embedder, recording_generator):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("vector databases store embeddings", encoding="utf-8")
    (data / "b.txt").write_text("tomato sauce needs garlic", encoding="utf-8")
    settings = Settings(data_dir=data, cache_dir=tmp_path / "cache")

    first = RAGPipeline.from_settings(settings, embedder=embedder, generator=recording_generator)
    assert (tmp_path / "cache" / "vectors.npy").is_file()
    assert len(first.index) == 2

    class ExplodingEmbedder:
        name = embedder.name

        def encode(self, texts):
            if len(texts) > 1:
                raise AssertionError("the index should have come from the cache")
            return embedder.encode(texts)

    second = RAGPipeline.from_settings(
        settings, embedder=ExplodingEmbedder(), generator=recording_generator
    )
    assert len(second.index) == 2

    (data / "c.txt").write_text("a brand new document", encoding="utf-8")
    third = RAGPipeline.from_settings(settings, embedder=embedder, generator=recording_generator)
    assert len(third.index) == 3
