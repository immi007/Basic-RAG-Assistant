import json

import pytest

from basic_rag.index import VectorIndex, fingerprint


def test_most_relevant_chunk_ranks_first(chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    query = embedder.encode(["where are embeddings stored in a vector database"])[0]

    hits = index.search(query, top_k=2)

    assert hits[0].chunk.source == "vectors.txt"
    assert hits[0].chunk.chunk_id == 0
    assert hits[0].score >= hits[1].score


def test_top_k_limits_results(chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    query = embedder.encode(["nearest neighbours"])[0]
    assert len(index.search(query, top_k=1)) == 1


def test_min_score_filters_irrelevant_chunks(chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    query = embedder.encode(["quantum chromodynamics lecture"])[0]
    assert index.search(query, top_k=4, min_score=0.2) == []


def test_build_rejects_empty_input(embedder):
    with pytest.raises(ValueError):
        VectorIndex.build([], embedder)


def test_length_mismatch_is_rejected(chunks, embedder):
    with pytest.raises(ValueError):
        VectorIndex(chunks, embedder.encode(["only one"]))


def test_cache_round_trip(tmp_path, chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    fp = fingerprint(chunks, embedder.name)
    index.save(tmp_path, fp)

    loaded = VectorIndex.load(tmp_path, fp)

    assert loaded is not None
    assert loaded.chunks == index.chunks
    assert (loaded.vectors == index.vectors).all()


def test_stale_cache_is_ignored(tmp_path, chunks, embedder):
    index = VectorIndex.build(chunks, embedder)
    index.save(tmp_path, fingerprint(chunks, embedder.name))

    changed = fingerprint(chunks[:-1], embedder.name)
    assert VectorIndex.load(tmp_path, changed) is None


def test_fingerprint_depends_on_model_and_text(chunks):
    base = fingerprint(chunks, "model-a")
    assert fingerprint(chunks, "model-b") != base
    assert fingerprint(chunks[:-1], "model-a") != base
    assert fingerprint(chunks, "model-a") == base


def test_missing_or_corrupt_cache_returns_none(tmp_path, chunks, embedder):
    fp = fingerprint(chunks, embedder.name)
    assert VectorIndex.load(tmp_path, fp) is None

    VectorIndex.build(chunks, embedder).save(tmp_path, fp)
    (tmp_path / "chunks.json").write_text("{not json", encoding="utf-8")
    assert VectorIndex.load(tmp_path, fp) is None

    (tmp_path / "chunks.json").write_text(json.dumps({"fingerprint": fp}), encoding="utf-8")
    assert VectorIndex.load(tmp_path, fp) is None
