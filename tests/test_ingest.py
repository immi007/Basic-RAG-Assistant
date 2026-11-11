import pytest

from basic_rag.ingest import Document, chunk_documents, chunk_text, load_documents


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n ", size=5, overlap=1) == []


def test_short_text_is_a_single_chunk():
    assert chunk_text("just three words", size=10, overlap=2) == ["just three words"]


def test_neighbouring_chunks_share_the_overlap():
    chunks = chunk_text(words(10), size=4, overlap=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_no_redundant_trailing_chunk():
    # 7 words, size 4, overlap 1: the second window already reaches the end.
    assert len(chunk_text(words(7), size=4, overlap=1)) == 2


def test_every_word_is_covered_and_chunks_respect_size():
    for total in range(1, 60):
        for size, overlap in [(5, 0), (5, 2), (8, 7), (1, 0)]:
            chunks = chunk_text(words(total), size, overlap)
            covered = {w for c in chunks for w in c.split()}
            assert covered == set(words(total).split())
            assert all(len(c.split()) <= size for c in chunks)


def test_invalid_parameters_raise():
    for size, overlap in [(0, 0), (-1, 0), (5, 5), (5, -1)]:
        with pytest.raises(ValueError):
            chunk_text("a b c", size, overlap)


def test_chunk_documents_numbers_chunks_per_source():
    docs = [Document("a.txt", words(10)), Document("b.txt", "short")]
    chunks = chunk_documents(docs, size=4, overlap=1)
    assert [(c.source, c.chunk_id) for c in chunks] == [
        ("a.txt", 0), ("a.txt", 1), ("a.txt", 2), ("b.txt", 0),
    ]


def test_load_documents_filters_and_sorts(tmp_path):
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  \n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.txt").write_text("gamma", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.source for d in docs] == ["a.txt", "b.md", "sub/c.txt"]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")


def test_directory_without_documents_raises(tmp_path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError):
        load_documents(tmp_path)


def test_summarise_chunks_shows_overlap_and_sizes():
    from basic_rag.ingest import summarise_chunks

    chunks = chunk_documents([Document("a.txt", words(10))], size=4, overlap=1)
    report = summarise_chunks(chunks)

    lines = report.splitlines()
    assert lines[0].startswith("3 chunks from 1 documents")
    assert " 0 shared" in lines[2] and " 1 shared" in lines[3] and " 1 shared" in lines[4]
    assert summarise_chunks([]) == "No chunks."
