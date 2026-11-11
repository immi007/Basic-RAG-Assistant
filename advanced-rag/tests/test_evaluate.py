import json
import math

import pytest

from basic_rag.evaluate import (
    EvalCase,
    format_table,
    is_relevant,
    load_cases,
    main,
    run_eval,
)
from basic_rag.ingest import Chunk, Document


def write_cases(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_cases_reads_answerable_and_unanswerable(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"question": "a?", "source": "x.txt", "evidence": "foo"}\n\n'
        '{"question": "b?", "source": null, "evidence": null}\n',
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [c.answerable for c in cases] == [True, False]
    assert cases[0].evidence == "foo"


def test_load_cases_reports_the_bad_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "ok?", "source": null, "evidence": null}\n{oops\n')
    with pytest.raises(ValueError) as info:
        load_cases(path)
    assert ":2:" in str(info.value)


def test_load_cases_rejects_half_labelled_and_missing_files(tmp_path):
    path = tmp_path / "q.jsonl"
    write_cases(path, [{"question": "a?", "source": "x.txt", "evidence": None}])
    with pytest.raises(ValueError):
        load_cases(path)
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "missing.jsonl")
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_cases(path)


def test_is_relevant_needs_matching_source_and_evidence():
    chunk = Chunk("a.txt", 0, "Text is  SILENTLY\ntruncated beyond the limit")
    case = EvalCase("q", "a.txt", "silently truncated")
    assert is_relevant(chunk, case)  # case and whitespace are ignored
    assert not is_relevant(Chunk("b.txt", 0, chunk.text), case)
    assert not is_relevant(Chunk("a.txt", 0, "nothing useful here"), case)
    assert not is_relevant(chunk, EvalCase("q", None, None))


def make_docs():
    return [
        Document("vectors.txt", "a vector database stores embeddings and finds nearest neighbours"),
        Document("cooking.txt", "simmer the tomato sauce with garlic and fresh basil"),
    ]


def test_run_eval_scores_hits_and_refusals(embedder):
    cases = [
        EvalCase("where are embeddings stored in a vector database", "vectors.txt",
                 "vector database stores embeddings"),
        EvalCase("quantum chromodynamics lecture", None, None),
    ]
    report = run_eval(make_docs(), embedder, cases,
                      chunk_words=50, chunk_overlap=5, k=2, min_score=0.15)

    assert report.n_answerable == 1 and report.n_unanswerable == 1
    assert report.hit_at_1 == 1.0 and report.hit_at_k == 1.0 and report.mrr == 1.0
    assert report.answerable_kept == 1.0
    assert report.unanswerable_rejected == 1.0
    assert report.misses == ()


def test_run_eval_records_misses(embedder):
    cases = [EvalCase("vector database embeddings", "vectors.txt", "text that is not there")]
    report = run_eval(make_docs(), embedder, cases,
                      chunk_words=50, chunk_overlap=5, k=2, min_score=0.15)

    assert report.hit_at_k == 0.0 and report.mrr == 0.0
    assert report.misses == (("vector database embeddings", "vectors.txt", "vectors.txt"),)


def test_overlap_rescues_evidence_that_straddles_a_boundary(embedder):
    docs = [Document("d.txt", "alpha beta gamma delta epsilon zeta eta theta")]
    cases = [EvalCase("delta epsilon", "d.txt", "delta epsilon")]

    def evaluate(overlap):
        return run_eval(
            docs, embedder, cases, chunk_words=4, chunk_overlap=overlap, k=2, min_score=0.0
        )

    no_overlap, with_overlap = evaluate(0), evaluate(1)

    assert no_overlap.hit_at_k == 0.0  # "delta" and "epsilon" land in different chunks
    assert with_overlap.hit_at_1 == 1.0  # the shared word keeps the phrase in one chunk


def test_ratios_are_nan_when_a_group_is_empty(embedder):
    cases = [EvalCase("quantum chromodynamics", None, None)]
    report = run_eval(make_docs(), embedder, cases,
                      chunk_words=50, chunk_overlap=5, k=2, min_score=0.15)
    assert math.isnan(report.hit_at_k)
    assert "n/a" in format_table([report])


def test_format_table_has_one_row_per_report(embedder):
    cases = [EvalCase("vector database embeddings", "vectors.txt", "vector database")]
    reports = [
        run_eval(make_docs(), embedder, cases,
                 chunk_words=size, chunk_overlap=1, k=2, min_score=0.1)
        for size in (4, 8)
    ]
    lines = format_table(reports).splitlines()
    assert len(lines) == 2 + 2  # header, separator, two rows
    assert lines[2].startswith("| 4 |") and lines[3].startswith("| 8 |")


def test_main_runs_a_sweep(tmp_path, embedder, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "vectors.txt").write_text(
        "a vector database stores embeddings and finds nearest neighbours", encoding="utf-8"
    )
    cases = tmp_path / "q.jsonl"
    write_cases(cases, [
        {"question": "where are embeddings stored", "source": "vectors.txt",
         "evidence": "vector database stores embeddings"},
        {"question": "quantum chromodynamics", "source": None, "evidence": None},
    ])

    code = main(["--data-dir", str(data), "--cases", str(cases), "--chunk-words", "4,8",
                 "--show-misses"], embedder=embedder)

    out = capsys.readouterr().out
    assert code == 0
    assert "| chunk words |" in out
    assert "| 4 |" in out and "| 8 |" in out


def test_main_reports_errors_with_exit_code_2(tmp_path, embedder, capsys):
    assert main(["--cases", str(tmp_path / "none.jsonl"), "--data-dir", str(tmp_path)],
                embedder=embedder) == 2
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    q = tmp_path / "q.jsonl"
    write_cases(q, [{"question": "x?", "source": None, "evidence": None}])
    assert main(["--data-dir", str(tmp_path), "--cases", str(q), "--chunk-words", "a,b"],
                embedder=embedder) == 2
    assert capsys.readouterr().err.count("Error:") == 2
