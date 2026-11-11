"""Guards the shipped evaluation set against drifting out of sync with the sample corpus."""

from pathlib import Path

from basic_rag.evaluate import _normalise, load_cases
from basic_rag.ingest import load_documents

ROOT = Path(__file__).resolve().parents[1]


def test_every_label_points_at_evidence_that_exists_exactly_once():
    corpus = {d.source: _normalise(d.text) for d in load_documents(ROOT / "data")}
    for case in load_cases(ROOT / "evals" / "questions.jsonl"):
        if not case.answerable:
            continue
        assert case.source in corpus, f"unknown source in: {case.question}"
        occurrences = corpus[case.source].count(_normalise(case.evidence))
        assert occurrences == 1, f"evidence must appear once ({occurrences}): {case.question}"


def test_dataset_is_large_enough_and_has_no_duplicate_questions():
    cases = load_cases(ROOT / "evals" / "questions.jsonl")
    questions = [c.question.lower() for c in cases]
    assert len(questions) == len(set(questions))
    assert sum(c.answerable for c in cases) >= 20
    assert sum(not c.answerable for c in cases) >= 5


def test_every_document_is_covered_by_at_least_one_question():
    sources = {d.source for d in load_documents(ROOT / "data")}
    covered = {c.source for c in load_cases(ROOT / "evals" / "questions.jsonl") if c.answerable}
    assert sources == covered
