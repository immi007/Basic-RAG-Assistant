"""Retrieval evaluation on a labelled question set, with a chunk-size sweep.

Each line of the dataset (JSON Lines) is one question::

    {"question": "...", "source": "chunking.txt", "evidence": "silently truncated"}
    {"question": "Who won the 2018 World Cup?", "source": null, "evidence": null}

A retrieved chunk counts as relevant when it comes from ``source`` AND contains the
``evidence`` phrase (compared case- and whitespace-insensitively). Using a phrase instead of
a chunk id keeps the labels valid when the chunk size changes. Questions with
``"source": null`` must NOT be answerable from the corpus; they test the refusal threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .index import Embedder, SentenceTransformerEmbedder, VectorIndex
from .ingest import Chunk, Document, chunk_documents, load_documents

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class EvalCase:
    question: str
    source: str | None  # None => the corpus cannot answer this question
    evidence: str | None

    @property
    def answerable(self) -> bool:
        return self.source is not None


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            question = item["question"]
            source = item.get("source")
            evidence = item.get("evidence")
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}:{number}: invalid evaluation case ({exc})") from exc
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{number}: 'question' must be a non-empty string")
        if (source is None) != (evidence is None):
            raise ValueError(
                f"{path}:{number}: 'source' and 'evidence' must both be set, or both null"
            )
        cases.append(EvalCase(question.strip(), source, evidence))
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def is_relevant(chunk: Chunk, case: EvalCase) -> bool:
    return (
        case.answerable
        and chunk.source == case.source
        and _normalise(case.evidence or "") in _normalise(chunk.text)
    )


@dataclass(frozen=True)
class Report:
    chunk_words: int
    chunk_overlap: int
    n_chunks: int
    avg_chunk_words: float
    k: int
    min_score: float
    n_answerable: int
    n_unanswerable: int
    hit_at_1: float  # share of answerable questions with a relevant chunk ranked first
    hit_at_k: float  # ... with a relevant chunk anywhere in the top k
    mrr: float  # mean reciprocal rank of the first relevant chunk (0 if not in top k)
    answerable_kept: float  # share of answerable questions whose best score >= min_score
    unanswerable_rejected: float  # share of unanswerable questions whose best score < min_score
    misses: tuple[tuple[str, str, str], ...]  # (question, expected source, top retrieved source)


def _ratio(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def run_eval(
    documents: list[Document],
    embedder: Embedder,
    cases: list[EvalCase],
    *,
    chunk_words: int,
    chunk_overlap: int,
    k: int,
    min_score: float,
) -> Report:
    chunks = chunk_documents(documents, chunk_words, chunk_overlap)
    index = VectorIndex.build(chunks, embedder)
    query_vectors = embedder.encode([case.question for case in cases])

    hit1 = hitk = kept = rejected = 0
    reciprocal_ranks: list[float] = []
    misses: list[tuple[str, str, str]] = []
    n_answerable = n_unanswerable = 0

    for case, query_vector in zip(cases, query_vectors, strict=True):
        hits = index.search(query_vector, top_k=k)  # no threshold: judge the ranking itself
        best_score = hits[0].score if hits else -math.inf
        if case.answerable:
            n_answerable += 1
            rank = next((i for i, h in enumerate(hits, 1) if is_relevant(h.chunk, case)), None)
            hit1 += rank == 1
            hitk += rank is not None
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            kept += best_score >= min_score
            if rank is None:
                top_source = hits[0].chunk.source if hits else "-"
                misses.append((case.question, case.source or "", top_source))
        else:
            n_unanswerable += 1
            rejected += best_score < min_score

    word_counts = [len(c.text.split()) for c in chunks]
    return Report(
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
        n_chunks=len(chunks),
        avg_chunk_words=sum(word_counts) / len(word_counts),
        k=k,
        min_score=min_score,
        n_answerable=n_answerable,
        n_unanswerable=n_unanswerable,
        hit_at_1=_ratio(hit1, n_answerable),
        hit_at_k=_ratio(hitk, n_answerable),
        mrr=_ratio(sum(reciprocal_ranks), n_answerable),
        answerable_kept=_ratio(kept, n_answerable),
        unanswerable_rejected=_ratio(rejected, n_unanswerable),
        misses=tuple(misses),
    )


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def format_table(reports: list[Report]) -> str:
    """Render reports as a Markdown table (paste it straight into a README)."""
    first = reports[0]
    header = (
        f"| chunk words | overlap | chunks | avg words | Hit@1 | Hit@{first.k} | MRR "
        f"| answerable kept (>= {first.min_score}) | unanswerable rejected |"
    )
    lines = [header, "|" + "---|" * 9]
    for r in reports:
        lines.append(
            f"| {r.chunk_words} | {r.chunk_overlap} | {r.n_chunks} | {r.avg_chunk_words:.0f} "
            f"| {_pct(r.hit_at_1)} | {_pct(r.hit_at_k)} | {_pct(r.mrr)} "
            f"| {_pct(r.answerable_kept)} | {_pct(r.unanswerable_rejected)} |"
        )
    return "\n".join(lines)


def _parse_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--chunk-words must be comma-separated integers, got {raw!r}") from exc
    if not sizes:
        raise ValueError("--chunk-words needs at least one size")
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basic-rag-eval",
        description="Evaluate retrieval quality; optionally sweep several chunk sizes.",
    )
    parser.add_argument("--data-dir", type=Path, help="documents folder (default: ./data)")
    parser.add_argument("--cases", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("-k", type=int, help="ranks to inspect (default: RAG_TOP_K, 3)")
    parser.add_argument("--min-score", type=float, help="refusal threshold to evaluate")
    parser.add_argument(
        "--chunk-words",
        help="comma-separated chunk sizes to compare, e.g. 40,80,120,200 "
        "(default: the configured size)",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        help="overlap as a fraction of chunk size (default: configured overlap / chunk size)",
    )
    parser.add_argument("--show-misses", action="store_true", help="list failed questions")
    return parser


def main(argv: list[str] | None = None, embedder: Embedder | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        data_dir = args.data_dir or settings.data_dir
        k = args.k or settings.top_k
        min_score = settings.min_score if args.min_score is None else args.min_score
        ratio = (
            settings.chunk_overlap / settings.chunk_words
            if args.overlap_ratio is None
            else args.overlap_ratio
        )
        if k <= 0 or not 0 <= ratio < 1:
            raise ValueError("-k must be positive and --overlap-ratio must be in [0, 1)")
        sizes = _parse_sizes(args.chunk_words) if args.chunk_words else [settings.chunk_words]

        documents = load_documents(data_dir)
        cases = load_cases(args.cases)
        embedder = embedder or SentenceTransformerEmbedder(settings.embed_model)

        reports = []
        for size in sizes:
            overlap = min(size - 1, round(size * ratio))
            reports.append(
                run_eval(
                    documents, embedder, cases,
                    chunk_words=size, chunk_overlap=overlap, k=k, min_score=min_score,
                )
            )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    first = reports[0]
    print(
        f"Evaluated {first.n_answerable} answerable and {first.n_unanswerable} unanswerable "
        f"questions over {len(documents)} documents (embedding model: {embedder.name}).\n"
    )
    print(format_table(reports))
    if args.show_misses:
        for r in reports:
            print(f"\nMisses at chunk_words={r.chunk_words} (relevant chunk not in top {r.k}):")
            for question, expected, got in r.misses or [("(none)", "", "")]:
                suffix = f"  expected {expected}, top hit was {got}" if expected else ""
                print(f"  - {question}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
