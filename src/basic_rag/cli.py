"""Command-line interface: ``basic-rag`` or ``python -m basic_rag``."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .generate import GenerationError
from .ingest import chunk_documents, load_documents, summarise_chunks
from .pipeline import Answer, RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="basic-rag",
        description="Ask questions about your documents using a local RAG pipeline.",
    )
    parser.add_argument(
        "question", nargs="?", help="ask one question and exit (omit for interactive chat)"
    )
    parser.add_argument("--data-dir", type=Path, help="folder with .txt/.md documents")
    parser.add_argument("--model", help="Ollama model name (default: llama3.2)")
    parser.add_argument("--top-k", type=int, help="passages to retrieve per question")
    parser.add_argument("--min-score", type=float, help="minimum cosine similarity to keep")
    parser.add_argument("--rebuild", action="store_true", help="ignore the cached index")
    parser.add_argument(
        "--show-chunks", action="store_true", help="print how the documents are chunked and exit"
    )
    parser.add_argument(
        "--retrieval-only", action="store_true", help="show matching passages, skip the LLM"
    )
    parser.add_argument(
        "--show-context", action="store_true", help="print the retrieved passage text"
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {
        "data_dir": args.data_dir,
        "llm_model": args.model,
        "top_k": args.top_k,
        "min_score": args.min_score,
    }
    return dataclasses.replace(
        Settings.from_env(), **{k: v for k, v in overrides.items() if v is not None}
    )


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _print_answer(answer: Answer, show_context: bool) -> None:
    if answer.text is not None:
        print(f"\nAnswer:\n{answer.text}")
    if answer.hits:
        print("\nSources:")
        for i, hit in enumerate(answer.hits, start=1):
            print(f"  [{i}] {hit.chunk.source} (score {hit.score:.2f})")
        if show_context:
            print("\nRetrieved passages:")
            for i, hit in enumerate(answer.hits, start=1):
                print(f"  [{i}] {hit.chunk.text}")


def _chat_loop(pipeline: RAGPipeline, show_context: bool) -> int:
    mode = "generated answers" if pipeline.generator else "retrieval-only"
    print(f"Ready: {len(pipeline.index)} chunks indexed ({mode}).")
    print("Ask a question. Press Enter on an empty line or type 'quit' to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"", "quit", "exit"}:
            return 0
        try:
            _print_answer(pipeline.ask(question), show_context)
        except GenerationError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    try:
        settings = _settings_from_args(args)
        if args.show_chunks:
            documents = load_documents(settings.data_dir)
            chunks = chunk_documents(documents, settings.chunk_words, settings.chunk_overlap)
            print(summarise_chunks(chunks))
            return 0
        pipeline = RAGPipeline.from_settings(
            settings, rebuild=args.rebuild, use_llm=not args.retrieval_only
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.question:
        try:
            _print_answer(pipeline.ask(args.question), args.show_context)
        except GenerationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0
    return _chat_loop(pipeline, args.show_context)
