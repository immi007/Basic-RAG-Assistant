# Advanced RAG update

This update packages the newer modular Python RAG implementation supplied in the uploaded `basic-rag.zip`.

## What changed from the beginner version

- Reorganised the single-file program into a `src/basic_rag/` Python package.
- Added configurable settings through environment variables.
- Added TXT and Markdown ingestion, chunk metadata, and improved chunk boundary handling.
- Added a reusable NumPy vector index with a local disk cache.
- Added Ollama answer generation, numbered retrieved passages, source display, and retrieval-only fallback.
- Added a labelled retrieval evaluation dataset (`evals/questions.jsonl`) and a chunk-size comparison command.
- Added automated tests, Ruff configuration, GitHub Actions CI, a README, and an MIT licence.

## Verification performed on the uploaded ZIP

- **54 tests passed** with `PYTHONPATH=src python3 -m pytest -q`.
- The application CLI help and evaluation CLI help both ran successfully.
- Ruff was **not** run in the review environment because Ruff was not installed there.
- A full real-model run with Sentence Transformers and Ollama was **not** performed.

The unit tests use fake embedding and LLM clients; passing them validates software behaviour but does not prove real-model answer quality.

## Run locally

Requirements: Python 3.10–3.12, Ollama, and the embedding dependencies.

```bash
cd basic-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,embeddings]"
ollama pull llama3.2
basic-rag
```

Run tests and lint:

```bash
pytest -q
ruff check .
```

Run retrieval evaluation and compare chunk sizes:

```bash
basic-rag-eval --chunk-words 40,80,120,200 --show-misses
```

## Scope

This is a local command-line RAG application. It is **not** a React frontend, hosted API, or deployed web application yet. Keep the original beginner implementation on `main`; use this branch to review the modular version before merging.
