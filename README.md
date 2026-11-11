# basic-rag

A small, tested, **fully local** Retrieval-Augmented Generation (RAG) application.
Ask questions about your own text documents. Answers are written by a local
[Ollama](https://ollama.com) model and grounded in the passages retrieved from your files.
No API keys, and no data leaves your machine.

```text
documents ──► chunks ──► embeddings ──► vector index (cached on disk)                  [once]

question ──► embedding ──► top-k chunks ──► prompt ──► Ollama ──► answer + sources    [per question]
```

**Contents:** [Features](#features) · [Quick start](#quick-start) · [Usage](#usage) ·
[How it works](#how-it-works) · [Chunking: how and why](#chunking-how-and-why) ·
[Evaluation](#evaluation) · [Sample corpus](#sample-corpus) · [Configuration](#configuration) ·
[Project layout](#project-layout) · [Development](#development) ·
[Design decisions](#design-decisions) · [Limitations and roadmap](#limitations-and-roadmap)

## Features

- **Grounded answers with sources.** Sources are listed from the retrieval step (not by asking the
  model), and passage numbers like `[1]` in an answer match the source list.
- **Refuses to guess.** If no chunk is similar enough to the question, the model is not called at all.
- **Chunking you can inspect and measure.** `--show-chunks` shows exactly how text is split, and
  `basic-rag-eval` compares chunk sizes on a labelled question set.
- **Fast restarts.** The vector index is cached in `.rag_cache/` and rebuilt automatically when your
  documents or the embedding model change.
- **Graceful degradation.** If Ollama isn't running, the app falls back to retrieval-only mode.
- **Tested.** Unit tests use a fake embedder and fake LLM client, so they run offline in about a second.

## Quick start

Requirements: Python 3.10 to 3.12 (PyTorch, used by `sentence-transformers`, can lag behind the newest
Python releases), [Ollama](https://ollama.com/download), and about 3 GB of free RAM for the default 3B model.

```bash
git clone https://github.com/<your-username>/basic-rag.git
cd basic-rag

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[embeddings]"

ollama pull llama3.2        # once; make sure the Ollama app/server is running
basic-rag                   # interactive chat over the documents in ./data
```

The first run downloads the embedding model (about 90 MB) and builds the index.

## Usage

```bash
basic-rag                                          # interactive chat
basic-rag "Why do neighbouring chunks overlap?"    # one question, then exit
basic-rag --retrieval-only "vector database"       # show matching chunks, skip the LLM
basic-rag --show-context "What is early stopping?" # also print the retrieved text
basic-rag --show-chunks                            # print how documents are chunked, then exit
basic-rag --data-dir ~/notes --model llama3.2:1b   # your own documents / a smaller model
basic-rag --rebuild                                # ignore the cached index
basic-rag-eval --chunk-words 40,80,120,200         # compare chunk sizes (see Evaluation)
python -m basic_rag --help                         # same CLI without the installed script
```

Example session:

```text
> What is early stopping?

Answer:
Early stopping halts training when the validation error starts to rise [1].

Sources:
  [1] overfitting.txt (score 0.xx)
```

(The answer wording and score are illustrative; real values depend on the model and documents.)

**Use your own documents:** put `.txt` or `.md` files in `data/` (subfolders are fine), or point
`--data-dir` at another folder. The index rebuilds automatically on the next run.

## How it works

| Step | Code | What happens |
|---|---|---|
| 1. Load | `ingest.load_documents` | Reads every `.txt` / `.md` file under the data folder (UTF-8, empty files skipped). |
| 2. Chunk | `ingest.chunk_text` | Splits each document into overlapping windows of words. [Details below.](#chunking-how-and-why) |
| 3. Embed | `index.SentenceTransformerEmbedder` | Turns each chunk into a 384-number vector with `all-MiniLM-L6-v2`, normalised to length 1. |
| 4. Index | `index.VectorIndex` | Keeps the vectors in one NumPy matrix and caches it in `.rag_cache/`. |
| 5. Retrieve | `VectorIndex.search` | Embeds the question; one dot product gives the cosine similarity to every chunk; the top `k` above `min_score` are kept. |
| 6. Generate | `generate.OllamaGenerator` | Sends the numbered chunks plus the question to Ollama with temperature 0 and a prompt that allows only context-based answers. |
| 7. Answer | `pipeline.RAGPipeline.ask` | Returns the answer together with the retrieved sources and their scores. |

Steps 1 to 4 run once (and are cached). Steps 5 to 7 run for every question.

## Chunking: how and why

### Why documents must be chunked at all

1. **Embedding models have an input limit.** `all-MiniLM-L6-v2` reads at most 256 word pieces and silently
   ignores the rest. A long document embedded whole would be represented only by its beginning.
2. **One vector can only describe one idea well.** A single vector for a 3,000-word document averages
   many topics into a blur that matches no specific question strongly.
3. **The prompt has a budget.** Only a few passages are sent to the model. Small, focused chunks put the
   relevant sentences in front of it instead of pages of unrelated text.
4. **Sources become precise.** A chunk is a small, citable unit, so an answer can point to the exact passage.

### How this project chunks

It uses a **sliding window over words**. The window is `chunk_words` words wide and moves forward by
`chunk_words - chunk_overlap` words, so neighbouring chunks share `chunk_overlap` words.
With the defaults (120 and 20) the window moves 100 words at a time.

Worked example with `size=4`, `overlap=1` (step = 3), on a 10-word text:

```text
words:    w0 w1 w2 w3 w4 w5 w6 w7 w8 w9
chunk 0:  [w0 w1 w2 w3]
chunk 1:           [w3 w4 w5 w6]          <- w3 is shared with chunk 0
chunk 2:                    [w6 w7 w8 w9]  <- window reaches the end of the text: stop
```

Rules the code follows (all covered by tests):

- **Chunks never cross documents.** Every chunk keeps its `source` file and a `chunk_id`.
- **Stop as soon as a window reaches the end.** An earlier version kept stepping and produced a final
  fragment made only of words that were already in the previous chunk. That redundant chunk is gone.
- **Every word lands in at least one chunk**, and no chunk is longer than `chunk_words`.
- **Invalid settings fail early:** `chunk_words > 0` and `0 <= chunk_overlap < chunk_words`.

See it on your own data (chunking needs no model, so this is instant):

```bash
basic-rag --show-chunks
```

```text
32 chunks from 14 documents (words per chunk: min 30, avg 98, max 120)

chunking.txt #0  120 words,  0 shared with previous | Chunking is the step ... when a section is
chunking.txt #1  120 words, 20 shared with previous | natural boundaries such as ... several topics into one
chunking.txt #2  113 words, 20 shared with previous | the language model needs. ... rather than by guessing.
classification_metrics.txt #0  120 words,  0 shared with previous | Accuracy is the fraction ... the sum of true
```

### Why these defaults (120 words, 20 overlap)

| Choice | Reason |
|---|---|
| **Unit: words** | Simple, readable, and independent of the embedding model. Tokens would be more exact, but they tie the chunker to one tokenizer. |
| **120 words** | English averages roughly 1.3 tokens per word, so 120 words is about 160 tokens: comfortably under the 256-token limit of MiniLM, with headroom for numbers and code. It is also small enough that a chunk is usually about one idea. |
| **20 words of overlap (about 17%)** | Roughly one to two sentences. A sentence cut at a boundary then appears whole in the neighbouring chunk. More overlap means more duplicated text and a larger index. |
| **`top_k = 3`** | Three chunks is about 360 words of context, which fits easily in a small local model's prompt and keeps the answer focused. |

Note that general advice (including the sample document `vector_search.txt`) quotes 150 to 300 words per
chunk. This project uses a smaller size because MiniLM cannot read beyond roughly 190 words. If you
switch to an embedding model with a longer input limit, larger chunks may work better.

**These defaults are reasoned starting points, not tuned optima.** The right size depends on your documents
and questions, so measure it (next section).

### Trade-offs and known weaknesses

| | Small chunks (about 40 to 80 words) | Large chunks (about 200+ words) |
|---|---|---|
| Retrieval precision | High: the match is the relevant sentence | Lower: several topics are blurred into one vector |
| Context for the LLM | Can lack the surrounding explanation | Rich context |
| Index size | More chunks, more vectors | Fewer chunks |
| Risk | The answer is split across two chunks | Hits the model's input limit; wastes prompt space |

Weaknesses of the current approach:

- Cuts can fall **mid-sentence**. Overlap softens this but doesn't remove it.
- It ignores headings, lists and tables. Structure-aware or semantic chunking would respect them.
- The last chunk of a document can be short and mostly overlap (for example 30 words, of which 20 are shared).
- Documents shorter than `chunk_words` become a single chunk, so chunking does nothing for them.

## Evaluation

Claims about retrieval quality should be measured. `basic-rag-eval` scores retrieval against a labelled
question set in `evals/questions.jsonl` and can compare several chunk sizes in one run.

### The dataset

One JSON object per line. There are 35 questions over the 14 sample documents:

```json
{"question": "What happens if a chunk is longer than the embedding model's input limit?", "source": "chunking.txt", "evidence": "silently truncated"}
{"question": "Who won the FIFA World Cup in 2018?", "source": null, "evidence": null}
```

- **29 answerable questions** name the expected `source` file and an `evidence` phrase. A retrieved chunk
  counts as relevant only if it comes from that file **and** contains the phrase (case and whitespace
  ignored). Labelling by phrase instead of chunk id keeps the labels valid when you change the chunk size.
- **6 unanswerable questions** (`"source": null`) have no answer in the corpus. They test whether the
  similarity threshold correctly refuses.
- A test checks that every evidence phrase occurs exactly once in its file and that every document is
  covered, so the dataset cannot silently drift from the corpus.

### Metrics

| Metric | Meaning |
|---|---|
| **Hit@1** | Share of answerable questions where the first retrieved chunk is relevant. |
| **Hit@k** | Share where a relevant chunk appears anywhere in the top `k` (default 3). This is recall@k with a single gold chunk. |
| **MRR** | Mean reciprocal rank: 1 if the first chunk is relevant, 1/2 if the second, 1/3 if the third, 0 if none. Rewards ranking the right chunk higher. |
| **answerable kept** | Share of answerable questions whose best score is at or above `min_score`. Low means the threshold rejects good questions. |
| **unanswerable rejected** | Share of unanswerable questions whose best score is below `min_score`. Low means the threshold lets off-topic questions through. |

Hit@1, Hit@k and MRR are computed **without** the threshold, so they judge the ranking itself.
The last two columns judge the threshold separately.

### Running it

```bash
basic-rag-eval                                   # the configured chunk size (120 words)
basic-rag-eval --chunk-words 40,80,120,200       # compare chunk sizes
basic-rag-eval --chunk-words 120 --overlap-ratio 0   # what does overlap buy you?
basic-rag-eval --min-score 0.25                  # try another refusal threshold
basic-rag-eval --show-misses                     # list the failed questions and what was retrieved instead
```

The overlap is scaled with the chunk size (default ratio: configured overlap / chunk size, about 0.17).
The output is a Markdown table you can paste straight into this README. It has this shape
(`x.xx` marks values you get from your own run; none are claimed here):

```text
| chunk words | overlap | chunks | avg words | Hit@1 | Hit@3 | MRR  | answerable kept (>= 0.15) | unanswerable rejected |
|---|---|---|---|---|---|---|---|---|
| 40  | 7  | ... | ... | x.xx | x.xx | x.xx | x.xx | x.xx |
| 120 | 20 | ... | ... | x.xx | x.xx | x.xx | x.xx | x.xx |
```

### Reading the results

- **Choosing the chunk size:** prefer the size with the best Hit@k and MRR, and break ties toward fewer,
  larger chunks (cheaper and more context for the LLM).
- **Choosing `min_score`:** run `--min-score` at a few values. Pick the highest value that still keeps almost
  all answerable questions, while rejecting as many unanswerable ones as possible.
- **Use `--show-misses`** to see whether failures come from chunking (evidence split across chunks), from
  the embedding model, or from a label that is too strict.

### Caveats

- With 29 answerable questions, **one question is worth about 3.4 percentage points**. Treat differences of
  one or two questions as noise.
- The corpus and questions were written for this repository. Results on your own documents will differ.
- This measures **retrieval only**. Answer quality (faithfulness to the sources) is not yet scored automatically.
- A relevant phrase that straddles a chunk boundary counts as a miss for that size. That is deliberate:
  it is exactly the failure that overlap is meant to prevent.

## Sample corpus

`data/` contains 14 short plain-text documents (about 2,800 words) written for this project, covering
RAG and LLM basics (`rag`, `embeddings`, `vector_search`, `transformers`, `evaluation`, `chunking`,
`tokenization`, `fine_tuning_vs_rag`, `prompt_engineering`, `hallucination`) and general machine learning
(`overfitting`, `classification_metrics`, `gradient_descent`, `data_cleaning`). Most documents contain
several distinct topics so that chunk size has a measurable effect. Replace them with your own files
whenever you like. If you do, also replace `evals/questions.jsonl`, otherwise the eval will fail.

## Configuration

Every option can be set by environment variable. CLI flags take precedence.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_DATA_DIR` | `data` | Folder with `.txt` / `.md` documents |
| `RAG_CACHE_DIR` | `.rag_cache` | Where the vector index is cached |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_LLM_MODEL` | `llama3.2` | Ollama model used for answers |
| `RAG_CHUNK_WORDS` | `120` | Words per chunk |
| `RAG_CHUNK_OVERLAP` | `20` | Words shared by neighbouring chunks |
| `RAG_TOP_K` | `3` | Chunks retrieved per question |
| `RAG_MIN_SCORE` | `0.15` | Minimum cosine similarity for a chunk to be used. A starting guess, so tune it with the eval |
| `RAG_LLM_TIMEOUT` | `120` | Seconds to wait for Ollama |

The Ollama server address is read from the standard `OLLAMA_HOST` variable.

## Project layout

```text
src/basic_rag/
  config.py     Settings dataclass + RAG_* environment overrides
  ingest.py     load documents, chunk them, describe chunks (--show-chunks)
  index.py      embedder, in-memory cosine-similarity index, on-disk cache
  generate.py   prompt construction and the Ollama client wrapper
  pipeline.py   RAGPipeline: ingest -> index -> retrieve -> generate
  evaluate.py   retrieval metrics and the chunk-size sweep (basic-rag-eval)
  cli.py        command-line interface (basic-rag)
data/           sample corpus
evals/          labelled evaluation questions (questions.jsonl)
tests/          offline unit tests (fake embedder, fake LLM client)
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

CI (GitHub Actions) runs both on Python 3.10, 3.11 and 3.12. The tests need neither Ollama nor the
embedding model.

## Design decisions

- **Exact search with NumPy.** For up to tens of thousands of chunks, a normalised matrix and one dot
  product is fast, dependency-free and easy to reason about. A vector database would be overkill here.
- **Dependency injection.** `RAGPipeline` takes an embedder and a generator object, so tests (and future
  backends) can swap them without patching.
- **Deterministic generation.** Temperature is 0 and the prompt requires answers to come only from the
  numbered passages.
- **Retrieved text is treated as untrusted.** The prompt tells the model to ignore instructions found
  inside passages. This reduces, but does not eliminate, prompt-injection risk.
- **Refusal without the model.** When no chunk clears `min_score`, the app answers "nothing relevant"
  itself. This is cheaper and safer than asking a small model not to hallucinate.
- **Cache keyed by content.** The cache fingerprint covers the embedding model and every chunk's text, so
  edits to documents or chunk settings never serve stale vectors.

## Limitations and roadmap

- Only `.txt` and `.md` files are supported; PDF and HTML loaders are not included yet.
- Chunking is word-based and structure-blind. Sentence-aware, recursive or semantic chunking is the next
  step, and the eval is how you would prove it helps.
- The whole index lives in memory and search is brute-force.
- No re-ranking, hybrid (keyword + vector) search or conversation memory yet.
- The eval covers retrieval only. A generation eval (faithfulness of answers to their sources) is not built.
- Small local models (1B to 3B) follow instructions less reliably than large ones. If answers drift from
  the sources, try a larger Ollama model.

## License

MIT. See [LICENSE](LICENSE).
