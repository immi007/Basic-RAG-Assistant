"""Basic RAG app: load -> chunk -> embed -> retrieve -> generate."""
import os
import sys
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHUNK_WORDS = 120
OVERLAP = 20
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"  # any model you have pulled with `ollama pull`


def load_docs(folder):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


def build_index(docs, model):
    chunks, sources = [], []
    for name, text in docs:
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(name)
    vectors = model.encode(chunks, normalize_embeddings=True)
    return chunks, sources, np.asarray(vectors)


def retrieve(question, model, chunks, sources, vectors, k=TOP_K):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = vectors @ q  # cosine similarity (vectors are normalised)
    top = np.argsort(scores)[::-1][:k]
    return [(chunks[i], sources[i], float(scores[i])) for i in top]


def generate(question, hits):
    import ollama
    context = "\n\n".join(f"[{src}] {text}" for text, src, _ in hits)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Mention the source file names you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"]


def ollama_ready():
    try:
        import ollama
        ollama.list()  # fails if the Ollama server is not running
        return True
    except Exception:
        return False


def main():
    print("Loading documents and building index...")
    model = SentenceTransformer(EMBED_MODEL)
    docs = load_docs(DATA_DIR)
    chunks, sources, vectors = build_index(docs, model)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} files.\n")

    use_llm = ollama_ready()
    if not use_llm:
        print("(Ollama is not running: showing retrieved chunks only.)\n")

    while True:
        q = input("Ask a question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        hits = retrieve(q, model, chunks, sources, vectors)
        print("\nTop matches:")
        for text, src, score in hits:
            print(f"  - {src} (score {score:.2f}): {text[:90]}...")
        if use_llm:
            print("\nAnswer:\n" + generate(q, hits))
        print()


if __name__ == "__main__":
    sys.exit(main())
