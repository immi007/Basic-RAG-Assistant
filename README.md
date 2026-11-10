# Basic RAG Assistant

A beginner-friendly, command-line Retrieval-Augmented Generation (RAG) project built with Python, Sentence Transformers, NumPy, and Ollama.

This project demonstrates the main RAG steps:

1. Load text documents from `data/`.
2. Split documents into overlapping chunks.
3. Convert each chunk into an embedding using `all-MiniLM-L6-v2`.
4. Retrieve the three most similar chunks for a question using cosine similarity (implemented as a dot product on normalized vectors).
5. Optionally send the retrieved context to a local Ollama model (`llama3.2`) to generate an answer.

## Current scope

- The interface is a terminal/command-line prompt. This is not currently a React, Streamlit, or web application.
- The vector index is held in memory with NumPy. It is rebuilt when the program starts; no persistent vector database is used yet.
- The app reads `.txt` files from `data/`.
- If Ollama is unavailable, the app still displays the retrieved text passages.

## Requirements

- Python 3.10 or newer is recommended.
- Internet access the first time the embedding model is downloaded.
- Ollama is optional if you only want to inspect retrieved passages. It is required for generated answers.

## Setup

### 1. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) Set up Ollama for generated answers

Install Ollama from https://ollama.com/, start it, and download the model:

```bash
ollama pull llama3.2
```

The model name in `rag_app.py` is `llama3.2`. Make sure that model is available in Ollama before asking it to generate answers.

### 4. Run the app

From the project root folder:

```bash
python rag_app.py
```

Ask a question at the prompt. Type `quit` or `exit` to stop.

## Add your own documents

Place plain-text `.txt` files inside the `data/` folder. Restart the application so that it loads and indexes the updated files.

## Configuration

The current settings are at the top of `rag_app.py`:

- `CHUNK_WORDS`: number of words in each chunk (currently 120)
- `OVERLAP`: words shared between neighbouring chunks (currently 20)
- `TOP_K`: number of matching chunks returned (currently 3)
- `EMBED_MODEL`: embedding model name
- `LLM_MODEL`: Ollama model name

## Limitations and next improvements

- Add checks for an empty `data/` folder and more robust error messages.
- Add retrieval evaluation questions and metrics such as Recall@k.
- Experiment with chunk size and overlap.
- Add persistent vector storage (for example, Chroma or FAISS) when the project grows.
- Add a web interface only after the command-line version is understood and tested.

## License

No license has been added yet. Ask the author before reusing, modifying, or redistributing this project.
