## Pinecone RAG Demo

This workspace now includes a small Python RAG flow in [rag_pinecone](rag_pinecone):

- [ingest_pdf.py](rag_pinecone/ingest_pdf.py) uploads a PDF into Pinecone.
- [chat.py](rag_pinecone/chat.py) opens an interactive chat that retrieves context from Pinecone before answering.

### Setup

Set these environment variables in your `.env` file:

- `PINECONE_API_KEY`
- `OPENAI_API_KEY` or `GEMINI_API_KEY`
- Optional: `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `EMBEDDING_DIMENSION`
- Optional: `EMBEDDING_BASE_URL`, `CHAT_BASE_URL`

The scripts default to a dense Pinecone index with 768-dimension embeddings.

### Install

The project dependencies are declared in `pyproject.toml`.

If the dependencies are not installed yet, sync the workspace venv with:

```powershell
& "d:/Agentic Ai/rag-chat/.venv/Scripts/python.exe" -m pip install -e .
```

### Run Commands (Windows PowerShell)

Activate the virtual environment:

```powershell
& "d:/Agentic Ai/rag-chat/.venv/Scripts/Activate.ps1"
```

If you do not activate the venv, use the interpreter directly:

```powershell
& "d:/Agentic Ai/rag-chat/.venv/Scripts/python.exe" <command>
```

### Upload a PDF

Run:

```powershell
python rag_pinecone/ingest_pdf.py
```

You can also pass a PDF path directly:

```powershell
python rag_pinecone/ingest_pdf.py "C:\path\to\file.pdf"
```

### Chat with the documents

Run:

```powershell
python rag_pinecone/chat.py
```

Type a question, then the script will embed the question, fetch the nearest chunks from Pinecone, and answer using that retrieved context.

### Web Chat (Uvicorn)

Run:

```powershell
python -m uvicorn rag_pinecone.chat:app --reload
```

Then open `http://127.0.0.1:8000` in your browser to use the chat frontend. The backend logic is the same retrieval + answer pipeline used by the CLI chat mode.
