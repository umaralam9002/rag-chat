"""Upload a PDF into Pinecone for later RAG queries.

This script opens a file chooser, extracts text from the selected PDF, chunks
the text, creates embeddings, and stores the chunks in Pinecone.

Environment variables:
    GEMINI_API_KEY
    EMBEDDING_MODEL (default: gemini-embedding-001)
    PINECONE_API_KEY
    PINECONE_INDEX_NAME (default: rag-chat)
    PINECONE_NAMESPACE (default: default)
    PINECONE_CLOUD (default: aws)
    PINECONE_REGION (default: us-east-1)
    EMBEDDING_DIMENSION (default: 3072)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from google import genai
from dotenv import find_dotenv, load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader


DEFAULT_INDEX_NAME = "rag-chat"
DEFAULT_NAMESPACE = "default"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 3072
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def build_gemini_client() -> genai.Client:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    return genai.Client(api_key=gemini_key)


def choose_pdf_file() -> Path:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = askopenfilename(
        title="Select a PDF to upload",
        filetypes=[("PDF files", "*.pdf")],
    )
    root.destroy()

    if not file_path:
        raise SystemExit("No PDF selected.")

    return Path(file_path)


def extract_pdf_pages(pdf_path: Path) -> list[dict[str, str | int]]:
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, str | int]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})

    if not pages:
        raise RuntimeError(f"No extractable text found in {pdf_path.name}.")

    return pages


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    cleaned = " ".join(text.split())
    chunks: list[str] = []
    start = 0

    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - chunk_overlap

    return chunks


def embed_texts(client: genai.Client, model: str, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(model=model, contents=texts)
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None:
        raise RuntimeError("Unexpected Gemini embed response shape")

    vectors: list[list[float]] = []
    for item in embeddings:
        values = getattr(item, "values", None)
        if not isinstance(values, list):
            raise RuntimeError("Could not parse embedding vector from Gemini response")
        vectors.append(values)

    return vectors


def get_pinecone_index() -> tuple[Pinecone, object, str]:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in your environment.")

    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    namespace = os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    cloud = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    dimension = int(os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))

    pc = Pinecone(api_key=api_key)

    try:
        existing_indexes = pc.list_indexes().names()
    except AttributeError:
        existing_indexes = [item["name"] for item in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        for _ in range(30):
            try:
                if index_name in pc.list_indexes().names():
                    break
            except AttributeError:
                break
            time.sleep(1)

    return pc, pc.Index(index_name), namespace


def main() -> None:
    load_dotenv(find_dotenv())

    pdf_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    pdf_path = pdf_arg if pdf_arg else choose_pdf_file()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    gemini_client = build_gemini_client()
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    chunk_size = int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))

    _, index, namespace = get_pinecone_index()

    pages = extract_pdf_pages(pdf_path)
    vectors: list[dict[str, object]] = []

    for page_data in pages:
        page_number = int(page_data["page"])
        page_text = str(page_data["text"])
        chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = embed_texts(gemini_client, embedding_model, chunks)

        for chunk_index, (chunk_text_value, embedding) in enumerate(zip(chunks, embeddings), start=1):
            vectors.append(
                {
                    "id": f"{pdf_path.stem}-p{page_number}-c{chunk_index}",
                    "values": embedding,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_index,
                        "text": chunk_text_value,
                    },
                }
            )

    if not vectors:
        raise RuntimeError("No chunks were generated from the selected PDF.")

    index.upsert(vectors=vectors, namespace=namespace)
    print(
        f"Uploaded {len(vectors)} chunks from {pdf_path.name} to Pinecone index "
        f"'{os.getenv('PINECONE_INDEX_NAME', DEFAULT_INDEX_NAME)}' in namespace '{namespace}'."
    )


if __name__ == "__main__":
    main()