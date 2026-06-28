"""Interactive RAG chat over documents stored in Pinecone.

The script embeds each question, retrieves the nearest document chunks from
Pinecone, and uses those chunks as grounding context for a chat model.

Environment variables:
    GEMINI_API_KEY
    CHAT_MODEL (default: gemini-2.5-flash-lite)
    EMBEDDING_MODEL (default: gemini-embedding-001)
    PINECONE_API_KEY
    PINECONE_INDEX_NAME (default: rag-chat)
    PINECONE_NAMESPACE (default: default)
"""

from __future__ import annotations

import os
from functools import lru_cache
from collections.abc import Iterable

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from pinecone import Pinecone
from pydantic import BaseModel, Field


DEFAULT_INDEX_NAME = "rag-chat"
DEFAULT_NAMESPACE = "default"
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 3072


CHAT_UI_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>RAG Pinecone Chat</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <style>
        :root {
            --bg-1: #f7f0e6;
            --bg-2: #dfe9f3;
            --ink: #1b1b1b;
            --muted: #575757;
            --panel: rgba(255, 255, 255, 0.8);
            --accent: #ff5a1f;
            --accent-2: #0050c8;
            --ok: #0a7f43;
            --line: rgba(27, 27, 27, 0.15);
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--ink);
            font-family: "Space Grotesk", sans-serif;
            background:
                radial-gradient(70vw 50vh at 0% 0%, rgba(255, 90, 31, 0.18), transparent 60%),
                radial-gradient(55vw 40vh at 100% 100%, rgba(0, 80, 200, 0.2), transparent 70%),
                linear-gradient(140deg, var(--bg-1), var(--bg-2));
            display: grid;
            place-items: center;
            padding: 24px;
        }

        .frame {
            width: min(980px, 100%);
            height: min(88vh, 900px);
            border: 1px solid var(--line);
            border-radius: 20px;
            overflow: hidden;
            backdrop-filter: blur(8px);
            background: var(--panel);
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.12);
            display: grid;
            grid-template-rows: auto 1fr auto;
            animation: rise 360ms ease-out;
        }

        @keyframes rise {
            from { transform: translateY(8px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .header {
            border-bottom: 1px solid var(--line);
            padding: 18px 20px;
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 12px;
        }

        .title {
            margin: 0;
            font-size: clamp(1rem, 1.8vw, 1.35rem);
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        .status {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            color: var(--ok);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .messages {
            overflow: auto;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .msg {
            max-width: 82%;
            padding: 12px 14px;
            border-radius: 14px;
            line-height: 1.45;
            border: 1px solid var(--line);
            white-space: pre-wrap;
            animation: fade 200ms ease-out;
        }

        @keyframes fade {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user {
            align-self: flex-end;
            background: color-mix(in srgb, var(--accent) 18%, white);
            border-color: color-mix(in srgb, var(--accent) 35%, var(--line));
        }

        .assistant {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.9);
        }

        .sources {
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px dashed var(--line);
            color: var(--muted);
            font-size: 0.86rem;
        }

        .composer {
            border-top: 1px solid var(--line);
            padding: 14px;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 10px;
        }

        .input {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--line);
            padding: 12px 14px;
            font: inherit;
            background: rgba(255, 255, 255, 0.9);
            outline: none;
        }

        .input:focus {
            border-color: var(--accent-2);
            box-shadow: 0 0 0 3px rgba(0, 80, 200, 0.12);
        }

        .btn {
            border: 0;
            border-radius: 12px;
            padding: 0 16px;
            min-height: 44px;
            background: linear-gradient(100deg, var(--accent), color-mix(in srgb, var(--accent) 65%, var(--accent-2)));
            color: white;
            font: inherit;
            font-weight: 700;
            cursor: pointer;
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        @media (max-width: 720px) {
            body { padding: 10px; }
            .frame {
                height: 95vh;
                border-radius: 14px;
            }
            .msg { max-width: 94%; }
            .composer {
                grid-template-columns: 1fr;
            }
            .btn { width: 100%; }
        }
    </style>
</head>
<body>
    <main class="frame">
        <header class="header">
            <h1 class="title">RAG Chat · Pinecone + Gemini</h1>
            <div id="status" class="status">ready</div>
        </header>

        <section id="messages" class="messages"></section>

        <form id="composer" class="composer">
            <input id="messageInput" class="input" placeholder="Ask about your uploaded documents..." autocomplete="off" />
            <button id="sendBtn" class="btn" type="submit">Send</button>
        </form>
    </main>

    <script>
        const messagesEl = document.getElementById("messages");
        const formEl = document.getElementById("composer");
        const inputEl = document.getElementById("messageInput");
        const sendBtn = document.getElementById("sendBtn");
        const statusEl = document.getElementById("status");

        function addMessage(role, text, sources) {
            const wrap = document.createElement("div");
            wrap.className = `msg ${role}`;
            wrap.textContent = text;

            if (Array.isArray(sources) && sources.length) {
                const src = document.createElement("div");
                src.className = "sources";
                src.textContent = `Sources: ${sources.join(" | ")}`;
                wrap.appendChild(src);
            }

            messagesEl.appendChild(wrap);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        async function sendMessage(message) {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                let detail = "Chat request failed.";
                try {
                    const payload = await response.json();
                    if (payload && payload.detail) {
                        detail = String(payload.detail);
                    }
                } catch {
                    // Fall back to default message.
                }
                throw new Error(detail);
            }

            return response.json();
        }

        formEl.addEventListener("submit", async (event) => {
            event.preventDefault();
            const question = inputEl.value.trim();
            if (!question) {
                return;
            }

            addMessage("user", question);
            inputEl.value = "";
            inputEl.focus();

            sendBtn.disabled = true;
            statusEl.textContent = "thinking";

            try {
                const result = await sendMessage(question);
                addMessage("assistant", result.answer || "No answer returned by the model.", result.sources || []);
                statusEl.textContent = "ready";
            } catch (err) {
                addMessage("assistant", `Error: ${err.message}`);
                statusEl.textContent = "error";
            } finally {
                sendBtn.disabled = false;
            }
        });

        addMessage("assistant", "I am ready. Ask any question about the PDFs uploaded to your Pinecone index.");
        inputEl.focus();
    </script>
</body>
</html>
"""


def build_gemini_client() -> genai.Client:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment.")

    return genai.Client(api_key=gemini_key)


def get_index() -> tuple[object, str]:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Set PINECONE_API_KEY in your environment.")

    index_name = os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME)
    namespace = os.getenv("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name), namespace


def embed_query(client: genai.Client, model: str, query: str) -> list[float]:
    response = client.models.embed_content(model=model, contents=[query])
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None or not embeddings:
        raise RuntimeError("Unexpected Gemini embed response shape")

    values = getattr(embeddings[0], "values", None)
    if not isinstance(values, list):
        raise RuntimeError("Could not parse embedding vector from Gemini response")

    return values


def retrieve_context(index: object, namespace: str, query_vector: list[float], top_k: int = 4) -> list[dict[str, object]]:
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return list(result.matches or [])


def get_match_value(match: object, key: str, default: object = None) -> object:
    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


def format_context(matches: Iterable[dict[str, object]]) -> str:
    blocks: list[str] = []
    for match in matches:
        metadata = get_match_value(match, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        text = str(metadata.get("text", "")).strip()
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        score = float(get_match_value(match, "score", 0.0) or 0.0)
        if text:
            blocks.append(f"Source: {source} | Page: {page} | Score: {score:.4f}\n{text}")

    return "\n\n---\n\n".join(blocks)


def answer_question(client: genai.Client, model: str, question: str, context: str) -> str:
    prompt = (
        "You are a document question-answering assistant. Use only the provided context. "
        "If the context is not enough, say so clearly.\n\n"
        f"Context:\n{context or 'No relevant context found.'}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts:
            first_part = parts[0]
            part_text = getattr(first_part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text

    return "No answer returned by the model."


@lru_cache(maxsize=1)
def get_rag_runtime() -> tuple[genai.Client, str, str, object, str]:
    load_dotenv(find_dotenv())

    gemini_client = build_gemini_client()
    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    chat_model = os.getenv("CHAT_MODEL", DEFAULT_GEMINI_CHAT_MODEL)
    index, namespace = get_index()
    return gemini_client, embedding_model, chat_model, index, namespace


def run_chat_turn(question: str, top_k: int = 4) -> tuple[str, list[dict[str, object]]]:
    if not question.strip():
        raise ValueError("question cannot be empty")

    gemini_client, embedding_model, chat_model, index, namespace = get_rag_runtime()
    query_vector = embed_query(gemini_client, embedding_model, question)
    matches = retrieve_context(index, namespace, query_vector, top_k=top_k)
    context = format_context(matches)
    answer = answer_question(gemini_client, chat_model, question, context)
    return answer, matches


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=12)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(title="RAG Pinecone Chat")


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return CHAT_UI_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat_api(payload: ChatRequest) -> ChatResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        answer, matches = run_chat_turn(question, top_k=payload.top_k)
    except Exception as exc:  # Keep runtime errors visible for easier setup debugging.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources: list[str] = []
    for match in matches:
        metadata = get_match_value(match, "metadata", {})
        if not isinstance(metadata, dict):
            continue

        source = str(metadata.get("source", "unknown"))
        page = str(metadata.get("page", "?"))
        source_label = f"{source} p.{page}"
        if source_label not in sources:
            sources.append(source_label)

    return ChatResponse(answer=answer, sources=sources)


def main() -> None:
    load_dotenv(find_dotenv())

    print("RAG chat ready. Type a question or 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        answer, _ = run_chat_turn(question)

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()