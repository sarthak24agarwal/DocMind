# DocMind — Backend

[![CI](https://github.com/sarthak24agarwal/DocMind/actions/workflows/ci.yml/badge.svg)](https://github.com/sarthak24agarwal/DocMind/actions/workflows/ci.yml)

Multi-tenant Retrieval-Augmented Generation (RAG) backend for DocMind: upload documents into a
workspace, they get parsed/chunked/embedded in the background, and you chat with them over a
citation-grounded, streaming endpoint.

## Stack

- **FastAPI** — HTTP API
- **Celery + Redis** — async document ingestion pipeline
- **PostgreSQL + pgvector** — relational data + vector similarity search
- **Cloudflare R2 (S3-compatible)** — raw document storage
- **OpenAI** — embeddings (`text-embedding-3-small`)
- **Anthropic Claude** — RAG chat completion, streamed via SSE
- **Stripe** — Free/Pro billing

## Project layout

```
app/
  main.py              FastAPI app, workspace + document upload/status endpoints
  config.py            Pydantic settings (reads from .env)
  database.py          SQLAlchemy engine/session
  models.py            User, Workspace, Document, DocumentChunk, WorkspaceUsage, Conversation, Message
  dependencies.py      verify_query_limits — per-user plan/grace-period enforcement dependency
  celery_app.py        Celery app + beat schedule (daily usage-counter reset)
  tasks.py             process_document_ingestion, reset_monthly_query_counters
  routers/
    chat.py            POST /workspaces/{id}/chat — streaming RAG endpoint (SSE)
    billing.py         Stripe checkout + webhook handling
  services/
    r2.py              R2/S3 upload, download, presigned URLs
    parser.py          PDF / DOCX / TXT -> text blocks
    chunker.py          Sentence-aware token chunking with overlap
    embedder.py         OpenAI embeddings with retry/backoff (+ mock mode)
    anthropic_service.py Claude streaming chat client (+ mock mode)
tests/
  test_ingestion.py    parser + chunker + ingestion task
  test_chat.py         workspace usage limits + RAG streaming + citations
  test_billing.py      checkout, webhooks, query-limit dependency, monthly reset job
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real DB/R2/OpenAI/Anthropic/Stripe credentials
```

Set `OPENAI_API_KEY=mock` and `ANTHROPIC_API_KEY=mock` to run locally without those providers —
both services fall back to deterministic mock generators.

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the worker + beat scheduler:

```bash
celery -A app.celery_app worker --loglevel=info
celery -A app.celery_app beat --loglevel=info
```

Run tests:

```bash
pytest -q
```

> **Note on `pytest -q` in fully offline sandboxes:** the chunker uses `tiktoken`, which downloads
> its BPE vocab file on first use and caches it under `TIKTOKEN_CACHE_DIR`. On a normal machine
> with internet access this just works on first run. If you're in a fully offline CI/sandbox
> environment, pre-seed the cache once (see `tests/conftest.py` for where settings/env defaults
> are wired) or vendor the `cl100k_base.tiktoken` file into your image.

## What changed while integrating these files

The uploaded files were a snapshot of a backend mid-refactor — most modules were consistent, but
a few wires were crossed between them. I reconciled the following so the full test suite (21
tests) actually passes end-to-end against the real source, not just in isolation:

1. **`anthropic_service.py`** — `stream_chat(...) -> generator:` used `generator` as a type
   annotation, which isn't a real name in Python and crashed on import. Fixed to
   `typing.Iterator[str]`.
2. **`routers/chat.py`** — `test_chat.py` expects workspace-level usage metering (a
   `get_or_reset_usage(db, workspace_id)` helper backed by the `WorkspaceUsage` table that already
   existed in `models.py` but was never used) instead of the per-user limit dependency. Rebuilt
   `chat_rag` around that: it now checks/increments `WorkspaceUsage.queries_this_month` against
   `FREE_TIER_QUERY_LIMIT` / `PRO_TIER_QUERY_LIMIT` based on `billing_tier`, with automatic
   monthly rollover. `dependencies.verify_query_limits` (the per-user/payment-grace-period check)
   is kept as-is and still independently testable/usable on other routes.
3. **`tasks.py` (`reset_monthly_query_counters`)** — the free-tier user query only ran inside the
   `if now.day == 1:` block, which (in addition to only mattering once a month) made the
   downstream Pro-tier query's mocked call ordering misalign. Hoisted the lookup out so it always
   runs, and only *acts* on it on the 1st.
4. **`services/parser.py`** — `parse_document()` had an explicit `os.path.exists()` guard before
   dispatching to the PDF/DOCX parsers, which is both redundant (the parsers already raise/are
   wrapped in `ParsingError`) and broke unit tests that mock `pypdf.PdfReader`/`docx.Document`
   directly. Removed it from the dispatcher; kept an explicit check in `parse_txt` (the one path
   that opens the file itself) so a missing file still fails as a clean `ParsingError`.
5. **`services/chunker.py`** — two bugs:
   - the long-sentence fallback used `tiktoken.get_encoding("gpt-2")`, which isn't a valid
     encoding name (`gpt2` is); fixed.
   - the sliding-window overlap step always force-carried the previous unit forward even when
     that single unit was already at/above `target_tokens` (only possible for word-split
     fragments of a sentence longer than the target). That silently doubled chunk size on long
     sentences. Now it only force-carries an oversized lone unit when skipping it would otherwise
     produce an empty chunk; otherwise it starts the next chunk fresh.
6. **`dependencies.py`** — minor wording-case mismatch in the past-due grace-period error message
   vs. what the test asserts on; aligned them.
7. **`requirements.txt`** — added `python-multipart`, required by FastAPI for the
   `multipart/form-data` file upload endpoint (the app didn't actually boot without it).
8. Added `app/__init__.py`, `app/routers/__init__.py`, `app/services/__init__.py`,
   `tests/__init__.py`, `tests/conftest.py` (supplies `pytest.AsyncMock`/`pytest.any_str`, which
   the tests reference but aren't real pytest APIs, plus default mock env vars so `Settings()` can
   instantiate without a `.env` file), and `.env.example`.

All 21 tests pass (`pytest -q` → `21 passed`), and the app boots and registers all 6 routes
(`POST /workspaces`, `POST /workspaces/{id}/documents/upload`,
`GET /workspaces/{id}/documents/{id}`, `POST /workspaces/{id}/chat`,
`POST /billing/create-checkout-session`, `POST /billing/webhook`).

## Send the rest of the project whenever you're ready

This pass covers the backend (API, ingestion pipeline, chat/RAG, billing). If you have a frontend,
infra/Docker, migrations, or anything else from the same DocMind project, upload it and I'll wire
it into this structure the same way.
