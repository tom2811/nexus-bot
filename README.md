<div align="center">

# Nexus-bot

**Hybrid AI assistant — RAG + database tools**

A reusable FastAPI backend that routes each query to document retrieval, structured data, or both. Built for e-commerce; adaptable to any domain.

</div>

---

## Overview

Most assistants do either document retrieval (RAG) or database queries — not both. Nexus-bot detects intent and routes to the right source, or combines them.

```
"Is size 10 available and what's the return policy?"
→ inventory database + policy documents
→ one answer, with sources and a trace
```

---

## Highlights

- **Intent-based routing** — automatically chooses RAG, database tools, or a hybrid of both.
- **Resilient, swappable LLM chain** — Groq → Groq fallback → local Ollama, with per-provider circuit breaking and budget-bounded retries.
- **Versioned host contract** — hosts declare the shape of their data (`grounding.kind`); unknown kinds are rejected, never mis-rendered.
- **Observability by default** — a structured JSON trace (route, retrieval, tools, model, latency, fallbacks) on every request.
- **Multi-tenant & multi-key** — isolated vector collections and API keys per client.
- **Measured, not assumed** — a routing/metrics golden set and load-test harness wired into CI.

---

## How It Works

```json
POST /api/v1/query
{
  "message": "Is size 10 available and what's the return policy?",
  "tenant_id": "kickskraze",
  "api_version": "1",
  "grounding": { "kind": "product",
                 "data": { "name": "Air Max", "availableSizes": ["10"], "retailPrice": 180 } }
}
```

```json
{
  "answer": "Size 10 is in stock at $180. Returns accepted within 14 days...",
  "route_type": "hybrid",
  "rag_used": true,
  "tools_used": ["caller_catalog"],
  "sources": [{ "source": "return_policy.md" }],
  "api_version": "1",
  "trace": {
    "trace_id": "a1b2c3d4e5f6",
    "route": "hybrid",
    "retrieval": [{ "source": "return_policy.md", "distance": 0.21 }],
    "tools": ["caller_catalog"],
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "fallbacks": [],
    "latency_ms": 812.4
  }
}
```

```
Client → Intent Detection ─┬─ RAG      (ChromaDB)
                           └─ Tools    (PostgreSQL)
                 ↓
         LLM provider chain (Groq → fallback → Ollama)
                 ↓
      Answer + sources + trace
```

---

## Host Contract & Grounding

Hosts declare the **shape** of the data they send (`grounding.kind`); the bot never guesses keys. Unknown kinds return `422` with the list of valid kinds. `api_version` is echoed on every request and response.

| `kind` | Shape | Tool | Rendering |
|--------|-------|------|-----------|
| `product` | `name`, `brand`, `colorway`, `retailPrice`, `availableSizes[]`, `inStock` | `caller_catalog` | Deterministic (no LLM) |
| `order` | `order_id`, `status`, `total_amount`, `customer_name` | `caller_order` | Deterministic (no LLM) |
| `generic` | any dict | `caller_data` | LLM over raw data |

The legacy `product_context` field still works — auto-classified as `product` when it validates, otherwise `generic`. `product` also accepts snake_case aliases (`available_sizes`, `in_stock`, `retail_price`).

---

## LLM Providers & Resilience

The LLM is a swappable chain: **Groq primary** → **Groq fallback** → **Ollama** (when `OLLAMA_BASE_URL` is set). A per-provider circuit breaker skips failing providers, and retries use exponential backoff that honors `Retry-After` while staying inside a per-request budget (`LLM_TIMEOUT`). Override the order with `LLM_PROVIDER_CHAIN`. Rate limits are per-account, so key rotation doesn't help — fallback and backoff do. All providers down → `503`; request timeout → `504`.

---

## Quick Start

```bash
uv sync
cp .env.example .env        # set GROQ_API_KEY, or OLLAMA_BASE_URL for local
uv run uvicorn app.main:app --reload
```

---

## API Examples

```bash
# Document question
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "What is the return policy?", "tenant_id": "kickskraze"}'

# Product availability (declared grounding)
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "Is size 10 available?", "tenant_id": "kickskraze",
       "grounding": {"kind": "product",
                     "data": {"name": "Air Max", "availableSizes": ["10"]}}}'

# Order status (declared grounding)
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "Where is my order?",
       "grounding": {"kind": "order",
                     "data": {"order_id": 1001, "status": "PAID", "total_amount": 180}}}'
```

---

## Multi-Tenant Setup

Each tenant gets an isolated ChromaDB collection. Set `TENANT_IDS=kickskraze,pos` to ingest only selected `data/` folders, and `API_KEYS=kickskraze_key,healthcare_key` for one key per client. Requests are scoped by `tenant_id`.

---

## Evaluation & Testing

```bash
# routing/metrics golden set (fixture = deterministic self-test, used in CI)
uv run python run_evaluation.py --backend fixture --judge none --min-route 1.0
# live backend with embedding-based faithfulness judge
uv run python run_evaluation.py --backend live --judge embedding --min-route 0.95

# load test: async latency percentiles + error budget
uv run python loadtest/run_loadtest.py

# 153 tests across routing, RAG, tools, API, providers, tracing, and metrics
uv run pytest tests/ -v
```

`run_evaluation.py` exits `0` (passed), `1` (thresholds failed), or `2` (environment unavailable); `--json out.json` writes machine-readable results. Tests needing a live Postgres/LLM skip gracefully.

---

## Deployment

```bash
# LLM
GROQ_API_KEY=<key>
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
LLM_TIMEOUT=60
# LLM_PROVIDER_CHAIN=            # optional explicit fallback order
# OLLAMA_BASE_URL=http://localhost:11434

# Auth (single key, or comma-separated list for multiple clients)
API_KEY=<key>
# API_KEYS=app1_key,app2_key

# Vector store
VECTOR_PERSIST_DIR=./chroma_db
TENANT_IDS=kickskraze,app2       # optional

# CORS / database
CORS_ORIGINS=https://yourapp.com
DB_HOST=<host>                   # optional
DB_NAME=nexus_bot
```

Railway/Docker auto-ingests documents on startup, persists `./chroma_db` on a volume, and exposes `GET /health`.

---

## Tech Stack

FastAPI · ChromaDB · PostgreSQL · Groq / Ollama · sentence-transformers · slowapi · pytest
