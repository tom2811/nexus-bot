<div align="center">

# Nexus-bot

**Hybrid AI assistant combining RAG + database tools**

A reusable FastAPI backend that intelligently routes queries between document knowledge and structured data. Built for e-commerce, adaptable to any domain.

</div>

---

## Overview

Most AI assistants use either document retrieval (RAG) or database queries, but not both. Nexus-bot automatically detects query intent and routes to the right data source—or combines both when needed.

```bash
Query: "Is size 10 available and what's the return policy?"
→ Checks inventory database + retrieves policy documents
→ Returns unified answer with sources
```

Real-world use case: Powers AI shopping assistants for e-commerce platforms.

---

## Features

- **Automatic intent detection** — RAG, tools, or both
- **Multi-tenant** — Isolated document collections per client
- **Multi-key auth** — Secure isolation between apps
- **External catalog support** — No database sync required
- **131 tests** — Production-ready
- **5-minute setup** — Groq or local Ollama

---

## Quick Start

```bash
# 1. Install
uv sync

# 2. Configure
cp .env.example .env
# Add GROQ_API_KEY or OLLAMA_BASE_URL

# 3. Run
uv run uvicorn app.main:app --reload
```

Visit [http://localhost:8000](http://localhost:8000)

---

## How It Works

**1. You send a query**
```json
POST /api/v1/query
{
  "message": "Is size 10 available and what's the return policy?",
  "tenant_id": "kickskraze",
  "product_context": { "name": "Air Max", "availableSizes": ["10"] }
}
```

**2. Intent detection routes it**
- Detects "available" → Tool query
- Detects "return policy" → RAG query
- Uses both sources

**3. Get a combined answer**
```json
{
  "answer": "Size 10 is in stock at $180. Returns accepted within 14 days...",
  "route_type": "hybrid",
  "sources": [{"source": "return_policy.md"}],
  "tools_used": ["caller_catalog"]
}
```

---

## Architecture

```
Client Request
    ↓
Intent Detection (keyword-based)
    ↓
  ┌─────┴─────┐
  │           │
 RAG      Database Tools
  │           │
Vector     PostgreSQL
Store      (optional)
  │           │
  └─────┬─────┘
        ↓
    LLM Synthesis
        ↓
   JSON Response
```

**Key Components:**
- **HybridRouter** — Intent detection + orchestration
- **RAGPipeline** — Multi-tenant document retrieval
- **VectorStore** — ChromaDB with isolated collections
- **Tools** — Validated database queries (inventory, orders, search)
- **Security** — Multi-key API authentication

---

## Multi-Tenant Setup

Isolated document collections per tenant:

```bash
data/
├── kickskraze/         # Sneaker marketplace docs
│   └── return_policy.md
├── healthcare/         # Medical app docs
│   └── hipaa_policy.md
└── pos/               # Point-of-sale docs
    └── operations.md
```

**Query with tenant isolation:**
```bash
curl -X POST /api/v1/query \
  -H "X-API-Key: kickskraze_key" \
  -d '{"message": "return policy?", "tenant_id": "kickskraze"}'
# Only searches kickskraze/ documents
```

**Environment config:**
```bash
# Ingest only specific tenants
TENANT_IDS=kickskraze,pos

# Multi-key auth (one per client)
API_KEYS=kickskraze_key,healthcare_key,pos_key
```

---

## API Examples

### Document Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your_key" \
  -d '{"message": "What is the return policy?", "tenant_id": "kickskraze"}'
```

Response:
```json
{
  "answer": "Returns accepted within 14 days in deadstock condition...",
  "route_type": "rag_only",
  "sources": [{"source": "return_policy.md"}]
}
```

### Product Query (External Catalog)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: your_key" \
  -d '{
    "message": "Is size 10 available?",
    "tenant_id": "kickskraze",
    "product_context": {
      "name": "Air Max",
      "availableSizes": ["10"],
      "retailPrice": 180
    }
  }'
```

### Hybrid Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key": "your_key" \
  -d '{
    "message": "Is size 10 available and what is the return policy?",
    "product_context": {...}
  }'
```

---

## Testing

**131 tests** across all components:

```bash
uv run pytest tests/ -v
# 9 security tests (multi-key auth)
# 15 RAG pipeline tests
# 12 database tool tests
# 28 API endpoint tests
# 7 routing tests
```

---

## Deployment

**Environment variables:**
```bash
# LLM
GROQ_API_KEY=<key>

# Auth (comma-separated for multiple clients)
API_KEYS=app1_key,app2_key

# Vector store
VECTOR_PERSIST_DIR=./chroma_db
TENANT_IDS=kickskraze,app2  # Optional

# CORS
CORS_ORIGINS=https://yourapp.com

# Database (optional)
DB_HOST=<host>
DB_NAME=nexus_bot
```

**Railway/Docker:**
- Auto-ingests documents on startup
- Persistent volume at `./chroma_db`
- Health check: `GET /health`

---

## Reusability

**Built for e-commerce, adaptable to any domain:**

| Domain | Documents | Tools |
|--------|-----------|-------|
| **E-commerce** | Policies, guides | Inventory, orders |
| **Healthcare** | Protocols, HIPAA | Appointments, records |
| **Legal** | Case law, contracts | Document search |
| **Support** | KB articles | Ticket status |

**What changes:** Your `data/` docs + database schema  
**What stays:** RAG, routing, auth, API contracts

---

## Tech Stack

FastAPI · ChromaDB · PostgreSQL · Groq/Ollama · sentence-transformers · pytest
