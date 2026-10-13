#!/usr/bin/env python3
"""Ingestion script for multi-tenant document collections in Nexus-bot."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.loaders import load_document
from app.chunking import chunk_documents
from app.vector_store import VectorStore


def ingest_tenant(data_dir: Path, collection_name: str, persist_dir: str = "./chroma_db"):
    """Ingest all markdown/text/pdf files from a directory into a ChromaDB collection."""
    if not data_dir.exists():
        print(f"Skipping {data_dir}: directory does not exist")
        return

    vector_store = VectorStore(persist_directory=persist_dir, collection_name=collection_name)
    existing = vector_store.count()
    if existing > 0:
        print(f"Collection '{collection_name}' already has {existing} chunks, skipping ingest")
        return

    all_chunks = []

    for file_path in data_dir.glob("**/*"):
        if file_path.suffix.lower() in [".md", ".txt", ".pdf"] and file_path.is_file():
            print(f"Loading [{collection_name}] -> {file_path.name}")
            docs = load_document(str(file_path))
            chunks = chunk_documents(docs, chunk_size=500, overlap=50)
            all_chunks.extend(chunks)

    if all_chunks:
        print(f"Embedding and adding {len(all_chunks)} chunks to collection '{collection_name}'...")
        vector_store.add_documents(all_chunks)
        print(f"Successfully ingested {len(all_chunks)} chunks into '{collection_name}'!")
    else:
        print(f"No documents found for collection '{collection_name}'.")


def main(data_root: Path | None = None, persist_dir: str | None = None):
    root_dir = Path(__file__).parent
    persist_dir = persist_dir or os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
    data_root = data_root or (root_dir / "data")

    if not data_root.exists():
        print("No 'data' folder found.")
        return

    # Optional: TENANT_IDS=kickskraze,pos  — ingest only those folders.
    allowed = os.getenv("TENANT_IDS")
    allowed_set = (
        {name.strip().lower() for name in allowed.split(",") if name.strip()}
        if allowed
        else None
    )

    # Ingest each subdirectory under data/ as its own collection
    for tenant_dir in data_root.iterdir():
        if tenant_dir.is_dir():
            tenant_name = tenant_dir.name.lower()
            if allowed_set is not None and tenant_name not in allowed_set:
                print(f"\n--- Skipping Tenant: {tenant_name} (not in TENANT_IDS) ---")
                continue
            print(f"\n--- Ingesting Tenant: {tenant_name} ---")
            ingest_tenant(tenant_dir, collection_name=tenant_name, persist_dir=persist_dir)


if __name__ == "__main__":
    main()
