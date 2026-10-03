"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import init_router, router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Initializes the hybrid router on startup and cleans up on shutdown.
    """
    # Startup: Initialize hybrid router
    vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
    init_router(vector_persist_dir=vector_persist_dir)
    
    yield
    
    # Shutdown: Cleanup if needed
    pass


app = FastAPI(
    title="Nexus-bot",
    description="Hybrid AI assistant backend combining RAG and controlled tools",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# Include API router
app.include_router(router)
