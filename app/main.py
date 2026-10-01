"""FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Nexus-bot",
    description="Hybrid AI assistant backend combining RAG and controlled tools",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
