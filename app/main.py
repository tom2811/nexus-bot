"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import __version__
from app.api import init_router, router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Initializes the hybrid router on startup and cleans up on shutdown.
    """
    logger.info("Starting Nexus-bot application")
    
    # Check if API key is configured
    api_key = os.getenv("API_KEY")
    if not api_key:
        logger.warning("API_KEY not set - authentication disabled (dev mode only)")
    else:
        logger.info("API key authentication enabled")
    
    # Startup: Initialize hybrid router and auto-ingest documents if needed
    vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
    
    # Auto-seed documents from data/ directory
    try:
        from ingest import main as run_ingest
        run_ingest()
    except Exception as e:
        logger.warning(f"Auto-ingest skipped or failed: {e}")

    init_router(vector_persist_dir=vector_persist_dir)
    logger.info("Hybrid router initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nexus-bot application")


app = FastAPI(
    title="Nexus-bot",
    description="Hybrid AI assistant backend combining RAG and controlled tools",
    version=__version__,
    lifespan=lifespan,
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@limiter.limit("10/minute")
def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "ok"}


# Include API router
app.include_router(router)
