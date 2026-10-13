"""Security middleware and utilities."""

import logging
import os
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# API Key auth
security = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
    header_key: Optional[str] = None,
) -> bool:
    """Verify API key from Bearer token or X-API-Key header.
    
    Supports multiple API keys for different client applications.
    Set API_KEYS (comma-separated) or API_KEY (single key) in environment.
    
    Args:
        credentials: Authorization credentials from request
        header_key: Optional X-API-Key header value
        
    Returns:
        True if valid, False otherwise
    """
    # Support both API_KEYS (multiple) and API_KEY (single) for backward compatibility
    api_keys_str = os.getenv("API_KEYS") or os.getenv("API_KEY")
    
    # If no API key configured, allow access (dev mode)
    if not api_keys_str:
        logger.warning("No API_KEY(S) configured - authentication disabled")
        return True
    
    # Parse comma-separated keys and strip whitespace
    valid_keys = {key.strip() for key in api_keys_str.split(",") if key.strip()}
    
    provided = header_key or (credentials.credentials if credentials else None)
    if not provided:
        return False
    
    return provided in valid_keys


async def api_key_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None
) -> None:
    """API key authentication dependency.
    
    Args:
        request: FastAPI request
        credentials: Optional authorization credentials
        
    Raises:
        HTTPException: If authentication fails
    """
    header_key = request.headers.get("X-API-Key")
    if not verify_api_key(credentials, header_key=header_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
