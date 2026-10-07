"""Security middleware and utilities."""

import logging
import os
from typing import Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# API Key auth
security = HTTPBearer(auto_error=False)


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials]) -> bool:
    """Verify API key from Authorization header.
    
    Args:
        credentials: Authorization credentials from request
        
    Returns:
        True if valid, False otherwise
    """
    # Get expected API key from environment
    expected_key = os.getenv("API_KEY")
    
    # If no API key configured, allow access (dev mode)
    if not expected_key:
        logger.warning("No API_KEY configured - authentication disabled")
        return True
    
    # Check credentials
    if not credentials:
        return False
    
    return credentials.credentials == expected_key


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
    if not verify_api_key(credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
