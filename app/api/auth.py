# app/api/auth.py
from fastapi import Header, HTTPException
from app.models.session import UserSession

def get_current_session(
    authorization: str = Header(None)
) -> UserSession:
    """
    Mock authentication dependency returns a mock user.
    
    Args:
        authorization: Authorization header (Bearer <token>)
    
    Returns:
        UserSession: Mock user session
    
    Raises:
        HTTPException: If no auth header provided
    """
    # Require some auth header (even if we ignore it)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )
    
    # Return mock user (same user every time)
    return UserSession(
        id="mock-user-123",
        email="testuser@example.com"
    )