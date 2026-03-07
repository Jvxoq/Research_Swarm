from pydantic import BaseModel

class UserSession(BaseModel):
    """
    Represents an authenticated user session.
    """
    id: str
    email: str