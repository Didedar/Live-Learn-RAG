"""API dependencies."""
from sqlalchemy.orm import Session
from typing import Optional
from ..database import get_db
from ..core.security import optional_auth

# Re-export for convenience
__all__ = ["get_db", "optional_auth"]