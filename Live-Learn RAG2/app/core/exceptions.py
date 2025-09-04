"""Custom exceptions for the RAG system."""

from typing import Any, Dict, Optional


class RAGException(Exception):
    """Base exception for RAG system."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result = {
            "error": self.error_code,
            "message": self.message
        }
        if self.details:
            result["details"] = self.details
        return result


class ValidationError(RAGException):
    """Raised when input validation fails."""
    pass


class DatabaseError(RAGException):
    """Raised when database operations fail."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMError(RAGException):
    """Raised when LLM operations fail."""
    pass


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    pass


class IngestionError(RAGException):
    """Raised when document ingestion fails."""
    pass


class FeedbackError(RAGException):
    """Raised when feedback processing fails."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass


def get_status_code(exception: RAGException) -> int:
    """Get HTTP status code for RAG exception."""
    status_map = {
        ValidationError: 400,
        ConfigurationError: 400,
        FeedbackError: 400,
        DatabaseError: 500,
        EmbeddingError: 500,
        LLMError: 500,
        RetrievalError: 500,
        IngestionError: 500,
        RAGException: 500,
    }
    
    return status_map.get(type(exception), 500)