"""Application configuration - Safe version that doesn't crash without API key."""

import os
from functools import lru_cache
from typing import Optional

from loguru import logger
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with safe defaults."""
    
    # App settings
    app_name: str = Field("Live-Learn RAG with Gemini", alias="APP_NAME")
    app_env: str = Field("dev", alias="APP_ENV")
    debug: bool = Field(False, alias="DEBUG")
    
    # Database
    db_path: str = Field("./rag.db", alias="DB_PATH")
    
    # Google AI Settings - ИСПРАВЛЕНО: сделал необязательным
    google_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")
    llm_model: str = Field("gemini-2.0-flash-exp", alias="LLM_MODEL")
    embedding_model: str = Field("text-embedding-004", alias="EMBEDDING_MODEL")
    
    # Alternative: Use OpenAI for embeddings
    use_openai_embeddings: bool = Field(False, alias="USE_OPENAI_EMBEDDINGS")
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    
    # RAG settings
    chunk_size: int = Field(400, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(40, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(6, alias="DEFAULT_TOP_K")
    
    # Feedback settings
    feedback_penalty_weight: float = Field(-0.3, alias="FEEDBACK_PENALTY_WEIGHT")
    feedback_boost_weight: float = Field(0.5, alias="FEEDBACK_BOOST_WEIGHT")
    
    # Security
    api_key: Optional[str] = Field(None, alias="API_KEY")
    api_key_header: str = Field("X-API-Key", alias="API_KEY_HEADER")
    
    # Performance
    request_timeout: int = Field(60, alias="REQUEST_TIMEOUT")
    max_retries: int = Field(3, alias="MAX_RETRIES")
    
    # Google AI specific settings
    gemini_temperature: float = Field(0.1, alias="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(8192, alias="GEMINI_MAX_TOKENS")
    gemini_top_p: float = Field(0.8, alias="GEMINI_TOP_P")
    gemini_top_k: int = Field(40, alias="GEMINI_TOP_K")
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure overlap is less than chunk size."""
        chunk_size = values.get("chunk_size", 400)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    def validate_google_api_key(self) -> str:
        """Validate Google API key when needed."""
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required for Gemini functionality. "
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        
        if self.google_api_key == "your_google_api_key_here":
            raise ValueError(
                "Please replace 'your_google_api_key_here' with your actual Google API key"
            )
        
        return self.google_api_key
    
    def validate_openai_api_key(self) -> str:
        """Validate OpenAI API key when needed."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI embeddings")
        
        return self.openai_api_key
    
    def is_gemini_configured(self) -> bool:
        """Check if Gemini is properly configured."""
        return (
            self.google_api_key is not None 
            and self.google_api_key != "your_google_api_key_here"
            and len(self.google_api_key) > 10
        )
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured."""
        return (
            self.use_openai_embeddings
            and self.openai_api_key is not None
            and self.openai_api_key != "your_openai_api_key_here"
            and len(self.openai_api_key) > 10
        )
    
    def get_ai_status(self) -> dict:
        """Get AI configuration status."""
        return {
            "gemini_configured": self.is_gemini_configured(),
            "openai_configured": self.is_openai_configured(),
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "use_openai_embeddings": self.use_openai_embeddings
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    try:
        settings = Settings()
        
        # Log configuration status
        if settings.is_gemini_configured():
            logger.info("✅ Gemini AI configured successfully")
        else:
            logger.warning("⚠️ Gemini AI not configured - set GOOGLE_API_KEY")
        
        return settings
        
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # Return settings anyway, let the application decide what to do
        return Settings()


# Global settings instance
settings = get_settings()