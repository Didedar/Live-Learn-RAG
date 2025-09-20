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
    app_name: str = Field("Live-Learn RAG with Ollama", alias="APP_NAME")
    app_env: str = Field("dev", alias="APP_ENV")
    debug: bool = Field(False, alias="DEBUG")
    
    # Database
    db_path: str = Field("./rag.db", alias="DB_PATH")
    
    # Removed Google AI and OpenAI - using only local Ollama
    
    # RAG settings
    chunk_size: int = Field(400, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(40, alias="CHUNK_OVERLAP")
    default_top_k: int = Field(6, alias="DEFAULT_TOP_K")
    
    # Feedback settings
    feedback_penalty_weight: float = Field(-0.3, alias="FEEDBACK_PENALTY_WEIGHT")
    feedback_boost_weight: float = Field(0.5, alias="FEEDBACK_BOOST_WEIGHT")
    
    # Anti-spam settings
    spam_detection_threshold: float = Field(0.3, alias="SPAM_DETECTION_THRESHOLD")
    trusted_user_threshold: float = Field(0.8, alias="TRUSTED_USER_THRESHOLD")
    max_feedback_rate_per_hour: int = Field(10, alias="MAX_FEEDBACK_RATE_PER_HOUR")
    min_correction_length: int = Field(10, alias="MIN_CORRECTION_LENGTH")
    
    # Content quality settings
    min_confidence_score: float = Field(0.3, alias="MIN_CONFIDENCE_SCORE")
    content_filter_enabled: bool = Field(True, alias="CONTENT_FILTER_ENABLED")
    
    # Security
    api_key: Optional[str] = Field(None, alias="API_KEY")
    api_key_header: str = Field("X-API-Key", alias="API_KEY_HEADER")
    
    # Performance
    request_timeout: int = Field(60, alias="REQUEST_TIMEOUT")
    max_retries: int = Field(3, alias="MAX_RETRIES")
    
    # Removed Gemini-specific settings
    
    # Ollama settings (local Llama models) - ALWAYS ENABLED
    use_ollama: bool = Field(True, alias="USE_OLLAMA")  # Always use Ollama
    ollama_url: str = Field("http://localhost:11434", alias="OLLAMA_URL")
    ollama_model: str = Field("llama3.2:latest", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field("nomic-embed-text:latest", alias="OLLAMA_EMBEDDING_MODEL")
    
    # Enhanced RAG settings
    use_enhanced_rag: bool = Field(True, alias="USE_ENHANCED_RAG")
    use_hybrid_search: bool = Field(True, alias="USE_HYBRID_SEARCH")
    mmr_lambda: float = Field(0.15, alias="MMR_LAMBDA")
    retrieval_threshold: float = Field(0.4, alias="RETRIEVAL_THRESHOLD")
    hybrid_alpha: float = Field(0.6, alias="HYBRID_ALPHA")  # Weight for dense vs keyword
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure overlap is less than chunk size."""
        chunk_size = values.get("chunk_size", 400)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    # Removed Google and OpenAI validation methods - not needed
    
    def is_ollama_configured(self) -> bool:
        """Check if Ollama is configured."""
        return bool(self.ollama_url and self.ollama_model)  # Always check Ollama
    
    def get_ai_status(self) -> dict:
        """Get AI configuration status."""
        return {
            "ollama_configured": self.is_ollama_configured(),
            "llm_model": self.ollama_model,
            "embedding_model": "mock_embeddings",
            "use_ollama": self.use_ollama,
            "ollama_url": self.ollama_url
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from old Gemini config


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    try:
        settings = Settings()
        
        # Log configuration status
        if settings.is_ollama_configured():
            logger.info("✅ Ollama configured successfully")
        else:
            logger.warning("⚠️ Ollama not configured - check OLLAMA_URL and OLLAMA_MODEL")
        
        return settings
        
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        # Return settings anyway, let the application decide what to do
        return Settings()


# Global settings instance
settings = get_settings()