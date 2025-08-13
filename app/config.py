from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    app_env: str = Field("dev", alias="APP_ENV")
    db_path: str = Field("./rag.db", alias="DB_PATH")

    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    embedding_model: str = Field("nomic-embed-text", alias="EMBEDDING_MODEL")
    llm_model: str = Field("llama3.1:8b", alias="LLM_MODEL")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
