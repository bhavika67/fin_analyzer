# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_store: str = "faiss"
    faiss_index_path: str = "data/embeddings/faiss_index"

    database_url: str = "sqlite:///data/processed/fin.db"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()