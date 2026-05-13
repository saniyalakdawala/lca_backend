"""
app/core/config.py
All environment-driven configuration for the Sustainability Intelligence Engine.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # ── App ──────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ── Groq API ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = ""

    GROQ_MODEL_MATCHING: str = "llama3-70b-8192"
    GROQ_MODEL_SYNTHESIS: str = "llama-3.3-70b-versatile"

    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048

    # ── Vector DB ────────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "ecoinvent-lca-index"
    PINECONE_DIMENSION: int = 384

    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: str = ""

    VECTOR_BACKEND: str = "pinecone"

    # ── Embedding Model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 256

    # ── LCA Data Paths ───────────────────────────────────────────────────────
    BOTTLES_PROCESSES_CSV: str = str(DATA_DIR / "bottles_processes.csv")

    LCA_FULL_RAG_CSV: str = str(
        DATA_DIR / "lca_full_rag_data_units.csv"
    )

    USDA_FLOWS_CSV: str = str(DATA_DIR / "usda_flows.csv")

    OPENLCA_FLOWS_CSV: str = str(
        DATA_DIR / "openlca_flows.csv"
    )

    # ── Scoring Engine Weights ───────────────────────────────────────────────
    SCORE_WEIGHT_CRITICALITY: float = 0.30
    SCORE_WEIGHT_TOXICITY: float = 0.35
    SCORE_WEIGHT_GHG: float = 0.35

    # ── GHG Benchmark ────────────────────────────────────────────────────────
    GHG_BENCHMARK_KG_CO2_KWH: float = 0.71

    # ── Grade Thresholds ─────────────────────────────────────────────────────
    GRADE_A_MAX: float = 30.0
    GRADE_B_MAX: float = 60.0
    GRADE_C_MAX: float = 80.0

    # ── RAG Retrieval ────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.60
    RAG_CACHE_TTL_SECONDS: int = 3600
    RAG_FLOWS_SAMPLE: int = 4000

    # ── Redis ────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = False

    # ── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # ── Charts ───────────────────────────────────────────────────────────────
    CHART_DPI: int = 150
    CHART_STYLE: str = "seaborn-v0_8-whitegrid"


settings = Settings()