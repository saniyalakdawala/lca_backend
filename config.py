"""
app/core/config.py
All environment-driven configuration for the Sustainability Intelligence Engine.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Groq API (LLaMA / Mixtral) ───────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL_MATCHING: str = "llama3-70b-8192"        # high-accuracy material matching
    GROQ_MODEL_SYNTHESIS: str = "llama-3.3-70b-versatile"   # mixtral-8x7b-32768 decommissioned    # eco-design narrative generation
    GROQ_TEMPERATURE: float = 0.1                        # near-deterministic for LCA tasks
    GROQ_MAX_TOKENS: int = 2048

    # ── Vector DB (Pinecone primary / Weaviate fallback) ─────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "ecoinvent-lca-index"
    PINECONE_DIMENSION: int = 384                       # sentence-transformers/all-MiniLM-L6-v2

    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: str = ""
    VECTOR_BACKEND: str = "pinecone"                   # "pinecone" | "weaviate" | "faiss"

    # ── Embedding Model ──────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 256

    # ── LCA Data Paths ───────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    BOTTLES_PROCESSES_CSV: str = "data/bottles_processes.csv"
    LCA_FULL_RAG_CSV: str = "data/lca_full_rag_data_units.csv"
    USDA_FLOWS_CSV: str = "data/usda_flows.csv"
    OPENLCA_FLOWS_CSV: str = "data/openlca_flows.csv"

    # ── Scoring Engine Weights ───────────────────────────────────────────────
    # Weighted score = w_crit * criticality + w_tox * toxicity + w_ghg * ghg_normalized
    SCORE_WEIGHT_CRITICALITY: float = 0.30
    SCORE_WEIGHT_TOXICITY: float = 0.35
    SCORE_WEIGHT_GHG: float = 0.35

    # GHG benchmark: 0.71 kg CO2/kWh (global average grid intensity)
    GHG_BENCHMARK_KG_CO2_KWH: float = 0.71

    # Grade thresholds (0–100 normalized score)
    GRADE_A_MAX: float = 30.0
    GRADE_B_MAX: float = 60.0
    GRADE_C_MAX: float = 80.0
    # > 80 → D

    # ── RAG Retrieval ────────────────────────────────────────────────────────
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.60   # lowered from 0.72 to increase hit rate
    RAG_CACHE_TTL_SECONDS: int = 3600
    RAG_FLOWS_SAMPLE: int = 4000             # max product flows to include in corpus

    # ── Redis (optional caching) ─────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = False

    # ── Celery (async jobs for large eBOM uploads) ───────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"

    # ── Chart Output ─────────────────────────────────────────────────────────
    CHART_DPI: int = 150
    CHART_STYLE: str = "seaborn-v0_8-whitegrid"


settings = Settings()