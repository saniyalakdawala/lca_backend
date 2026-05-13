"""
app/core/startup.py
Boot-time initialization:
  1. Load all LCA CSV datasets into memory
  2. Build / connect to vector index (FAISS in-process for dev, Pinecone for prod)
  3. Warm up embedding model
  4. Warm up Groq client
  5. Pre-build RAG retriever
"""

import logging
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (injected into services at startup) ───────────────
_lca_store: dict = {}
_vector_index = None
_embedding_model = None
_groq_client = None


async def initialize_services():
    """Called once at FastAPI lifespan startup."""
    await asyncio.gather(
        _load_lca_datasets(),
        _init_embedding_model(),
        _init_groq_client(),
    )
    await _build_vector_index()
    logger.info("All services initialized.")


# ── LCA Dataset Loader ────────────────────────────────────────────────────────

async def _load_lca_datasets():
    global _lca_store
    loop = asyncio.get_event_loop()
    _lca_store = await loop.run_in_executor(None, _load_sync)
    logger.info(
        f"LCA datasets loaded: "
        f"bottles={len(_lca_store['bottles'])} | "
        f"rag_full={len(_lca_store['rag_full'])} | "
        f"usda={len(_lca_store['usda'])} | "
        f"openlca={len(_lca_store['openlca'])}"
    )


def _load_sync() -> dict:
    """Synchronous CSV loading with dtype optimization."""
    store = {}

    store["bottles"] = pd.read_csv(settings.BOTTLES_PROCESSES_CSV, low_memory=False)
    store["rag_full"] = pd.read_csv(settings.LCA_FULL_RAG_CSV, low_memory=False)
    store["usda"] = pd.read_csv(settings.USDA_FLOWS_CSV, low_memory=False)
    store["openlca"] = pd.read_csv(settings.OPENLCA_FLOWS_CSV, low_memory=False)

    # Build a unified lookup corpus: process_name + semantic_text + flow_name
    bottles_corpus = store["bottles"][["process_name", "semantic_text", "material_tags",
                                       "lifecycle_stage", "category_path", "unit",
                                       "Climate change", "Human toxicity", "Metal depletion"]].copy()
    bottles_corpus["source"] = "ecoinvent_bottles"

    rag_corpus = store["rag_full"][["process_name", "lifecycle_stage", "category_path",
                                    "Global warming 100a (incl. NMVOC av.)",
                                    "Human toxicity 100a", "Abiotic depletion (elem., econ. reserve)"]].copy()
    rag_corpus = rag_corpus.rename(columns={
        "Global warming 100a (incl. NMVOC av.)": "Climate change",
        "Human toxicity 100a": "Human toxicity",
        "Abiotic depletion (elem., econ. reserve)": "Metal depletion",
    })
    rag_corpus["semantic_text"] = rag_corpus["process_name"]
    rag_corpus["material_tags"] = "unknown"
    rag_corpus["unit"] = "kg"
    rag_corpus["source"] = "lca_rag_full"

    # Enrich semantic text with category and tags for better embedding quality
    bottles_corpus["semantic_text"] = (
        bottles_corpus["process_name"].fillna("") + " " +
        bottles_corpus["material_tags"].fillna("") + " " +
        bottles_corpus["category_path"].fillna("")
    ).str.strip()
    rag_corpus["semantic_text"] = (
        rag_corpus["process_name"].fillna("") + " " +
        rag_corpus["lifecycle_stage"].fillna("") + " " +
        rag_corpus["category_path"].fillna("")
    ).str.strip()

    store["combined_corpus"] = pd.concat([bottles_corpus, rag_corpus], ignore_index=True)
    store["flows_corpus"] = pd.concat([
        store["usda"][["semantic_text", "flow_name", "material_tags", "unit", "category_path", "flow_type"]].assign(source="usda"),
        store["openlca"][["semantic_text", "flow_name", "material_tags", "unit", "category_path", "flow_type"]].assign(source="openlca"),
    ], ignore_index=True)

    return store


# ── Embedding Model ───────────────────────────────────────────────────────────

async def _init_embedding_model():
    global _embedding_model
    loop = asyncio.get_event_loop()
    _embedding_model = await loop.run_in_executor(None, _load_embedding_model)
    logger.info(f"Embedding model ready: {settings.EMBEDDING_MODEL}")


def _load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(settings.EMBEDDING_MODEL)
    except ImportError:
        logger.warning("sentence-transformers not installed; embedding unavailable.")
        return None


# ── Groq Client ───────────────────────────────────────────────────────────────

async def _init_groq_client():
    global _groq_client
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — LLM features disabled.")
        return
    try:
        from groq import AsyncGroq
        _groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        logger.info(f"Groq client ready. Models: {settings.GROQ_MODEL_MATCHING} / {settings.GROQ_MODEL_SYNTHESIS}")
    except ImportError:
        logger.warning("groq package not installed.")


# ── Vector Index ──────────────────────────────────────────────────────────────

async def _build_vector_index():
    global _vector_index
    if _embedding_model is None:
        logger.warning("Embedding model not available; skipping vector index.")
        return

    backend = settings.VECTOR_BACKEND
    if backend == "pinecone" and settings.PINECONE_API_KEY:
        _vector_index = await _build_pinecone_index()
    else:
        _vector_index = await _build_faiss_index()

    logger.info(f"Vector index ready [{backend}]")


async def _build_faiss_index():
    """In-process FAISS index — ideal for development / offline use."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _build_faiss_sync)


def _build_faiss_sync():
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed; falling back to cosine numpy search.")
        return _NumpyIndex(_lca_store)

    corpus = _build_expanded_corpus()
    # Persist expanded corpus so _search_sync can look up rows by FAISS index
    _lca_store["combined_corpus"] = corpus

    texts = corpus["semantic_text"].fillna("").tolist()
    logger.info(f"Encoding {len(texts)} LCA corpus entries (expanded)...")
    embeddings = _embedding_model.encode(
        texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return {"type": "faiss", "index": index, "corpus": corpus, "embeddings": embeddings}


def _build_expanded_corpus() -> "pd.DataFrame":
    """Extend combined_corpus with a sample of product/material flows for richer vocabulary."""
    base = _lca_store["combined_corpus"].copy()

    flows = _lca_store["flows_corpus"].copy()
    # Keep only product/material flows; exclude elementary emission flows
    product_mask = flows["flow_type"].str.lower().str.contains(
        r"product|material|waste", na=False, regex=True
    )
    flows = flows[product_mask].copy()

    # Deduplicate by flow_name and sample
    flows = flows.drop_duplicates(subset=["flow_name"])
    if len(flows) > settings.RAG_FLOWS_SAMPLE:
        flows = flows.sample(settings.RAG_FLOWS_SAMPLE, random_state=42)

    flows_exp = pd.DataFrame({
        "process_name": flows["flow_name"].values,
        "semantic_text": (
            flows["flow_name"].fillna("") + " " +
            flows["material_tags"].fillna("") + " " +
            flows["category_path"].fillna("")
        ).str.strip().values,
        "material_tags":   flows["material_tags"].values,
        "lifecycle_stage": "unknown",
        "category_path":   flows["category_path"].values,
        "unit":            flows["unit"].values,
        "Climate change":  np.nan,
        "Human toxicity":  np.nan,
        "Metal depletion": np.nan,
        "source":          flows["source"].values,
    })

    expanded = pd.concat([base, flows_exp], ignore_index=True)
    logger.info(
        f"Expanded corpus: {len(base)} base + {len(flows_exp)} flows = {len(expanded)} total"
    )
    return expanded


async def _build_pinecone_index():
    """Connect to existing Pinecone index; upsert if empty."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        idx = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = idx.describe_index_stats()
        logger.info(f"Pinecone index stats: {stats}")
        return {"type": "pinecone", "index": idx, "corpus": _lca_store["combined_corpus"]}
    except Exception as e:
        logger.error(f"Pinecone init failed: {e}. Falling back to FAISS.")
        return await _build_faiss_index()


class _NumpyIndex:
    """Zero-dependency cosine similarity fallback."""
    def __init__(self, store):
        corpus = _build_expanded_corpus()
        store["combined_corpus"] = corpus
        texts = corpus["semantic_text"].fillna("").tolist()
        self.corpus = corpus
        self.embeddings = _embedding_model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def search(self, query_vec: np.ndarray, k: int):
        sims = (self.embeddings @ query_vec).flatten()
        top_k = np.argsort(sims)[::-1][:k]
        return sims[top_k], top_k


# ── Public Accessors ──────────────────────────────────────────────────────────

def get_lca_store() -> dict:
    return _lca_store

def get_vector_index():
    return _vector_index

def get_embedding_model():
    return _embedding_model

def get_groq_client():
    return _groq_client