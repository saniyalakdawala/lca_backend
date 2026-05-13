"""
app/services/rag/pipeline.py
RAG (Retrieval-Augmented Generation) Pipeline
──────────────────────────────────────────────
Flow for each material query:
  1. Embed query → dense vector (sentence-transformers/all-MiniLM-L6-v2)
  2. ANN search against Ecoinvent + USDA + OpenLCA index (FAISS / Pinecone)
  3. Return top-K candidates with cosine similarity scores
  4. If top-1 similarity ≥ RAG_SIMILARITY_THRESHOLD → trust vector match
  5. Else → call Groq LLaMA-70b with candidate context for LLM-arbitrated match
  6. If Groq returns no confident match → generate synthetic LCA estimates via Mixtral
  7. Return RAGMatchResult + LCAImpactVector
"""

import json
import logging
import hashlib
import asyncio
import numpy as np
from typing import List, Optional, Tuple, Dict

from config import settings
from startup import get_vector_index, get_embedding_model, get_groq_client, get_lca_store
from ebom import RAGMatchResult, LCAImpactVector, MatchConfidence

logger = logging.getLogger(__name__)

# In-memory result cache (keyed by SHA-256 of material_name.lower())
_match_cache: Dict[str, Tuple[RAGMatchResult, LCAImpactVector]] = {}


class RAGPipeline:
    """
    Singleton service. Injected via FastAPI Depends().
    """

    async def match_material(
        self,
        material_name: str,
        quantity_kg: float = 1.0,
        lifecycle_stage: Optional[str] = None,
        context: Optional[str] = None,
        top_k: int = None,
        lang: str = "en",
    ) -> Tuple[RAGMatchResult, LCAImpactVector]:
        """
        Main entry: returns best RAGMatchResult + LCAImpactVector for a material query.
        """
        top_k = top_k or settings.RAG_TOP_K
        cache_key = hashlib.sha256(f"{material_name.lower()}|{lifecycle_stage}".encode()).hexdigest()

        if cache_key in _match_cache:
            logger.debug(f"Cache hit: {material_name}")
            return _match_cache[cache_key]

        # ── Step 1: Embed ──────────────────────────────────────────────────
        query_vec = await self._embed(material_name)

        # ── Step 2: Vector Search ──────────────────────────────────────────
        candidates = await self._vector_search(query_vec, top_k, lifecycle_stage)

        # ── Step 3: Decision ───────────────────────────────────────────────
        best_candidate = candidates[0] if candidates else None
        above_threshold = (
            best_candidate is not None
            and best_candidate["similarity"] >= settings.RAG_SIMILARITY_THRESHOLD
        )
        has_lca_data = best_candidate is not None and best_candidate.get("has_lca_data", True)

        if above_threshold and has_lca_data:
            match_result = self._build_match(material_name, best_candidate, MatchConfidence.HIGH)
            impacts = self._extract_impacts(best_candidate)
        elif candidates:
            # LLM arbitration
            match_result, impacts = await self._llm_arbitrate(
                material_name, candidates, lifecycle_stage, context, lang=lang
            )
        else:
            # Full synthetic fallback
            match_result, impacts = await self._synthesize_impacts(
                material_name, lifecycle_stage, context
            )

        _match_cache[cache_key] = (match_result, impacts)
        return match_result, impacts

    async def batch_match(
        self,
        materials: List[dict],   # [{"name": str, "qty": float, "stage": str}, ...]
        lang: str = "en",
    ) -> List[Tuple[RAGMatchResult, LCAImpactVector]]:
        """Process a batch of materials concurrently (bounded semaphore to avoid rate limits)."""
        sem = asyncio.Semaphore(8)

        async def safe_match(mat):
            async with sem:
                return await self.match_material(
                    mat["name"], mat.get("qty", 1.0), mat.get("stage"), lang=lang
                )

        return await asyncio.gather(*[safe_match(m) for m in materials])

    # ── Embedding ──────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> np.ndarray:
        model = get_embedding_model()
        if model is None:
            return np.zeros(settings.PINECONE_DIMENSION, dtype=np.float32)
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None, lambda: model.encode([text], normalize_embeddings=True)[0]
        )
        return vec.astype(np.float32)

    # ── Vector Search ──────────────────────────────────────────────────────

    async def _vector_search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        lifecycle_stage: Optional[str],
    ) -> List[dict]:
        index = get_vector_index()
        store = get_lca_store()
        if index is None or store is None:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._search_sync(index, store, query_vec, top_k, lifecycle_stage)
        )

    def _search_sync(self, index, store, query_vec, top_k, lifecycle_stage) -> List[dict]:
        corpus = store["combined_corpus"]
        idx_type = index.get("type") if isinstance(index, dict) else "numpy"

        if idx_type == "faiss":
            import faiss
            D, I = index["index"].search(query_vec.reshape(1, -1), top_k * 3)
            similarities, indices = D[0], I[0]
        elif idx_type == "pinecone":
            results = index["index"].query(
                vector=query_vec.tolist(), top_k=top_k * 3, include_metadata=True
            )
            similarities = [m.score for m in results.matches]
            indices = [int(m.id) for m in results.matches]
        else:
            # Numpy fallback
            sims = (index.embeddings @ query_vec).flatten()
            indices = np.argsort(sims)[::-1][:top_k * 3]
            similarities = sims[indices]

        candidates = []
        for sim, idx in zip(similarities, indices):
            if idx < 0 or idx >= len(corpus):
                continue
            row = corpus.iloc[idx]
            # Optional lifecycle stage filter
            if lifecycle_stage and str(row.get("lifecycle_stage", "")) != lifecycle_stage:
                continue
            cc = row.get("Climate change")
            ht = row.get("Human toxicity")
            md = row.get("Metal depletion")
            # NaN means this row came from flows corpus and has no measured LCA data
            import math
            has_lca = not (
                (cc is None or (isinstance(cc, float) and math.isnan(cc))) and
                (ht is None or (isinstance(ht, float) and math.isnan(ht))) and
                (md is None or (isinstance(md, float) and math.isnan(md)))
            )
            candidates.append({
                "similarity": float(sim),
                "process_name": str(row.get("process_name", "")),
                "source": str(row.get("source", "")),
                "lifecycle_stage": str(row.get("lifecycle_stage", "")),
                "category_path": str(row.get("category_path", "")),
                "unit": str(row.get("unit", "kg")),
                "material_tags": str(row.get("material_tags", "")),
                "climate_change": float(cc or 0) if has_lca else 0.0,
                "human_toxicity": float(ht or 0) if has_lca else 0.0,
                "metal_depletion": float(md or 0) if has_lca else 0.0,
                "has_lca_data": has_lca,
                "corpus_idx": int(idx),
            })
            if len(candidates) >= top_k:
                break

        return candidates

    # ── LLM Arbitration (Groq LLaMA-70b) ──────────────────────────────────

    async def _llm_arbitrate(
        self,
        material_name: str,
        candidates: List[dict],
        lifecycle_stage: Optional[str],
        context: Optional[str],
        lang: str = "en",
    ) -> Tuple[RAGMatchResult, LCAImpactVector]:
        groq = get_groq_client()
        if groq is None:
            # Fallback to best vector match without LLM
            best = candidates[0]
            return self._build_match(material_name, best, MatchConfidence.MEDIUM), self._extract_impacts(best)

        candidate_text = "\n".join([
            f"{i+1}. [{c['similarity']:.3f}] {c['process_name']} "
            f"(source: {c['source']}, stage: {c['lifecycle_stage']}, "
            f"GHG: {c['climate_change']:.4f} kg CO2eq/unit)"
            for i, c in enumerate(candidates[:5])
        ])

        kannada_note = (
            ' Write the "reasoning" value in Kannada (ಕನ್ನಡ) script.'
            if lang == "kn" else ""
        )
        system_prompt = (
            "You are an expert LCA (Life Cycle Assessment) data specialist. "
            "Given a material from an electronic BOM and LCA candidate processes, select the BEST match. "
            f"Respond in JSON only with keys: best_index (1-based int), reasoning (str), confidence (high/medium/low).{kannada_note}"
        )

        user_prompt = f"""Material to match: "{material_name}"
Lifecycle context: {lifecycle_stage or 'unknown'}
Additional context: {context or 'none'}

Candidate LCA processes:
{candidate_text}

Which candidate best represents this material's environmental impacts?"""

        try:
            response = await groq.chat.completions.create(
                model=settings.GROQ_MODEL_MATCHING,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            best_idx = int(parsed.get("best_index", 1)) - 1
            reasoning = parsed.get("reasoning", "")
            confidence_str = parsed.get("confidence", "medium")
            confidence = MatchConfidence(confidence_str) if confidence_str in MatchConfidence.__members__.values() else MatchConfidence.MEDIUM

            best = candidates[min(best_idx, len(candidates) - 1)]
            best["llm_reasoning"] = reasoning
            return self._build_match(material_name, best, confidence), self._extract_impacts(best)

        except Exception as e:
            logger.warning(f"Groq LLM arbitration failed for '{material_name}': {e}")
            best = candidates[0]
            return self._build_match(material_name, best, MatchConfidence.MEDIUM), self._extract_impacts(best)

    # ── Synthetic Impact Generation (Groq Mixtral) ─────────────────────────

    async def _synthesize_impacts(
        self,
        material_name: str,
        lifecycle_stage: Optional[str],
        context: Optional[str],
    ) -> Tuple[RAGMatchResult, LCAImpactVector]:
        """
        When no vector match passes threshold: use Mixtral to estimate LCA values
        based on material chemistry and known patterns. Flagged as is_synthetic=True.
        """
        groq = get_groq_client()
        if groq is None:
            return self._zero_match(material_name), LCAImpactVector()

        system_prompt = """You are an expert LCA engineer. Generate estimated environmental impact
values for a material. Respond ONLY in JSON with exact keys:
{
  "climate_change_kg_co2eq": float,
  "human_toxicity": float,
  "metal_depletion": float,
  "freshwater_ecotoxicity": float,
  "marine_ecotoxicity": float,
  "freshwater_eutrophication": float,
  "ozone_depletion": float,
  "particulate_matter": float,
  "terrestrial_acidification": float,
  "water_depletion": float,
  "matched_process_name": str,
  "reasoning": str
}
Base values per 1 kg of material. Use Ecoinvent-compatible units."""

        user_prompt = f"""Estimate LCA impacts for: "{material_name}"
Lifecycle stage: {lifecycle_stage or 'manufacturing'}
Context: {context or 'electronic component / product'}
Use published literature or Ecoinvent 3.x equivalents as basis."""

        try:
            response = await groq.chat.completions.create(
                model=settings.GROQ_MODEL_SYNTHESIS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            impacts = LCAImpactVector(
                climate_change_kg_co2eq=float(data.get("climate_change_kg_co2eq", 0)),
                human_toxicity=float(data.get("human_toxicity", 0)),
                metal_depletion=float(data.get("metal_depletion", 0)),
                freshwater_ecotoxicity=float(data.get("freshwater_ecotoxicity", 0)),
                marine_ecotoxicity=float(data.get("marine_ecotoxicity", 0)),
                freshwater_eutrophication=float(data.get("freshwater_eutrophication", 0)),
                ozone_depletion=float(data.get("ozone_depletion", 0)),
                particulate_matter=float(data.get("particulate_matter", 0)),
                terrestrial_acidification=float(data.get("terrestrial_acidification", 0)),
                water_depletion=float(data.get("water_depletion", 0)),
            )
            match = RAGMatchResult(
                query_material=material_name,
                matched_process=data.get("matched_process_name", f"Estimated: {material_name}"),
                matched_source="groq_synthetic",
                cosine_similarity=0.0,
                confidence=MatchConfidence.LOW,
                lifecycle_stage=lifecycle_stage,
                llm_reasoning=data.get("reasoning"),
                is_synthetic=True,
            )
            return match, impacts

        except Exception as e:
            logger.error(f"Synthetic generation failed for '{material_name}': {e}")
            return self._zero_match(material_name), LCAImpactVector()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_match(self, query: str, candidate: dict, confidence: MatchConfidence) -> RAGMatchResult:
        return RAGMatchResult(
            query_material=query,
            matched_process=candidate["process_name"],
            matched_source=candidate["source"],
            cosine_similarity=round(candidate["similarity"], 4),
            confidence=confidence,
            lifecycle_stage=candidate.get("lifecycle_stage"),
            category_path=candidate.get("category_path"),
            unit=candidate.get("unit"),
            llm_reasoning=candidate.get("llm_reasoning"),
            is_synthetic=False,
        )

    def _extract_impacts(self, candidate: dict) -> LCAImpactVector:
        return LCAImpactVector(
            climate_change_kg_co2eq=candidate.get("climate_change", 0.0),
            human_toxicity=candidate.get("human_toxicity", 0.0),
            metal_depletion=candidate.get("metal_depletion", 0.0),
        )

    def _zero_match(self, material: str) -> RAGMatchResult:
        return RAGMatchResult(
            query_material=material,
            matched_process="No match found",
            matched_source="none",
            cosine_similarity=0.0,
            confidence=MatchConfidence.LOW,
            is_synthetic=True,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
_rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline