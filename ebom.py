"""
app/schemas/ebom.py
Pydantic v2 schemas for all API contracts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class SustainabilityGrade(str, Enum):
    A = "A"   # Low impact       (score ≤ 30)
    B = "B"   # Medium           (score 31–60)
    C = "C"   # High impact      (score 61–80)
    D = "D"   # Critical impact  (score > 80)


class LifecycleStage(str, Enum):
    RAW_MATERIAL = "Raw Material Extraction"
    MANUFACTURING = "Manufacturing"
    TRANSPORT = "Transport"
    USE_PHASE = "Use Phase"
    END_OF_LIFE = "End of Life"
    PRODUCT_AS_SERVICE = "Product-as-a-Service"


class MatchConfidence(str, Enum):
    HIGH = "high"       # cosine sim ≥ 0.90 + LLM confirmed
    MEDIUM = "medium"   # cosine sim 0.72–0.89 or LLM fallback
    LOW = "low"         # synthetic / estimated


# ── eBOM Input ────────────────────────────────────────────────────────────────

class BOMLineItem(BaseModel):
    item_id: str = Field(..., description="Unique part number / BOM line ID")
    material_name: str = Field(..., description="Raw material name from eBOM")
    quantity_kg: float = Field(..., gt=0, description="Mass in kilograms")
    lifecycle_stage: Optional[LifecycleStage] = None
    supplier_country: Optional[str] = Field(None, description="ISO 3166-1 alpha-2 country code")
    manufacturing_process: Optional[str] = None
    component_description: Optional[str] = None

    @field_validator("material_name")
    @classmethod
    def strip_material(cls, v):
        return v.strip()


class EBOMUploadRequest(BaseModel):
    product_name: str
    product_id: str
    bom_version: Optional[str] = "1.0"
    items: List[BOMLineItem] = Field(..., min_length=1)
    functional_unit: Optional[str] = "1 unit"
    system_boundary: Optional[str] = "cradle-to-gate"


# ── RAG Match ─────────────────────────────────────────────────────────────────

class RAGMatchResult(BaseModel):
    query_material: str
    matched_process: str
    matched_source: str                              # ecoinvent_bottles | lca_rag_full | usda | openlca
    cosine_similarity: float
    confidence: MatchConfidence
    lifecycle_stage: Optional[str] = None
    category_path: Optional[str] = None
    unit: Optional[str] = None
    llm_reasoning: Optional[str] = None
    is_synthetic: bool = False                      # True if values were AI-estimated


# ── LCA Impact Vector ─────────────────────────────────────────────────────────

class LCAImpactVector(BaseModel):
    climate_change_kg_co2eq: float = 0.0
    human_toxicity: float = 0.0
    metal_depletion: float = 0.0
    freshwater_ecotoxicity: float = 0.0
    marine_ecotoxicity: float = 0.0
    freshwater_eutrophication: float = 0.0
    ozone_depletion: float = 0.0
    particulate_matter: float = 0.0
    terrestrial_acidification: float = 0.0
    water_depletion: float = 0.0
    photochemical_oxidant: float = 0.0
    ionising_radiation: float = 0.0


# ── Scored Line Item ──────────────────────────────────────────────────────────

class ScoredBOMItem(BaseModel):
    item_id: str
    material_name: str
    quantity_kg: float
    rag_match: RAGMatchResult
    lca_impacts: LCAImpactVector
    criticality_score: float = Field(..., ge=0, le=100)
    toxicity_score: float = Field(..., ge=0, le=100)
    ghg_score: float = Field(..., ge=0, le=100)
    composite_score: float = Field(..., ge=0, le=100, description="Weighted composite 0–100")
    grade: SustainabilityGrade
    is_hotspot: bool = False
    hotspot_reason: Optional[str] = None
    eco_recommendations: List[str] = []


# ── Product-Level Report ──────────────────────────────────────────────────────

class ProductSustainabilityReport(BaseModel):
    product_id: str
    product_name: str
    bom_version: str
    total_items: int
    total_mass_kg: float
    overall_grade: SustainabilityGrade
    overall_score: float
    total_ghg_kg_co2eq: float
    hotspot_items: List[str]                         # item_ids
    scored_items: List[ScoredBOMItem]
    lifecycle_breakdown: Dict[str, float]           # stage → GHG contribution
    material_breakdown: Dict[str, float]            # material → GHG share %
    charts: Optional[Dict[str, str]] = None         # chart_name → base64 PNG
    validation_mae: Optional[float] = None         # vs Ecoinvent benchmark
    processing_time_ms: Optional[float] = None


# ── What-If Simulation ────────────────────────────────────────────────────────

class MaterialSwap(BaseModel):
    item_id: str
    original_material: str
    proposed_material: str
    proposed_quantity_kg: Optional[float] = None    # if mass changes too


class WhatIfRequest(BaseModel):
    product_id: str
    swaps: List[MaterialSwap]
    narrative: bool = True                          # ask LLM to narrate delta


class WhatIfResult(BaseModel):
    product_id: str
    original_score: float
    projected_score: float
    score_delta: float
    original_ghg: float
    projected_ghg: float
    ghg_reduction_pct: float
    grade_change: Optional[str] = None
    per_swap_analysis: List[Dict[str, Any]] = []
    llm_narrative: Optional[str] = None


# ── Chart Response ────────────────────────────────────────────────────────────

class ChartResponse(BaseModel):
    chart_type: str
    title: str
    base64_png: str
    description: Optional[str] = None


# ── RAG Single-Query ──────────────────────────────────────────────────────────

class MaterialMatchRequest(BaseModel):
    material_name: str
    quantity_kg: Optional[float] = 1.0
    lifecycle_stage: Optional[str] = None
    context: Optional[str] = None
    top_k: int = Field(default=5, le=20)


class MaterialMatchResponse(BaseModel):
    query: str
    matches: List[RAGMatchResult]
    synthetic_impacts: Optional[LCAImpactVector] = None
    llm_explanation: Optional[str] = None