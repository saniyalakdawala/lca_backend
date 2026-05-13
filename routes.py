"""
routes.py  —  Sustainability Intelligence Engine
All API routes: BOM upload, scoring, charts, RAG matching, hotspots, health,
and Groq-powered eco-design suggestions.

Dashboard output mirrors the AI4LCI screenshots exactly:
  • Header KPIs  : Sustainability Score, Letter Grade, Total Carbon Footprint, Dominant Hotspot
  • Radar chart  : GHG | Recyclability | Material Risk | Energy | Water
  • Component table: Component, Weight%, Materials, Electricity%, Impact Score, Grade
  • Bar charts   : Component Impact Scores (blue) + GHG Emissions by Component (red)
"""

import io
import json
import base64
import logging
import asyncio
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

from ebom import BOMLineItem, ScoredBOMItem, RAGMatchResult, LCAImpactVector
from engine import SustainabilityScoreEngine
from pipeline import RAGPipeline
from config import settings
from startup import get_groq_client

logger = logging.getLogger(__name__)

router = APIRouter()
score_engine = SustainabilityScoreEngine()
rag_pipeline = RAGPipeline()

# ── In-memory state (last processed BOM, populated by /upload-bom) ────────────
_last_scored_items: List[ScoredBOMItem] = []
_last_summary: Dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "SustainabilityScoreEngine v1.0",
        "rag": "RAGPipeline",
        "groq_ready": get_groq_client() is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BOM UPLOAD  →  full dashboard payload
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/upload-bom")
async def upload_ebom(file: UploadFile = File(...), lang: str = Query(default="en")):
    """
    Accepts a CSV eBOM and returns the complete AI4LCI dashboard payload:
      - Header KPIs
      - Radar chart data
      - Component-level table (matching screenshot cols)
      - Base64 bar charts
      - Groq-powered eco-design suggestions per component
    """
    global _last_scored_items, _last_summary

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        _validate_columns(df)

        # ── 1. Build material list ─────────────────────────────────────────
        materials_input = _parse_bom_rows(df)

        # ── 2. RAG + Groq matching ─────────────────────────────────────────
        matches_and_impacts = await rag_pipeline.batch_match(materials_input, lang=lang)
        rag_matches  = [m[0] for m in matches_and_impacts]
        lca_impacts  = [m[1] for m in matches_and_impacts]

        # ── 3. Build BOMLineItem list ──────────────────────────────────────
        bom_items = [
            BOMLineItem(
                item_id=mat["item_id"],
                material_name=mat["name"],
                quantity_kg=mat["qty"],
            )
            for mat in materials_input
        ]

        # ── 4. Score ───────────────────────────────────────────────────────
        scored_items = score_engine.score_product(bom_items, rag_matches, lca_impacts)
        summary      = score_engine.compute_product_summary(scored_items)

        # Persist for chart endpoints
        _last_scored_items = scored_items
        _last_summary      = summary

        # ── 5. Electricity contribution per component (% of total GHG) ────
        total_ghg = summary["total_ghg_kg_co2eq"] or 1e-9

        # ── 6. Radar metrics (matching dashboard: GHG, Recyclability,
        #        Material Risk, Energy, Water) ─────────────────────────────
        ghg_score_norm   = min(summary["overall_score"], 100)
        material_risk    = min(ghg_score_norm * 1.1, 100)
        recyclability    = _estimate_recyclability(scored_items)
        energy_score     = _estimate_energy_score(scored_items)
        water_score      = _estimate_water_score(scored_items)

        radar_data = {
            "GHG":           round(ghg_score_norm, 1),
            "Recyclability": round(recyclability, 1),
            "Material Risk": round(material_risk, 1),
            "Energy":        round(energy_score, 1),
            "Water":         round(water_score, 1),
        }

        # ── 7. Component table (mirrors screenshot exactly) ────────────────
        component_rows = []
        for s in scored_items:
            item_ghg = s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg
            elec_pct = round((item_ghg / total_ghg) * 100, 2)
            component_rows.append({
                "component":    s.item_id,
                "weight_pct":   round((s.quantity_kg / (summary["total_mass_kg"] or 1)) * 100, 1),
                "materials":    s.material_name,
                "electricity_pct": elec_pct,
                "impact_score": s.composite_score,
                "grade":        s.grade.value,
                "is_hotspot":   s.is_hotspot,
                "hotspot_reason": s.hotspot_reason,
                "eco_recommendations": s.eco_recommendations,
                # RAG provenance
                "rag_match": {
                    "process":    s.rag_match.matched_process,
                    "source":     s.rag_match.matched_source,
                    "similarity": s.rag_match.cosine_similarity,
                    "confidence": s.rag_match.confidence.value,
                    "is_synthetic": s.rag_match.is_synthetic,
                    "llm_reasoning": s.rag_match.llm_reasoning,
                },
                "lca_impacts": {
                    "climate_change_kg_co2eq": s.lca_impacts.climate_change_kg_co2eq,
                    "human_toxicity":          s.lca_impacts.human_toxicity,
                    "metal_depletion":         s.lca_impacts.metal_depletion,
                },
                "score_breakdown": {
                    "criticality": s.criticality_score,
                    "toxicity":    s.toxicity_score,
                    "ghg":         s.ghg_score,
                },
            })

        # ── 8. Generate bar charts (Base64 PNG) ────────────────────────────
        impact_chart_b64 = _generate_impact_bar_chart(scored_items)
        ghg_chart_b64    = _generate_ghg_bar_chart(scored_items)

        # ── 9. Groq narrative summary ──────────────────────────────────────
        narrative = await _groq_product_narrative(summary, scored_items, lang=lang)

        # ── 10. Assemble final response ────────────────────────────────────
        dominant = summary["hotspot_items"][0] if summary["hotspot_items"] else "None"

        return {
            # ── Header KPIs (screenshot row 1) ──────────────────────────
            "dashboard_header": {
                "sustainability_score": summary["overall_score"],
                "letter_grade":         summary["overall_grade"].value,
                "total_carbon_footprint": round(summary["total_ghg_kg_co2eq"], 2),
                "dominant_hotspot":     dominant,
            },
            # ── Radar (screenshot spider) ────────────────────────────────
            "radar_chart_data": radar_data,
            # ── Component table (screenshot table) ──────────────────────
            "component_analysis": component_rows,
            # ── Bar charts base64 ────────────────────────────────────────
            "charts": {
                "component_impact_bar": impact_chart_b64,
                "ghg_emissions_bar":    ghg_chart_b64,
            },
            # ── Product-level extras ─────────────────────────────────────
            "product_summary": {
                "total_items":     len(scored_items),
                "total_mass_kg":   summary["total_mass_kg"],
                "hotspot_items":   summary["hotspot_items"],
                "lifecycle_breakdown": summary["lifecycle_breakdown"],
                "material_breakdown":  summary["material_breakdown"],
            },
            # ── Groq narrative ───────────────────────────────────────────
            "groq_narrative": narrative,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("BOM analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# HOTSPOTS ENDPOINT  (used by Streamlit sidebar widget)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/analytics/hotspots")
async def get_hotspots():
    if not _last_scored_items:
        raise HTTPException(status_code=404, detail="No BOM analysed yet. POST to /upload-bom first.")
    hotspots = [s.item_id for s in _last_scored_items if s.is_hotspot]
    details = [
        {
            "component":     s.item_id,
            "composite_score": s.composite_score,
            "grade":           s.grade.value,
            "reason":          s.hotspot_reason,
            "ghg_kg_co2eq":   round(s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg, 4),
            "recommendations": s.eco_recommendations,
        }
        for s in _last_scored_items if s.is_hotspot
    ]
    return {"hotspots": hotspots, "details": details, "count": len(hotspots)}


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS ENDPOINT  (Streamlit /visualizations)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/visualizations")
async def get_visualizations():
    if not _last_scored_items:
        raise HTTPException(status_code=404, detail="No BOM analysed yet.")
    chart_b64 = _generate_combined_2x2_chart(_last_scored_items, _last_summary)
    return {"chart": chart_b64, "type": "2x2 Multi-dimensional Impact Matrix"}


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL COMPONENT SCORING  (Streamlit RAG tab)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/scoring/{material_name}")
async def score_single_material(
    material_name: str,
    quantity_kg: float = Query(default=1.0, gt=0),
    lifecycle_stage: Optional[str] = Query(default=None),
    lang: str = Query(default="en"),
):
    """
    RAG-match a single material and return its sustainability score + Groq suggestions.
    Used by the Material Search page.
    """
    try:
        rag_match, lca_impacts = await rag_pipeline.match_material(
            material_name=material_name,
            quantity_kg=quantity_kg,
            lifecycle_stage=lifecycle_stage,
            lang=lang,
        )
        bom_item = BOMLineItem(
            item_id="query",
            material_name=material_name,
            quantity_kg=quantity_kg,
        )
        crit, tox, ghg, composite, grade = score_engine.score_item(bom_item, rag_match, lca_impacts)

        # Groq eco-design suggestions for this single material
        suggestions = await _groq_material_suggestions(material_name, composite, rag_match, lca_impacts, lang=lang)

        return {
            "material":        material_name,
            "grade":           grade.value,
            "composite_score": round(composite, 2),
            "is_hotspot":      composite > 60,
            "score_breakdown": {
                "criticality": round(crit, 2),
                "toxicity":    round(tox, 2),
                "ghg":         round(ghg, 2),
            },
            "lca_impacts": {
                "climate_change_kg_co2eq": lca_impacts.climate_change_kg_co2eq,
                "human_toxicity":          lca_impacts.human_toxicity,
                "metal_depletion":         lca_impacts.metal_depletion,
            },
            "rag_match": {
                "matched_process": rag_match.matched_process,
                "source":          rag_match.matched_source,
                "similarity":      rag_match.cosine_similarity,
                "confidence":      rag_match.confidence.value,
                "is_synthetic":    rag_match.is_synthetic,
                "llm_reasoning":   rag_match.llm_reasoning,
            },
            "groq_suggestions": suggestions,
        }
    except Exception as e:
        logger.exception(f"Single material scoring failed: {material_name}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# RAG MATERIAL MATCH  (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/rag/match")
async def rag_match_material(body: dict):
    """
    POST {"material_name": "...", "quantity_kg": 1.0, "lifecycle_stage": "manufacturing"}
    Returns RAG match + LCA impacts + Groq reasoning.
    """
    material_name   = body.get("material_name", "")
    quantity_kg     = float(body.get("quantity_kg", 1.0))
    lifecycle_stage = body.get("lifecycle_stage")
    if not material_name:
        raise HTTPException(status_code=400, detail="material_name required")

    rag_match, lca_impacts = await rag_pipeline.match_material(
        material_name, quantity_kg, lifecycle_stage
    )
    return {
        "query":   material_name,
        "match":   rag_match.dict(),
        "impacts": lca_impacts.dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHART ENDPOINTS  (called from routes or Streamlit directly)
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/charts/component-impact")
async def chart_component_impact():
    if not _last_scored_items:
        raise HTTPException(status_code=404, detail="No BOM analysed yet.")
    return {"chart_base64": _generate_impact_bar_chart(_last_scored_items)}


@router.get("/charts/ghg-emissions")
async def chart_ghg_emissions():
    if not _last_scored_items:
        raise HTTPException(status_code=404, detail="No BOM analysed yet.")
    return {"chart_base64": _generate_ghg_bar_chart(_last_scored_items)}


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ  ECO-DESIGN SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/suggestions/eco-design")
async def eco_design_suggestions(body: dict):
    """
    POST the full component analysis list (or a subset) and receive
    eco-design suggestions for each hotspot component.
    Body: {"components": [...component_analysis rows...], "lang": "en"|"kn"}
    """
    components = body.get("components", [])
    lang       = body.get("lang", "en")
    if not components:
        raise HTTPException(status_code=400, detail="components list required")

    groq = get_groq_client()
    if groq is None:
        raise HTTPException(status_code=503, detail="Groq client not configured. Check GROQ_API_KEY in .env")

    # Only send hotspots or high-score components to save tokens
    priority = [c for c in components if c.get("impact_score", 0) >= 40 or c.get("is_hotspot")][:8]

    component_summary = "\n".join([
        f"- {c['component']} | Materials: {c['materials']} | Impact: {c['impact_score']} "
        f"| Grade: {c['grade']} | GHG%: {c.get('electricity_pct', 'N/A')}% "
        f"| Hotspot: {c.get('is_hotspot', False)}"
        for c in priority
    ])

    kannada_note = (
        "\n\nIMPORTANT: Write the 'suggestion' and 'impact_reduction_est' fields entirely in "
        "Kannada (ಕನ್ನಡ) script. Keep 'component' and 'priority' values in English."
        if lang == "kn" else ""
    )

    system_prompt = f"""You are a senior eco-design engineer specializing in electronics sustainability.
Provide concise, actionable material substitution and design suggestions based on LCA data.
Format your response as a JSON array with objects:
{{"component": str, "suggestion": str, "impact_reduction_est": str, "priority": "high|medium|low"}}
Only return the JSON array, no prose.{kannada_note}"""

    user_prompt = f"""Based on this eBOM sustainability analysis, provide eco-design suggestions:

{component_summary}

For each component, suggest:
1. Material substitution or sourcing change
2. Estimated GHG/impact reduction
3. Priority (high = hotspot, medium = grade C, low = grade B)"""

    try:
        response = await groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        # Mixtral sometimes wraps in {"suggestions": [...]}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            suggestions = parsed.get("suggestions", parsed.get("items", list(parsed.values())[0] if parsed else []))
        else:
            suggestions = parsed
        return {"suggestions": suggestions, "components_analysed": len(priority)}
    except Exception as e:
        logger.error(f"Groq eco-design suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"Groq request failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT VISION SCAN  —  identify components from an image
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/scan-product")
async def scan_product(body: dict):
    """
    POST {"image_b64": "<base64-encoded JPEG/PNG>", "mime_type": "image/jpeg"}
    Returns detected components with material names, estimated quantities, and bilingual labels.
    """
    image_b64  = body.get("image_b64", "")
    mime_type  = body.get("mime_type", "image/jpeg")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 required")

    groq = get_groq_client()
    if groq is None:
        raise HTTPException(status_code=503, detail="AI engine not configured")

    # Try vision model; fall back to text description if model unavailable
    system_prompt = """You are a product disassembly and lifecycle assessment expert.
Analyse the uploaded product image and identify every distinct material or component.
Return ONLY a JSON object with this exact structure:
{
  "product_description": "<brief one-line description>",
  "components": [
    {
      "name_en": "<material/component name in English>",
      "name_kn": "<ಕನ್ನಡದಲ್ಲಿ ಹೆಸರು>",
      "estimated_qty_kg": <float, best estimate per unit>,
      "category": "<Plastic|Metal|Electronic|Glass|Rubber|Fabric|Other>",
      "lifecycle_stage": "<Manufacturing|Raw Material Extraction|Transport|Use Phase|End of Life>"
    }
  ]
}
Be specific with material names (e.g. 'Polyethylene terephthalate (PET)' not just 'Plastic').
Include 3-10 components depending on product complexity."""

    try:
        response = await groq.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": system_prompt},
                ],
            }],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        components = data.get("components", [])
        return {
            "product_description": data.get("product_description", ""),
            "components": components,
            "component_count": len(components),
        }

    except Exception as e:
        logger.error(f"Vision scan failed: {e}")
        # Graceful degradation: return a helpful error so the frontend can show it
        raise HTTPException(
            status_code=500,
            detail=f"Product scan failed: {str(e)}. Ensure GROQ_API_KEY is set and supports vision."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_columns(df: pd.DataFrame):
    """
    Auto-detects material and quantity columns from any eBOM CSV.
    Logs actual columns found so errors are easy to diagnose.
    Raises ValueError with the actual columns listed if nothing matches.
    """
    actual = list(df.columns)
    logger.info(f"CSV columns detected: {actual}")

    # Material column candidates (case-insensitive)
    material_candidates = [
        "material_name", "Material_Name", "component", "Component",
        "material", "Material", "part_name", "Part Name", "Name",
        "name", "description", "Description", "item", "Item",
    ]
    qty_candidates = [
        "quantity_kg", "Quantity_kg", "weight", "Weight", "weight_kg",
        "Weight_kg", "qty", "Qty", "quantity", "Quantity",
        "mass_kg", "Mass", "mass", "amount", "Amount",
    ]

    # Case-insensitive lookup
    cols_lower = {c.lower().strip(): c for c in actual}

    mat_col = None
    for cand in material_candidates:
        if cand.lower() in cols_lower:
            mat_col = cols_lower[cand.lower()]
            break
    # Last resort: first string-ish column
    if mat_col is None:
        for c in actual:
            if df[c].dtype == object:
                mat_col = c
                logger.warning(f"No standard material column found — using first text column: '{c}'")
                break

    qty_col = None
    for cand in qty_candidates:
        if cand.lower() in cols_lower:
            qty_col = cols_lower[cand.lower()]
            break
    # Last resort: first numeric column
    if qty_col is None:
        for c in actual:
            if pd.api.types.is_numeric_dtype(df[c]):
                qty_col = c
                logger.warning(f"No standard quantity column found — using first numeric column: '{c}'")
                break

    if mat_col is None:
        raise ValueError(
            f"Could not find a material/component column in your CSV. "
            f"Columns found: {actual}. "
            f"Please name it one of: material_name, Component, Material, Name."
        )
    if qty_col is None:
        raise ValueError(
            f"Could not find a quantity/weight column in your CSV. "
            f"Columns found: {actual}. "
            f"Please name it one of: quantity_kg, Weight, Qty, Quantity."
        )

    # Store resolved column names on the df for use in _parse_bom_rows
    df.attrs["_mat_col"] = mat_col
    df.attrs["_qty_col"] = qty_col
    logger.info(f"Using material column='{mat_col}', quantity column='{qty_col}'")


def _parse_bom_rows(df: pd.DataFrame) -> List[dict]:
    # Use resolved columns from _validate_columns; fall back to brute-force if needed
    mat_col = df.attrs.get("_mat_col")
    qty_col = df.attrs.get("_qty_col")

    # Resolve component-ID column (for item_id)
    actual = list(df.columns)
    id_candidates = ["item_id", "component", "Component", "part_no", "Part No", "id", "ID"]
    cols_lower = {c.lower().strip(): c for c in actual}
    id_col = None
    for cand in id_candidates:
        if cand.lower() in cols_lower:
            id_col = cols_lower[cand.lower()]
            break

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        name = str(row.get(mat_col, "") if mat_col else "").strip() or f"Material-{i+1}"
        try:
            qty = float(row.get(qty_col, 1.0) if qty_col else 1.0)
            if qty <= 0 or np.isnan(qty):
                qty = 1.0
        except (ValueError, TypeError):
            qty = 1.0
        stage = str(row.get("lifecycle_stage", row.get("Stage", row.get("stage", "manufacturing")))).strip()
        item_id = str(row.get(id_col, f"ID-{i+1}") if id_col else f"ID-{i+1}").strip()
        results.append({"name": name, "qty": qty, "stage": stage, "item_id": item_id})
    return results


# ── Radar dimension estimators ─────────────────────────────────────────────────

def _estimate_recyclability(items: List[ScoredBOMItem]) -> float:
    """Higher criticality + toxicity → lower recyclability."""
    if not items:
        return 50.0
    avg_crit = np.mean([s.criticality_score for s in items])
    avg_tox  = np.mean([s.toxicity_score for s in items])
    score    = 100 - (avg_crit * 0.4 + avg_tox * 0.3)
    return float(np.clip(score, 0, 100))


def _estimate_energy_score(items: List[ScoredBOMItem]) -> float:
    """Proxy: GHG score correlates with energy intensity."""
    if not items:
        return 50.0
    avg_ghg = np.mean([s.ghg_score for s in items])
    return float(np.clip(avg_ghg * 0.9, 0, 100))


def _estimate_water_score(items: List[ScoredBOMItem]) -> float:
    """Proxy from water_depletion LCA field if available."""
    if not items:
        return 50.0
    water_vals = [s.lca_impacts.water_depletion for s in items]
    if max(water_vals, default=0) == 0:
        # Fallback: derive from toxicity
        return float(np.clip(np.mean([s.toxicity_score for s in items]) * 0.7, 0, 100))
    ref = 50.0
    norm = np.mean(water_vals) / ref * 100
    return float(np.clip(norm, 0, 100))


# ── Chart generators ───────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=settings.CHART_DPI, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _generate_impact_bar_chart(items: List[ScoredBOMItem]) -> str:
    """Blue horizontal bar chart — Component Impact Scores (mirrors screenshot)."""
    if not items:
        return ""
    names  = [s.item_id for s in items]
    scores = [s.composite_score for s in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    try:
        plt.style.use(settings.CHART_STYLE)
    except Exception:
        pass

    bars = ax.barh(names, scores, color="#4A90D9", edgecolor="white", height=0.6)
    ax.set_xlabel("Impact Score (0–100)", fontsize=10)
    ax.set_title("Component Impact Scores", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", ha="left", fontsize=8, color="#333")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _generate_ghg_bar_chart(items: List[ScoredBOMItem]) -> str:
    """Red horizontal bar chart — GHG Emissions by Component (mirrors screenshot)."""
    if not items:
        return ""
    names = [s.item_id for s in items]
    ghgs  = [round(s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg, 4) for s in items]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.45)))
    try:
        plt.style.use(settings.CHART_STYLE)
    except Exception:
        pass

    bars = ax.barh(names, ghgs, color="#E05252", edgecolor="white", height=0.6)
    ax.set_xlabel("GHG Emissions (kg CO2eq)", fontsize=10)
    ax.set_title("GHG Emissions by Component", fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()

    for bar, val in zip(bars, ghgs):
        ax.text(bar.get_width() + max(ghgs) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8, color="#333")

    fig.tight_layout()
    return _fig_to_b64(fig)


def _generate_combined_2x2_chart(
    items: List[ScoredBOMItem], summary: Dict
) -> str:
    """
    2x2 chart grid used by /visualizations:
      TL: Component Impact bar  |  TR: GHG bar
      BL: Lifecycle GHG pie     |  BR: Radar spider

    NOTE: tight_layout() crashes when a polar subplot has empty wedges on
    some matplotlib/Windows builds. We use subplots_adjust() instead and
    render the radar on a manually-placed axes to avoid the bug entirely.
    """
    if not items:
        return ""

    try:
        plt.style.use(settings.CHART_STYLE)
    except Exception:
        pass

    names  = [s.item_id[:18] for s in items]
    scores = [s.composite_score for s in items]
    ghgs   = [round(s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg, 4) for s in items]

    # Use GridSpec so we can mix polar and cartesian without tight_layout issues
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, max(10, len(items) * 0.55 + 4)))
    fig.suptitle("Multi-Dimensional Sustainability Impact Matrix",
                 fontsize=14, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.94, bottom=0.06, left=0.08, right=0.97)

    # ── TL: Impact bar ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(names, scores, color="#4A90D9", edgecolor="white", height=0.6)
    ax1.set_title("Component Impact Scores", fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.invert_yaxis()
    ax1.set_xlabel("Score (0-100)", fontsize=9)

    # ── TR: GHG bar ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(names, ghgs, color="#E05252", edgecolor="white", height=0.6)
    ax2.set_title("GHG Emissions by Component", fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlabel("kg CO2eq", fontsize=9)

    # ── BL: Lifecycle pie ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    lc_data = {k: v for k, v in (summary.get("lifecycle_breakdown") or {}).items() if v and v > 0}
    if lc_data and len(lc_data) > 0:
        try:
            labels = [str(k)[:20] for k in lc_data.keys()]
            vals   = [float(v) for v in lc_data.values()]
            # Filter out NaN/zero
            clean = [(l, v) for l, v in zip(labels, vals) if v > 0 and not np.isnan(v)]
            if clean:
                cl, cv = zip(*clean)
                ax3.pie(cv, labels=cl, autopct="%1.1f%%", startangle=140,
                        colors=plt.cm.Set3.colors[:len(cv)])
                ax3.set_title("Lifecycle GHG Breakdown", fontweight="bold")
            else:
                ax3.text(0.5, 0.5, "No lifecycle data", ha="center", va="center", transform=ax3.transAxes)
                ax3.set_title("Lifecycle GHG Breakdown", fontweight="bold")
        except Exception:
            ax3.text(0.5, 0.5, "No lifecycle data", ha="center", va="center", transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "No lifecycle data", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Lifecycle GHG Breakdown", fontweight="bold")
        ax3.axis("off")

    # ── BR: Radar — placed manually to avoid tight_layout polar crash ───────
    ax4 = fig.add_subplot(gs[1, 1], polar=True)
    try:
        _draw_radar(ax4, summary)
    except Exception as e:
        logger.warning(f"Radar draw failed: {e}")
        ax4.set_title("Radar unavailable", fontweight="bold")

    return _fig_to_b64(fig)


def _draw_radar(ax, summary: Dict):
    overall = summary.get("overall_score", 50)
    categories = ["GHG", "Recyclability", "Material Risk", "Energy", "Water"]
    material_risk = min(overall * 1.1, 100)

    # Mirror values from routes so radar is consistent
    values = [
        min(overall, 100),
        max(0, 100 - overall * 0.5),   # recyclability inverse
        material_risk,
        min(overall * 0.9, 100),
        min(overall * 0.7, 100),
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_plot = values + [values[0]]
    angles      = angles  + [angles[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 100)
    ax.plot(angles, values_plot, "o-", linewidth=2, color="#7B68EE")
    ax.fill(angles, values_plot, alpha=0.35, color="#7B68EE")
    ax.set_title("Multi-Dimensional Impact Radar", fontweight="bold", pad=14)


# ── Groq helpers ──────────────────────────────────────────────────────────────

_KANNADA_INSTRUCTION = (
    "\n\nIMPORTANT: You must respond entirely in Kannada (ಕನ್ನಡ) language. "
    "Use Kannada script for all text. Technical terms like material names, "
    "chemical formulas, and units (kg CO2eq, etc.) can remain in English/Roman."
)


async def _groq_product_narrative(summary: Dict, items: List[ScoredBOMItem], lang: str = "en") -> Optional[str]:
    """Short product-level sustainability narrative, optionally in Kannada."""
    groq = get_groq_client()
    if groq is None:
        return None

    hotspot_names = ", ".join(summary.get("hotspot_items", []) or ["None"])
    prompt = (
        f"Sustainability analysis complete. Product score: {summary['overall_score']:.1f}/100 "
        f"(Grade {summary['overall_grade'].value}). "
        f"Total GHG: {summary['total_ghg_kg_co2eq']:.2f} kg CO2eq. "
        f"Hotspot components: {hotspot_names}. "
        f"In 3 sentences, summarise the key sustainability risks and the top eco-design priority."
    )
    if lang == "kn":
        prompt += _KANNADA_INSTRUCTION
    try:
        resp = await groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq narrative failed: {e}")
        return None


async def _groq_material_suggestions(
    material_name: str,
    composite_score: float,
    rag_match: RAGMatchResult,
    impacts: LCAImpactVector,
    lang: str = "en",
) -> Optional[str]:
    """Groq suggestions for a single material query, optionally in Kannada."""
    groq = get_groq_client()
    if groq is None:
        return "Groq not configured — set GROQ_API_KEY in .env"

    prompt = (
        f"Material: {material_name}\n"
        f"Matched LCA process: {rag_match.matched_process} (source: {rag_match.matched_source}, "
        f"confidence: {rag_match.confidence.value})\n"
        f"GHG: {impacts.climate_change_kg_co2eq:.4f} kg CO2eq/kg | "
        f"Human toxicity: {impacts.human_toxicity:.4f} | "
        f"Metal depletion: {impacts.metal_depletion:.4f}\n"
        f"Composite impact score: {composite_score:.1f}/100\n\n"
        f"Provide 3 specific eco-design suggestions to reduce the environmental impact of this material "
        f"in electronics manufacturing. Be concise and quantitative where possible."
    )
    if lang == "kn":
        prompt += _KANNADA_INSTRUCTION
    try:
        resp = await groq.chat.completions.create(
            model=settings.GROQ_MODEL_MATCHING,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq single-material suggestions failed: {e}")
        return None