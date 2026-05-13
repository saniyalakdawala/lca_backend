"""
app/services/scoring/engine.py
SustainabilityScoreEngine
─────────────────────────
Deterministic, auditable scoring of eBOM line items.

Composite score formula:
  S = w_crit * normalize(criticality)
    + w_tox  * normalize(human_toxicity + metal_depletion)
    + w_ghg  * blend(intensity_score=0.4, absolute_score=0.6)

LCA impact fields (climate_change_kg_co2eq, human_toxicity, metal_depletion) are
per-kg-of-material intensity values from Ecoinvent/LCA databases. Absolute contribution
is derived by multiplying intensity × quantity_kg — this is what scores and hotspot
detection must both use for consistency.

All weights and thresholds are configurable via settings.
Grade: A (≤30), B (31–60), C (61–80), D (>80).
GHG hotspot: item contributes >20 % of product total GHG.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from config import settings
from ebom import (
    ScoredBOMItem, SustainabilityGrade, LCAImpactVector,
    RAGMatchResult, BOMLineItem,
)

logger = logging.getLogger(__name__)

# Criticality tiers — materials on EU/US critical raw material lists
CRITICAL_MATERIALS = {
    "high": {
        "cobalt", "lithium", "rare earth", "neodymium", "dysprosium",
        "indium", "gallium", "germanium", "platinum", "palladium",
        "iridium", "rhodium", "ruthenium", "osmium", "tellurium",
        "hafnium", "niobium", "tantalum", "tungsten", "antimony",
        "beryllium", "graphite", "magnesium", "silicon metal",
    },
    "medium": {
        "copper", "aluminium", "aluminum", "nickel", "chromium",
        "manganese", "lead", "zinc", "tin", "titanium", "vanadium",
        "molybdenum", "bismuth", "boron", "fluorspar",
    },
    "low": {
        "steel", "iron", "glass", "plastic", "polyethylene", "polypropylene",
        "nylon", "rubber", "paper", "wood", "cotton", "wool",
    },
}

CRITICALITY_SCORES = {"high": 90.0, "medium": 55.0, "low": 20.0, "unknown": 40.0}


class SustainabilityScoreEngine:
    """
    Transform matched LCA impact vectors into normalized 0–100 scores with letter grades.
    This class is stateless and can be instantiated once and reused.
    """

    def __init__(self):
        self.w_crit = settings.SCORE_WEIGHT_CRITICALITY
        self.w_tox  = settings.SCORE_WEIGHT_TOXICITY
        self.w_ghg  = settings.SCORE_WEIGHT_GHG
        self.ghg_benchmark = settings.GHG_BENCHMARK_KG_CO2_KWH

        # Per-kg intensity reference values (95th percentile from bottles_processes dataset)
        self._ghg_ref    = 12.5     # kg CO2eq per kg material (high end: virgin aluminium ~12kg)
        self._tox_ref    = 150.0    # human toxicity CTUh per kg material
        self._metal_ref  = 1.5      # metal depletion kg Fe-eq per kg material

        # Absolute contribution reference values (total item impact)
        self._abs_ghg_ref = 500.0   # kg CO2eq total — score 100 at this threshold

    # ── Public API ──────────────────────────────────────────────────────────

    def score_item(
        self,
        bom_item: BOMLineItem,
        rag_match: RAGMatchResult,
        lca_impacts: LCAImpactVector,
    ) -> Tuple[float, float, float, float, SustainabilityGrade]:
        """
        Returns (criticality_score, toxicity_score, ghg_score, composite_score, grade).
        All scores are 0–100.
        """
        crit_score  = self._criticality_score(bom_item.material_name)
        tox_score   = self._toxicity_score(lca_impacts)
        ghg_score   = self._ghg_score(lca_impacts, bom_item.quantity_kg)
        composite   = (
            self.w_crit * crit_score +
            self.w_tox  * tox_score  +
            self.w_ghg  * ghg_score
        )
        composite   = float(np.clip(composite, 0, 100))
        grade       = self._assign_grade(composite)
        logger.debug(
            "[SCORE] %s | qty=%.3fkg | ghg_intensity=%.4f | total_ghg=%.2f | "
            "ghg_score=%.1f | composite=%.1f | grade=%s",
            bom_item.material_name, bom_item.quantity_kg,
            lca_impacts.climate_change_kg_co2eq,
            lca_impacts.climate_change_kg_co2eq * bom_item.quantity_kg,
            ghg_score, composite, grade,
        )
        return crit_score, tox_score, ghg_score, composite, grade

    def score_product(
        self,
        bom_items: List[BOMLineItem],
        rag_matches: List[RAGMatchResult],
        lca_impacts_list: List[LCAImpactVector],
    ) -> List[ScoredBOMItem]:
        """
        Score all items, flag GHG hotspots, attach eco-recommendations.
        """
        # First pass: score each item
        scored = []
        for bom_item, match, impacts in zip(bom_items, rag_matches, lca_impacts_list):
            c, t, g, comp, grade = self.score_item(bom_item, match, impacts)
            scored.append(ScoredBOMItem(
                item_id=bom_item.item_id,
                material_name=bom_item.material_name,
                quantity_kg=bom_item.quantity_kg,
                rag_match=match,
                lca_impacts=impacts,
                criticality_score=round(c, 2),
                toxicity_score=round(t, 2),
                ghg_score=round(g, 2),
                composite_score=round(comp, 2),
                grade=grade,
            ))

        # Second pass: mark GHG hotspots (>20% of product total)
        total_ghg = sum(s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg for s in scored)
        for s in scored:
            item_ghg = s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg
            if total_ghg > 0 and (item_ghg / total_ghg) > 0.20:
                s.is_hotspot = True
                s.hotspot_reason = (
                    f"Contributes {(item_ghg/total_ghg)*100:.1f}% of total product GHG emissions "
                    f"({item_ghg:.3f} kg CO2eq)"
                )
            s.eco_recommendations = self._eco_recommendations(s)

        return scored

    def compute_product_summary(self, scored_items: List[ScoredBOMItem]) -> Dict:
        """Aggregate product-level metrics from scored items."""
        total_mass = sum(s.quantity_kg for s in scored_items)
        total_ghg  = sum(s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg for s in scored_items)

        # GHG-contribution-weighted composite score: heaviest GHG emitters dominate the grade
        if total_ghg > 0:
            overall_score = sum(
                s.composite_score * (s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg)
                for s in scored_items
            ) / total_ghg
        elif total_mass > 0:
            overall_score = sum(s.composite_score for s in scored_items) / len(scored_items)
        else:
            overall_score = 0
        overall_grade = self._assign_grade(overall_score)

        hotspot_items = [s.item_id for s in scored_items if s.is_hotspot]

        # Lifecycle stage breakdown (GHG)
        lifecycle_ghg: Dict[str, float] = {}
        for s in scored_items:
            stage = s.rag_match.lifecycle_stage or "Unknown"
            lifecycle_ghg[stage] = lifecycle_ghg.get(stage, 0) + (
                s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg
            )

        # Material breakdown (% GHG share)
        material_ghg: Dict[str, float] = {}
        for s in scored_items:
            mat = s.material_name
            material_ghg[mat] = material_ghg.get(mat, 0) + (
                s.lca_impacts.climate_change_kg_co2eq * s.quantity_kg
            )
        material_share = {
            mat: round((ghg / total_ghg) * 100, 2) if total_ghg > 0 else 0
            for mat, ghg in sorted(material_ghg.items(), key=lambda x: -x[1])
        }

        return {
            "overall_score": round(overall_score, 2),
            "overall_grade": overall_grade,
            "total_mass_kg": round(total_mass, 3),
            "total_ghg_kg_co2eq": round(total_ghg, 4),
            "hotspot_items": hotspot_items,
            "lifecycle_breakdown": lifecycle_ghg,
            "material_breakdown": material_share,
        }

    # ── Private Scoring Components ──────────────────────────────────────────

    def _criticality_score(self, material_name: str) -> float:
        name_lower = material_name.lower()
        for tier, keywords in CRITICAL_MATERIALS.items():
            if any(kw in name_lower for kw in keywords):
                return CRITICALITY_SCORES[tier]
        return CRITICALITY_SCORES["unknown"]

    def _toxicity_score(self, impacts: LCAImpactVector) -> float:
        """Normalize human toxicity + metal depletion to 0–100.
        Impact fields are per-kg intensity — do NOT divide by qty_kg again."""
        combined = (impacts.human_toxicity / self._tox_ref) * 70 + (impacts.metal_depletion / self._metal_ref) * 30
        return float(np.clip(combined, 0, 100))

    def _ghg_score(self, impacts: LCAImpactVector, qty_kg: float) -> float:
        """Blend of material GHG intensity (40%) and absolute item contribution (60%).
        Impact field is per-kg intensity — do NOT divide by qty_kg again."""
        ghg_intensity  = impacts.climate_change_kg_co2eq          # kg CO2eq per kg material
        total_ghg      = ghg_intensity * qty_kg                   # kg CO2eq total for this item

        intensity_score = np.clip((ghg_intensity / self._ghg_ref) * 100, 0, 100)
        abs_score       = np.clip((total_ghg / self._abs_ghg_ref) * 100, 0, 100)

        return float(0.4 * intensity_score + 0.6 * abs_score)

    def _assign_grade(self, score: float) -> SustainabilityGrade:
        if score <= settings.GRADE_A_MAX:
            return SustainabilityGrade.A
        elif score <= settings.GRADE_B_MAX:
            return SustainabilityGrade.B
        elif score <= settings.GRADE_C_MAX:
            return SustainabilityGrade.C
        return SustainabilityGrade.D

    def _eco_recommendations(self, item: ScoredBOMItem) -> List[str]:
        recs = []
        mat = item.material_name.lower()

        if item.criticality_score >= 80:
            recs.append(f"⚠️ High criticality — explore substitute materials with lower supply-chain risk.")
        if item.ghg_score >= 70:
            recs.append("🌍 High GHG footprint — consider recycled-content or bio-based alternatives.")
        if item.toxicity_score >= 70:
            recs.append("☣️ High toxicity profile — investigate material substitution or closed-loop recovery.")
        if "alumin" in mat and item.ghg_score >= 50:
            recs.append("♻️ Switch from primary to secondary (recycled) aluminium to reduce GHG by ~90%.")
        if "cobalt" in mat:
            recs.append("🔋 Cobalt — investigate cobalt-free battery chemistry (LFP) or recycled cobalt sourcing.")
        if "plastic" in mat or "polyethylene" in mat or "polypropylene" in mat:
            recs.append("🌱 Consider bio-based or ocean-bound recycled plastic alternatives.")
        if item.is_hotspot:
            recs.append("🔥 Carbon hotspot — priority target for design-for-sustainability iteration.")

        return recs or ["✅ Impact within acceptable range — continue monitoring."]