"""
app.py  —  AI4LCI Sustainability Intelligence Engine
Streamlit frontend — complete dashboard on single "Run Analysis" click.
"""

import streamlit as st
import requests
import pandas as pd
import base64
import json
from io import BytesIO

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="AI4LCI | Sustainability Intelligence Engine",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8ecf0;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-label { font-size: 12px; color: #8a94a6; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 34px; font-weight: 700; color: #1a1f36; font-family: 'Space Mono', monospace; line-height: 1.1; }
.kpi-sub   { font-size: 11px; color: #a0aab8; margin-top: 4px; }

/* ── Grade badges ── */
.badge-A { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-B { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-C { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:20px; font-weight:700; font-size:13px; }
.badge-hotspot { background:#fff1f0; color:#cf1322; padding:3px 10px; border-radius:20px; font-weight:600; font-size:12px; border:1px solid #ffa39e; }

/* ── Section headers ── */
.section-title { font-size:18px; font-weight:700; color:#1a1f36; margin:28px 0 14px; border-left:4px solid #10b981; padding-left:12px; }

/* ── Component table ── */
.comp-table { width:100%; border-collapse:collapse; font-size:13px; }
.comp-table th { background:#f8fafc; color:#64748b; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; padding:10px 14px; text-align:left; border-bottom:2px solid #e2e8f0; }
.comp-table td { padding:10px 14px; border-bottom:1px solid #f1f5f9; color:#374151; vertical-align:middle; }
.comp-table tr:hover td { background:#fafbfd; }
.hotspot-row td { background:#fff8f8 !important; }

/* ── Suggestion cards ── */
.sugg-card { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.sugg-card.high  { background:#fff1f0; border-color:#ffa39e; }
.sugg-card.medium{ background:#fffbe6; border-color:#ffe58f; }
.sugg-card .comp-name { font-weight:700; font-size:13px; color:#1a1f36; margin-bottom:4px; }
.sugg-card .sugg-text  { font-size:13px; color:#374151; line-height:1.5; }
.sugg-card .impact-tag { font-size:11px; background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-weight:600; display:inline-block; margin-top:6px; }

/* ── Radar number strip ── */
.radar-strip { display:flex; justify-content:space-between; background:#f8fafc; border-radius:10px; padding:12px 20px; margin-top:8px; border:1px solid #e2e8f0; }
.radar-dim { text-align:center; }
.radar-dim .dim-val { font-size:22px; font-weight:700; color:#1a1f36; font-family:'Space Mono',monospace; }
.radar-dim .dim-label { font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; }

/* ── Narrative box ── */
.narrative-box { background:linear-gradient(135deg,#ecfdf5 0%,#f0fdf4 100%); border:1px solid #a7f3d0; border-radius:12px; padding:18px 22px; font-size:14px; color:#064e3b; line-height:1.7; }
.narrative-box .nar-label { font-size:11px; font-weight:700; color:#059669; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px; }

/* ── Error/warning ── */
.err-box { background:#fff1f0; border:1px solid #ffa39e; border-radius:10px; padding:14px 18px; color:#a8071a; font-size:13px; }

/* ── Streamlit overrides ── */
div[data-testid="stTabs"] button { font-weight:600; font-size:14px; }
div[data-testid="metric-container"] { background:#fff; border:1px solid #e8ecf0; border-radius:12px; padding:12px 16px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def grade_badge(grade: str) -> str:
    return f'<span class="badge-{grade}">{grade}</span>'

def score_bar(score: float, max_score: float = 100) -> str:
    pct = min(score / max_score * 100, 100)
    color = "#ef4444" if pct > 70 else "#f59e0b" if pct > 40 else "#10b981"
    return f"""
    <div style="background:#f1f5f9;border-radius:4px;height:7px;width:100%;min-width:80px;">
      <div style="background:{color};width:{pct:.1f}%;height:7px;border-radius:4px;"></div>
    </div>
    <span style="font-size:11px;color:#64748b;">{score:.1f}</span>
    """

def elec_bar(pct: float) -> str:
    w = min(pct * 4, 100)   # scale so 25% fills bar
    return f"""
    <div style="background:#f1f5f9;border-radius:4px;height:7px;width:100%;min-width:60px;">
      <div style="background:#6366f1;width:{w:.1f}%;height:7px;border-radius:4px;"></div>
    </div>
    <span style="font-size:11px;color:#64748b;">{pct:.2f}%</span>
    """

def kpi_card(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

def render_kpis(header: dict):
    score = header.get("sustainability_score", 0)
    grade = header.get("letter_grade", "N/A")
    ghg   = header.get("total_carbon_footprint", 0)
    hotspot = header.get("dominant_hotspot", "None")

    grade_color = {"A": "#10b981", "B": "#f59e0b", "C": "#ef4444"}.get(grade, "#6b7280")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Sustainability Score", f"{score:.1f}<span style='font-size:18px;color:#94a3b8'>/100</span>", "Overall environmental performance"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Letter Grade", f"<span style='color:{grade_color}'>{grade}</span>", "Based on lifecycle impact"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Total Carbon Footprint", f"{ghg:.2f}", "Cradle-to-gate emissions (kg CO2eq)"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Dominant Hotspot", hotspot, "Highest impact category"), unsafe_allow_html=True)

def render_radar_strip(radar: dict):
    dims_html = ""
    for dim, val in radar.items():
        dims_html += f"""
        <div class="radar-dim">
          <div class="dim-val">{val:.1f}</div>
          <div class="dim-label">{dim}</div>
        </div>"""
    st.markdown(f'<div class="radar-strip">{dims_html}</div>', unsafe_allow_html=True)

def render_component_table(components: list):
    rows_html = ""
    for c in components:
        grade = c.get("grade", "N/A")
        is_hot = c.get("is_hotspot", False)
        row_class = "hotspot-row" if is_hot else ""
        hotspot_tag = '<span class="badge-hotspot">🔥 Hotspot</span>' if is_hot else ""
        rag = c.get("rag_match", {})
        conf = rag.get("confidence", "")
        conf_color = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")
        rows_html += f"""
        <tr class="{row_class}">
          <td><b>{c.get('component','')}</b><br><span style="font-size:11px;color:#94a3b8;">{rag.get('process','')[:40]}</span></td>
          <td>{c.get('weight_pct',0):.1f}%</td>
          <td><span style="font-size:12px;background:#f1f5f9;padding:2px 7px;border-radius:6px;">{c.get('materials','')}</span></td>
          <td>{elec_bar(c.get('electricity_pct', 0))}</td>
          <td>{score_bar(c.get('impact_score', 0))}</td>
          <td>{grade_badge(grade)}</td>
          <td>{hotspot_tag}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="comp-table">
      <thead><tr>
        <th>Component</th>
        <th>Weight %</th>
        <th>Materials</th>
        <th>GHG Share %</th>
        <th>Impact Score</th>
        <th>Grade</th>
        <th>Flag</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

def render_lca_breakdown_table(components: list):
    """Detailed LCA + score breakdown table."""
    rows_html = ""
    for c in components:
        bd = c.get("score_breakdown", {})
        lca = c.get("lca_impacts", {})
        grade = c.get("grade", "N/A")
        rows_html += f"""
        <tr>
          <td><b>{c.get('component','')}</b></td>
          <td>{grade_badge(grade)}</td>
          <td><b>{c.get('impact_score',0):.1f}</b></td>
          <td>{bd.get('criticality',0):.1f}</td>
          <td>{bd.get('toxicity',0):.1f}</td>
          <td>{bd.get('ghg',0):.1f}</td>
          <td>{lca.get('climate_change_kg_co2eq',0):.4f}</td>
          <td>{lca.get('human_toxicity',0):.4f}</td>
          <td>{lca.get('metal_depletion',0):.4f}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="comp-table">
      <thead><tr>
        <th>Component</th><th>Grade</th><th>Composite</th>
        <th>Criticality</th><th>Toxicity</th><th>GHG Score</th>
        <th>CO2eq (kg/kg)</th><th>Human Tox</th><th>Metal Dep.</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

def render_hotspot_cards(components: list):
    hotspots = [c for c in components if c.get("is_hotspot")]
    if not hotspots:
        st.info("✅ No GHG hotspots detected — all components below 20% share threshold.")
        return
    for h in hotspots:
        recs = h.get("eco_recommendations", [])
        recs_html = "".join(f"<li style='margin-bottom:4px'>{r}</li>" for r in recs)
        st.markdown(f"""
        <div style="background:#fff8f8;border:1px solid #fca5a5;border-radius:12px;padding:16px 20px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:22px">🔥</span>
            <span style="font-weight:700;font-size:16px;color:#991b1b;">{h.get('component','')}</span>
            <span style="font-size:12px;color:#b91c1c;background:#fee2e2;padding:2px 10px;border-radius:20px;">
              Score {h.get('impact_score',0):.1f} · Grade {h.get('grade','N/A')}
            </span>
          </div>
          <p style="color:#7f1d1d;font-size:13px;margin:0 0 8px;">{h.get('hotspot_reason','')}</p>
          <ul style="color:#374151;font-size:13px;margin:0;padding-left:18px;">{recs_html}</ul>
        </div>""", unsafe_allow_html=True)

def render_groq_suggestions(suggestions):
    if not suggestions:
        return
    for s in suggestions:
        priority = s.get("priority", "low")
        card_class = "high" if priority == "high" else ("medium" if priority == "medium" else "")
        impact = s.get("impact_reduction_est", "")
        st.markdown(f"""
        <div class="sugg-card {card_class}">
          <div class="comp-name">🔧 {s.get('component','')}</div>
          <div class="sugg-text">{s.get('suggestion','')}</div>
          {f'<span class="impact-tag">Est. reduction: {impact}</span>' if impact else ''}
        </div>""", unsafe_allow_html=True)

def render_rag_match_info(components: list):
    """Shows RAG provenance for each component."""
    rows_html = ""
    for c in components:
        rag = c.get("rag_match", {})
        sim = rag.get("similarity", 0)
        conf = rag.get("confidence", "")
        is_synth = rag.get("is_synthetic", False)
        conf_color = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")
        source_tag = "🤖 Synthetic" if is_synth else f"📚 {rag.get('source','')}"
        reasoning = rag.get("llm_reasoning") or "Vector match"
        rows_html += f"""
        <tr>
          <td><b>{c.get('component','')}</b></td>
          <td style="font-size:12px">{rag.get('process','')[:50]}</td>
          <td><span style="color:{conf_color};font-weight:600;">{conf.upper()}</span></td>
          <td>{sim:.3f}</td>
          <td style="font-size:11px;color:#6b7280;">{source_tag}</td>
          <td style="font-size:11px;color:#6b7280;max-width:220px;">{reasoning[:120]}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;">
    <table class="comp-table">
      <thead><tr>
        <th>Component</th><th>Matched LCA Process</th>
        <th>Confidence</th><th>Similarity</th><th>Source</th><th>LLM Reasoning</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌱 AI4LCI")
    st.caption("Sustainability Intelligence Engine")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"✅ Backend: {health.get('status','ok')}")
        groq_ok = health.get("groq_ready", False)
        st.info(f"🤖 Groq: {'Ready' if groq_ok else 'Not configured'}")
        st.caption(f"Engine: {health.get('engine','')}")
    except Exception:
        st.error("❌ Backend Offline\nRun: `uvicorn main:app --reload`")

    st.divider()
    st.markdown("**About**")
    st.caption("Automated eBOM → LCA scoring using RAG + Groq AI. Maps materials to Ecoinvent benchmarks and scores Criticality, Toxicity, and GHG (0.71 kg CO2/kWh baseline).")
    st.divider()
    st.caption("L&T Technology Services | Restricted")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📊 BOM Analysis", "🔍 RAG Material Search"])


# ── TAB 1: FULL BOM ANALYSIS ──────────────────────────────────────────────────
with tab1:
    st.markdown("## Upload Electronic BOM")
    st.caption("Upload a CSV file. Required columns: component/material name + quantity/weight.")

    uploaded_file = st.file_uploader("Choose eBOM CSV", type="csv", key="bom_upload")

    if uploaded_file:
        # Preview
        try:
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            with st.expander(f"📄 Preview — {len(preview_df)} rows, {len(preview_df.columns)} columns", expanded=False):
                st.dataframe(preview_df.head(10), use_container_width=True)
        except Exception:
            pass

        run_btn = st.button("🚀 Run Sustainability Analysis", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("⏳ Running RAG matching + Groq AI scoring..."):
                try:
                    uploaded_file.seek(0)
                    response = requests.post(
                        f"{BACKEND_URL}/upload-bom",
                        files={"file": ("bom.csv", uploaded_file.read(), "text/csv")},
                        timeout=120,
                    )
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach backend. Is uvicorn running on port 8000?")
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out after 120s. Try a smaller BOM.")
                    st.stop()

            if response.status_code != 200:
                st.markdown(f'<div class="err-box">❌ Backend error {response.status_code}: {response.text[:400]}</div>', unsafe_allow_html=True)
                st.stop()

            data = response.json()
            st.success("✅ Analysis complete!")

            header     = data.get("dashboard_header", {})
            radar      = data.get("radar_chart_data", {})
            components = data.get("component_analysis", [])
            charts     = data.get("charts", {})
            summary    = data.get("product_summary", {})
            narrative  = data.get("groq_narrative")

            # ── 1. KPI Header ────────────────────────────────────────────────
            st.markdown('<div class="section-title">Analysis Dashboard</div>', unsafe_allow_html=True)
            render_kpis(header)

            # ── 2. Groq Narrative ────────────────────────────────────────────
            if narrative:
                st.markdown(f"""
                <div class="narrative-box">
                  <div class="nar-label">🤖 Groq AI Assessment</div>
                  {narrative}
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # ── 3. Radar strip ───────────────────────────────────────────────
            st.markdown('<div class="section-title">Multi-Dimensional Impact Analysis</div>', unsafe_allow_html=True)
            render_radar_strip(radar)

            # ── 4. Bar charts ────────────────────────────────────────────────
            st.markdown('<div class="section-title">Component & GHG Charts</div>', unsafe_allow_html=True)
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                impact_b64 = charts.get("component_impact_bar", "")
                if impact_b64:
                    st.image(base64.b64decode(impact_b64), caption="Component Impact Scores", use_container_width=True)
            with col_chart2:
                ghg_b64 = charts.get("ghg_emissions_bar", "")
                if ghg_b64:
                    st.image(base64.b64decode(ghg_b64), caption="GHG Emissions by Component", use_container_width=True)

            # ── 5. Component table ───────────────────────────────────────────
            st.markdown('<div class="section-title">Component-Level Analysis</div>', unsafe_allow_html=True)

            # Quick summary stats
            total = len(components)
            grade_counts = {}
            for c in components:
                g = c.get("grade", "?")
                grade_counts[g] = grade_counts.get(g, 0) + 1
            hotspot_count = sum(1 for c in components if c.get("is_hotspot"))

            ms1, ms2, ms3, ms4, ms5 = st.columns(5)
            ms1.metric("Total Components", total)
            ms2.metric("Grade A", grade_counts.get("A", 0), delta="Low impact", delta_color="normal")
            ms3.metric("Grade B", grade_counts.get("B", 0))
            ms4.metric("Grade C", grade_counts.get("C", 0), delta="High impact", delta_color="inverse")
            ms5.metric("Hotspots", hotspot_count, delta="Need action" if hotspot_count else "None", delta_color="inverse" if hotspot_count else "off")

            render_component_table(components)

            # ── 6. LCA Breakdown ─────────────────────────────────────────────
            with st.expander("🔬 Detailed LCA & Score Breakdown", expanded=False):
                render_lca_breakdown_table(components)

            # ── 7. Hotspot Analysis ──────────────────────────────────────────
            st.markdown('<div class="section-title">⚠️ Hotspot Analysis & Eco-Recommendations</div>', unsafe_allow_html=True)
            render_hotspot_cards(components)

            # ── 8. Groq Eco-Design Suggestions ──────────────────────────────
            st.markdown('<div class="section-title">🤖 Groq AI Eco-Design Suggestions</div>', unsafe_allow_html=True)

            with st.spinner("Generating Groq AI eco-design suggestions..."):
                try:
                    sugg_resp = requests.post(
                        f"{BACKEND_URL}/suggestions/eco-design",
                        json={"components": components},
                        timeout=60,
                    )
                    if sugg_resp.status_code == 200:
                        suggestions = sugg_resp.json().get("suggestions", [])
                        if suggestions:
                            render_groq_suggestions(suggestions)
                        else:
                            st.info("No suggestions returned.")
                    else:
                        st.warning(f"Suggestions endpoint returned {sugg_resp.status_code}: {sugg_resp.text[:200]}")
                except Exception as e:
                    st.warning(f"Could not fetch Groq suggestions: {e}")

            # ── 9. RAG Provenance ────────────────────────────────────────────
            with st.expander("🔗 RAG Match Provenance (how each material was matched)", expanded=False):
                render_rag_match_info(components)

            # ── 10. 2×2 Combined Visual ──────────────────────────────────────
            with st.expander("📈 2×2 Multi-Dimensional Impact Matrix", expanded=False):
                with st.spinner("Generating combined chart..."):
                    try:
                        viz_resp = requests.get(f"{BACKEND_URL}/visualizations", timeout=30).json()
                        if "chart" in viz_resp:
                            st.image(base64.b64decode(viz_resp["chart"]), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Visualization chart failed: {e}")

            # ── 11. Raw JSON download ────────────────────────────────────────
            with st.expander("⬇️ Download Full Analysis JSON", expanded=False):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(data, indent=2),
                    file_name="sustainability_analysis.json",
                    mime="application/json",
                )

    else:
        st.info("👆 Upload a CSV file to begin. The engine will automatically detect your column names.")
        st.markdown("""
        **Expected columns (flexible naming):**
        | Column | Accepted names |
        |---|---|
        | Material / Component name | `material_name`, `Component`, `Material`, `Name` |
        | Quantity / Weight (kg) | `quantity_kg`, `Weight`, `Qty`, `Quantity` |
        | Lifecycle stage (optional) | `lifecycle_stage`, `Stage` |
        """)


# ── TAB 2: RAG MATERIAL SEARCH ────────────────────────────────────────────────
with tab2:
    st.markdown("## Single Material RAG Search")
    st.caption("Search any material by name — the engine RAG-matches it to Ecoinvent LCA data and scores it with Groq AI.")

    col_input, col_qty = st.columns([3, 1])
    with col_input:
        comp_name = st.text_input("Material name", placeholder="e.g. Tempered Glass, Li-ion Battery, Cobalt")
    with col_qty:
        qty_kg = st.number_input("Quantity (kg)", min_value=0.01, value=1.0, step=0.1)

    stage = st.selectbox("Lifecycle stage", ["manufacturing", "raw material extraction", "transport", "use phase", "end of life"], index=0)

    search_btn = st.button("🔍 Calculate Impact Score", type="primary")

    if search_btn and comp_name:
        with st.spinner(f"RAG matching '{comp_name}'..."):
            try:
                res = requests.get(
                    f"{BACKEND_URL}/scoring/{requests.utils.quote(comp_name)}",
                    params={"quantity_kg": qty_kg, "lifecycle_stage": stage},
                    timeout=60,
                ).json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        if "detail" in res:
            st.error(f"Backend error: {res['detail']}")
            st.stop()

        # ── KPI row ──────────────────────────────────────────────────────────
        grade = res.get("grade", "N/A")
        score = res.get("composite_score", 0)
        is_hot = res.get("is_hotspot", False)
        grade_color = {"A": "#10b981", "B": "#f59e0b", "C": "#ef4444"}.get(grade, "#6b7280")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(kpi_card("Grade", f"<span style='color:{grade_color}'>{grade}</span>", "A=Low · B=Med · C=High"), unsafe_allow_html=True)
        c2.markdown(kpi_card("Composite Score", f"{score:.1f}", "0–100 weighted"), unsafe_allow_html=True)
        c3.markdown(kpi_card("Hotspot?", "🔥 YES" if is_hot else "✅ NO", "Score > 60"), unsafe_allow_html=True)
        bd = res.get("score_breakdown", {})
        c4.markdown(kpi_card("GHG Score", f"{bd.get('ghg',0):.1f}", "Normalized 0–100"), unsafe_allow_html=True)

        st.markdown("")

        # ── Score breakdown ───────────────────────────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Score Breakdown</div>', unsafe_allow_html=True)
            st.progress(int(bd.get("criticality", 0)), text=f"Criticality: {bd.get('criticality',0):.1f}/100")
            st.progress(int(bd.get("toxicity", 0)),    text=f"Toxicity: {bd.get('toxicity',0):.1f}/100")
            st.progress(int(bd.get("ghg", 0)),         text=f"GHG: {bd.get('ghg',0):.1f}/100")

        with col_b:
            st.markdown('<div class="section-title">LCA Impact Values</div>', unsafe_allow_html=True)
            lca = res.get("lca_impacts", {})
            lca_df = pd.DataFrame([
                {"Indicator": "Climate Change (kg CO2eq/kg)", "Value": lca.get("climate_change_kg_co2eq", 0)},
                {"Indicator": "Human Toxicity (CTUh)",        "Value": lca.get("human_toxicity", 0)},
                {"Indicator": "Metal Depletion (kg Fe-eq)",   "Value": lca.get("metal_depletion", 0)},
            ])
            st.dataframe(lca_df, use_container_width=True, hide_index=True)

        # ── RAG Match ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">RAG Match Result</div>', unsafe_allow_html=True)
        rag = res.get("rag_match", {})
        conf = rag.get("confidence", "")
        conf_color = {"high": "#10b981", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")
        synth_tag = "🤖 Groq Synthetic estimate" if rag.get("is_synthetic") else f"📚 {rag.get('source','')}"
        reasoning = rag.get("llm_reasoning") or "Top vector match"

        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;">
          <div style="margin-bottom:8px;">
            <b>Matched Process:</b>
            <span style="font-family:'Space Mono',monospace;font-size:13px;color:#1a1f36;margin-left:8px;">
              {rag.get('process','')}
            </span>
          </div>
          <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:13px;">
            <span><b>Confidence:</b> <span style="color:{conf_color};font-weight:700;">{conf.upper()}</span></span>
            <span><b>Similarity:</b> {rag.get('similarity',0):.3f}</span>
            <span><b>Source:</b> {synth_tag}</span>
          </div>
          <div style="margin-top:10px;font-size:13px;color:#64748b;background:#fff;border-radius:8px;padding:10px 14px;border:1px solid #f1f5f9;">
            <b>LLM Reasoning:</b> {reasoning}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Groq Suggestions ──────────────────────────────────────────────────
        suggestions_text = res.get("groq_suggestions")
        if suggestions_text:
            st.markdown('<div class="section-title">🤖 Groq AI Eco-Design Suggestions</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="narrative-box">
              <div class="nar-label">Groq LLaMA-70b Recommendations</div>
              {suggestions_text.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)

    elif search_btn:
        st.warning("Please enter a material name.")