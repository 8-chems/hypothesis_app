"""
app.py — StatSense: guided hypothesis testing for non-experts.
Run with: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.data_loader   import load_data
from core.wizard        import run_wizard
from core.test_runner   import select_columns, run_test
from core.interpreter   import interpret
from plots.distribution_plot  import make_distribution_plot
from plots.diagnostic_plots   import (
    group_box_plot, group_histogram, paired_change_plot,
    qq_plots, scatter_regression, differences_histogram,
    contingency_heatmap, effect_size_gauge,
)
from theory.cards        import get_theory_card, render_theory_card
from theory.academy_page import render_academy_page

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatSense · Hypothesis Testing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Global background ── */
.stApp {
    background: linear-gradient(135deg, #060a14 0%, #0c1225 50%, #0a1020 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,16,32,0.95) !important;
    border-right: 1px solid rgba(79,142,247,0.15);
}
[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}

/* ── Sidebar header ── */
.sidebar-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #4F8EF7 !important;
    letter-spacing: -0.02em;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(79,142,247,0.2);
    margin-bottom: 1.2rem;
}

/* ── Result verdict card ── */
.verdict-card {
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.verdict-yes {
    background: linear-gradient(135deg, rgba(52,211,153,0.08) 0%, rgba(79,142,247,0.06) 100%);
    border-color: rgba(52,211,153,0.35);
}
.verdict-no {
    background: linear-gradient(135deg, rgba(255,75,110,0.07) 0%, rgba(162,89,255,0.05) 100%);
    border-color: rgba(255,75,110,0.25);
}
.verdict-card h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
}
.verdict-yes h2  { color: #34D399; }
.verdict-no  h2  { color: #FF8FA3; }
.verdict-card p  { color: #c8d0e0; font-size: 0.97rem; line-height: 1.6; margin: 0.3rem 0; }
.verdict-card .detail { color: #8899b0; font-size: 0.88rem; }
.verdict-card .effect-line {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid rgba(255,255,255,0.07);
    font-size: 0.88rem;
    color: #a0b0c8;
}

/* ── Assumption pill ── */
.check-ok   { background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3);
              border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.5rem; color:#34D399; font-size:0.85rem; }
.check-warn { background:rgba(255,179,71,0.10); border:1px solid rgba(255,179,71,0.3);
              border-radius:8px; padding:0.5rem 0.8rem; margin-bottom:0.5rem; color:#FFB347; font-size:0.85rem; }

/* ── Section label ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(79,142,247,0.15);
}

/* ── Test badge ── */
.test-badge {
    display:inline-block;
    background:rgba(79,142,247,0.12);
    border:1px solid rgba(79,142,247,0.3);
    border-radius:6px;
    padding:0.3rem 0.7rem;
    font-family:'IBM Plex Mono',monospace;
    font-size:0.78rem;
    color:#4F8EF7;
    margin-top:0.3rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:12px;
    padding:1rem 1.2rem;
    text-align:center;
}
.metric-tile .val {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.5rem;
    font-weight:600;
    color:#c8d0e0;
}
.metric-tile .lbl {
    font-size:0.75rem;
    color:#8899b0;
    margin-top:0.2rem;
    letter-spacing:0.04em;
}

/* ── Streamlit overrides ── */
.stSelectbox label, .stRadio label, .stSlider label,
.stTextArea label, .stFileUploader label { color: #c8d0e0 !important; }

.stRadio [data-testid="stMarkdownContainer"] p { color: #c8d0e0 !important; font-size:0.93rem; }

div[data-testid="metric-container"] { background:rgba(255,255,255,0.02); border-radius:10px; padding:0.5rem; }

.stExpander { border-color: rgba(255,255,255,0.08) !important; }
.stExpander summary { color: #8899b0 !important; font-size:0.88rem; }

hr { border-color: rgba(255,255,255,0.07) !important; }

h1,h2,h3 { color: #e8eef8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def section(label: str):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)


def render_verdict_card(interp: dict, config: dict):
    is_yes = interp["reject"]
    card_class = "verdict-yes" if is_yes else "verdict-no"
    icon = "✅" if is_yes else "❔"

    st.markdown(f"""
    <div class="verdict-card {card_class}">
        <h2>{icon} {interp['headline'].replace('**','').replace('*','')}</h2>
        <p>{interp['detail']}</p>
        <div class="effect-line">{interp['effect_sentence'].replace('**','').replace('*','')}</div>
        {"<div class='effect-line'>" + interp['ci_sentence'] + "</div>" if interp.get('ci_sentence') else ""}
    </div>
    """, unsafe_allow_html=True)


def render_assumption_checks(assumption_result: dict, config: dict):
    if assumption_result.get("switched"):
        orig = config.get("technical_name", "parametric test")
        st.info(
            f"💡 Your data didn't fully meet the assumptions for the parametric test, "
            f"so we automatically switched to a safer non-parametric equivalent. "
            f"The interpretation is the same — just more robust.",
            icon="🔄"
        )
    for check in assumption_result.get("checks", []):
        css = "check-ok" if check["status"] == "ok" else "check-warn"
        icon = "✓" if check["status"] == "ok" else "⚠"
        st.markdown(
            f'<div class="{css}"><strong>{icon} {check["label"]}</strong><br>{check["message"]}</div>',
            unsafe_allow_html=True,
        )
    with st.expander("🔬 Technical details of assumption checks"):
        for check in assumption_result.get("checks", []):
            st.markdown(f"**{check['label']}:** {check['technical']}")


def render_metrics_row(result: dict, interp: dict):
    cols = st.columns(4)
    p = result.get("p", 1.0)
    alpha = result.get("alpha", 0.05)
    stat_label = result.get("stat_label", "stat")
    stat_val = result.get("stat", 0.0)
    effect = result.get("effect_size", 0.0)
    effect_label = result.get("effect_label", "effect")

    p_color = "#34D399" if p < alpha else "#FF8FA3"

    with cols[0]:
        st.markdown(f"""<div class="metric-tile">
            <div class="val" style="color:{p_color}">{p:.4f}</div>
            <div class="lbl">p-value (threshold: {alpha})</div>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""<div class="metric-tile">
            <div class="val">{stat_val:.3f}</div>
            <div class="lbl">{stat_label} statistic</div>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f"""<div class="metric-tile">
            <div class="val">{abs(effect):.3f}</div>
            <div class="lbl">{effect_label}</div>
        </div>""", unsafe_allow_html=True)
    with cols[3]:
        conf = round((1 - alpha) * 100)
        st.markdown(f"""<div class="metric-tile">
            <div class="val">{conf}%</div>
            <div class="lbl">Confidence level</div>
        </div>""", unsafe_allow_html=True)


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    # ── Academy page toggle ──────────────────────────────────────────────────
    if st.session_state.get("show_academy", False):
        render_academy_page()
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">🔬 StatSense</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.8rem;color:#8899b0;margin-bottom:1.2rem">Hypothesis testing for everyone</div>', unsafe_allow_html=True)

        section("STEP 1 · YOUR DATA")
        df, dataset_type = load_data()

        if df is None:
            st.info("👆 Load some data to get started.")
            _show_landing()
            return

        st.success(f"Dataset ready: {len(df):,} rows × {len(df.columns)} columns")

        with st.expander("👀 Preview data"):
            st.dataframe(df.head(10), use_container_width=True, height=200)

        st.markdown("---")
        section("STEP 2 · YOUR QUESTION")
        config = run_wizard(dataset_type)

        st.markdown("---")
        section("STEP 3 · MAP COLUMNS")
        col_map = select_columns(df, config)

        if col_map is None:
            st.warning("Please select columns above.")
            _show_landing()
            return

        run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    # ── Main panel ────────────────────────────────────────────────────────────
    if "result" not in st.session_state:
        _show_landing()
        if "run_btn" in dir() and not run_btn:
            return

    if run_btn if "run_btn" in dir() else False:
        with st.spinner("Running analysis…"):
            result, assumption_result = run_test(df, config, col_map)
            interp = interpret(result, config)
            st.session_state["result"] = result
            st.session_state["assumption_result"] = assumption_result
            st.session_state["interp"] = interp
            st.session_state["config"] = config
            st.session_state["df"] = df
            st.session_state["col_map"] = col_map

    if "result" not in st.session_state:
        _show_landing()
        return

    result          = st.session_state["result"]
    assumption_result = st.session_state["assumption_result"]
    interp          = st.session_state["interp"]
    config          = st.session_state["config"]
    df_saved        = st.session_state["df"]
    col_map_saved   = st.session_state["col_map"]

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"## Results: {config['friendly_name']}")
        st.markdown(
            f'<span class="test-badge">Technical test: {config["technical_name"]}</span>',
            unsafe_allow_html=True,
        )
    with c2:
        reject_color = "#34D399" if result.get("reject") else "#FF8FA3"
        reject_text = "SIGNIFICANT" if result.get("reject") else "NOT SIGNIFICANT"
        st.markdown(f"""
        <div style="text-align:right;padding-top:0.5rem">
            <span style="font-family:'IBM Plex Mono';font-size:0.75rem;
                         font-weight:600;letter-spacing:0.1em;
                         color:{reject_color}">{reject_text}</span><br>
            <span style="font-family:'IBM Plex Mono';font-size:0.75rem;color:#8899b0">
                α = {result['alpha']}  ·  p = {result['p']:.4f}
            </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Assumption checks ─────────────────────────────────────────────────────
    with st.expander("🩺 Data quality checks — click to see", expanded=True):
        render_assumption_checks(assumption_result, config)

    st.markdown("")

    # ── Verdict card ──────────────────────────────────────────────────────────
    section("VERDICT")
    render_verdict_card(interp, config)

    # ── Metric row ────────────────────────────────────────────────────────────
    section("KEY NUMBERS")
    render_metrics_row(result, interp)

    st.markdown("")

    # ── Numbers expander ──────────────────────────────────────────────────────
    with st.expander("🔢 Show me the technical numbers"):
        st.code(interp["technical"], language=None)
        if result.get("ci"):
            ci = result["ci"]
            st.markdown(f"**95% Confidence interval:** [{ci[0]:.4f}, {ci[1]:.4f}]")
        if result.get("group_means"):
            names = result.get("group_names", result.get("_group_names", []))
            means = result["group_means"]
            stds  = result.get("group_stds", [])
            ns    = result.get("group_ns", [])
            rows = []
            for i, (m, name) in enumerate(zip(means, names)):
                row = {"Group": name, "Mean/Median": f"{m:.3f}"}
                if stds and i < len(stds):
                    row["Std Dev"] = f"{stds[i]:.3f}"
                if ns and i < len(ns):
                    row["n"] = ns[i]
                rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Visualizations ────────────────────────────────────────────────────────
    section("VISUALIZATIONS")

    test_id = result.get("test_id", "")
    groups  = result.get("_groups_raw", [])
    gnames  = result.get("_group_names", [])

    # Distribution plot
    tab_dist, tab_data, tab_diagnostics, tab_effect = st.tabs([
        "📊 Hypothesis test", "📈 Data explorer", "🔍 Diagnostics", "⚡ Effect size"
    ])

    with tab_dist:
        st.markdown("**What the test actually does** — the curve shows all possible outcomes by chance. Your result is the orange line. The red zones would make us reject the null hypothesis.")
        fig_dist = make_distribution_plot(result)
        st.plotly_chart(fig_dist, use_container_width=True)

        # P-value explanation visual
        p = result.get("p", 0.5)
        alpha_val = result.get("alpha", 0.05)
        _render_pvalue_meter(p, alpha_val)

    with tab_data:
        if test_id in ("independent_t", "mann_whitney", "anova", "kruskal") and len(groups) >= 2:
            value_label = col_map_saved.get("value_col", "Value")
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(group_box_plot(groups, gnames, value_label),
                                use_container_width=True)
            with col_b:
                st.plotly_chart(group_histogram(groups, gnames, value_label),
                                use_container_width=True)

        elif test_id in ("paired_t", "wilcoxon") and len(groups) >= 2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    paired_change_plot(groups[0], groups[1],
                                       gnames[0] if gnames else "Before",
                                       gnames[1] if len(gnames)>1 else "After"),
                    use_container_width=True)
            with col_b:
                diffs = result.get("diffs", groups[1] - groups[0])
                st.plotly_chart(
                    differences_histogram(diffs,
                                         gnames[0] if gnames else "Before",
                                         gnames[1] if len(gnames)>1 else "After"),
                    use_container_width=True)

        elif test_id in ("pearson_r", "spearman_r"):
            x = result.get("x", groups[0] if groups else np.array([]))
            y = result.get("y", groups[1] if len(groups)>1 else np.array([]))
            col_names = result.get("col_names", gnames)
            st.plotly_chart(
                scatter_regression(x, y,
                                   col_names[0] if col_names else "X",
                                   col_names[1] if len(col_names)>1 else "Y",
                                   r=result.get("stat", 0.0)),
                use_container_width=True)
            r2 = result.get("r_squared", result.get("stat", 0)**2)
            st.info(f"📊 R² = {r2:.3f} — {r2*100:.1f}% of the variation in one variable is explained by the other.")

        elif test_id == "chi2_indep":
            ct = result.get("contingency_table")
            if ct is not None:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(contingency_heatmap(ct, "Observed counts"),
                                    use_container_width=True)
                with col_b:
                    exp = result.get("expected")
                    if exp is not None:
                        st.plotly_chart(contingency_heatmap(exp.round(1), "Expected counts (if no relationship)"),
                                        use_container_width=True)

    with tab_diagnostics:
        if test_id not in ("chi2_indep", "mcnemar", "pearson_r", "spearman_r") and len(groups) >= 1:
            numeric_groups = [g for g in groups if len(g) > 2]
            if numeric_groups:
                valid_names = gnames[:len(numeric_groups)]
                st.plotly_chart(qq_plots(numeric_groups, valid_names), use_container_width=True)
                st.caption("**How to read Q-Q plots:** if points lie close to the diagonal line, your data is roughly bell-shaped (normal). Points curving away from the line suggest non-normality.")
        elif test_id in ("pearson_r", "spearman_r"):
            x = result.get("x", np.array([]))
            y = result.get("y", np.array([]))
            col_names = result.get("col_names", ["X", "Y"])
            if len(x) > 2 and len(y) > 2:
                st.plotly_chart(qq_plots([x, y], col_names), use_container_width=True)
        elif test_id == "chi2_indep":
            ct = result.get("contingency_table")
            exp = result.get("expected")
            if ct is not None and exp is not None:
                residuals = (ct - exp) / np.sqrt(exp.replace(0, np.nan))
                st.markdown("**Standardized residuals** — cells far from zero contribute most to the chi-square statistic:")
                st.plotly_chart(contingency_heatmap(residuals.round(2), "Standardized residuals"),
                                use_container_width=True)
                st.caption("Values > 2 or < -2 are unusually large and drive the result.")
        else:
            st.info("Diagnostic plots are not applicable for this test type.")

    with tab_effect:
        st.markdown("**Effect size tells you how big the difference is** — separate from whether it's statistically significant. A result can be significant but tiny (and practically meaningless), or large but not significant (because your sample is small).")

        effect_val  = result.get("effect_size", 0.0)
        effect_lbl  = result.get("effect_label", "effect size")
        test_id_now = result.get("test_id", "")

        # Threshold benchmarks per test type
        if test_id_now in ("independent_t", "paired_t", "mann_whitney", "wilcoxon"):
            thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}
        elif test_id_now in ("anova", "kruskal"):
            thresholds = {"small": 0.01, "medium": 0.06, "large": 0.14}
        elif test_id_now in ("pearson_r", "spearman_r"):
            thresholds = {"small": 0.1, "medium": 0.3, "large": 0.5}
        elif test_id_now == "chi2_indep":
            thresholds = {"small": 0.1, "medium": 0.3, "large": 0.5}
        else:
            thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

        st.plotly_chart(
            effect_size_gauge(effect_val, effect_lbl, thresholds),
            use_container_width=True,
        )

        # R² visualization for correlations
        if test_id_now in ("pearson_r", "spearman_r"):
            r2 = result.get("r_squared", effect_val**2)
            _render_r2_visual(r2)

        st.markdown("---")
        st.markdown("**Benchmarks for this test:**")
        for size, val in thresholds.items():
            st.markdown(f"- **{size.capitalize()}** effect: ≥ {val}")

    st.markdown("---")

    # ── Theory card for current test ─────────────────────────────────────────
    section("LEARN THE THEORY")
    card = get_theory_card(result.get("test_id", ""))
    if card:
        with st.expander(f"{card['emoji']} What is a {card['name']}? — click to learn", expanded=False):
            _render_theory_card_ui(card)

    st.markdown("---")
    section("STATISTICS ACADEMY")
    st.caption("Deep-dive interactive lessons on the foundations behind every test.")
    if st.button("🎓 Open Statistics Academy", use_container_width=True):
        st.session_state["show_academy"] = True
        st.rerun()


# ─── Sub-renderers ────────────────────────────────────────────────────────────

def _render_pvalue_meter(p: float, alpha: float):
    """Visual slider showing where p sits relative to alpha."""
    pct = min(p * 100, 100)
    alpha_pct = alpha * 100
    color = "#34D399" if p < alpha else "#FF8FA3"
    st.markdown(f"""
    <div style="margin:1rem 0 0.5rem">
        <div style="font-family:'IBM Plex Mono';font-size:0.72rem;color:#8899b0;margin-bottom:0.4rem;letter-spacing:0.06em">
            P-VALUE METER
        </div>
        <div style="position:relative;height:18px;background:rgba(255,255,255,0.06);border-radius:9px;overflow:hidden">
            <div style="position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;
                        background:linear-gradient(90deg,{color}88,{color});border-radius:9px;
                        transition:width 0.5s ease"></div>
            <div style="position:absolute;left:{alpha_pct:.1f}%;top:0;height:100%;
                        width:2px;background:#FFB347;opacity:0.9"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:0.3rem">
            <span style="font-family:'IBM Plex Mono';font-size:0.7rem;color:{color}">p = {p:.4f}</span>
            <span style="font-family:'IBM Plex Mono';font-size:0.7rem;color:#FFB347">threshold α = {alpha}</span>
            <span style="font-family:'IBM Plex Mono';font-size:0.7rem;color:#8899b0">p = 1.0</span>
        </div>
        <div style="font-size:0.8rem;color:#8899b0;margin-top:0.4rem">
            {"✅ p is below the threshold — result is statistically significant." if p < alpha
             else "❔ p is above the threshold — result is not statistically significant."}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_r2_visual(r2: float):
    """Visualizes R-squared as a filled square."""
    pct = round(r2 * 100, 1)
    explained = min(100, pct)
    unexplained = 100 - explained
    st.markdown(f"""
    <div style="margin-top:1rem">
        <div style="font-family:'IBM Plex Mono';font-size:0.72rem;color:#8899b0;margin-bottom:0.5rem;letter-spacing:0.06em">
            R² = {r2:.3f} — VARIANCE EXPLAINED
        </div>
        <div style="height:30px;border-radius:6px;overflow:hidden;display:flex">
            <div style="width:{explained}%;background:linear-gradient(90deg,#4F8EF7,#A259FF);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'IBM Plex Mono';font-size:0.72rem;color:white;white-space:nowrap">
                {explained:.1f}% explained
            </div>
            <div style="width:{unexplained}%;background:rgba(255,255,255,0.04);
                        display:flex;align-items:center;justify-content:center;
                        font-family:'IBM Plex Mono';font-size:0.72rem;color:#8899b0;white-space:nowrap">
                {unexplained:.1f}% other factors
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_theory_card_ui(card: dict):
    """Renders a rich theory card with tabs."""
    tab_when, tab_hyp, tab_assume, tab_errors, tab_intuition = st.tabs([
        "When to use", "Hypotheses", "Assumptions", "Errors & power", "Intuition"
    ])

    with tab_when:
        st.markdown(f"**{card['when_to_use']}**")
        st.markdown("**Examples:**")
        for ex in card.get("examples", []):
            st.markdown(f"- {ex}")

    with tab_hyp:
        h = card.get("hypotheses", {})
        st.markdown("#### Null Hypothesis (H₀)")
        st.info(h.get("H0", ""))
        st.markdown("#### Alternative Hypothesis (H₁)")
        st.success(h.get("H1", ""))
        st.markdown("The test checks whether the data gives enough evidence to **reject H₀** in favour of H₁.")

    with tab_assume:
        for name, desc in card.get("assumptions", []):
            st.markdown(f"**{name}:** {desc}")
        st.caption("Violations of assumptions can lead to unreliable results. The app checks the most critical ones automatically.")

    with tab_errors:
        errs = card.get("errors", {})
        col1, col2 = st.columns(2)
        with col1:
            st.error(f"**Type I error (false alarm)**\n\n{errs.get('type1', '')}")
        with col2:
            st.warning(f"**Type II error (missed effect)**\n\n{errs.get('type2', '')}")
        st.markdown(effect_label_from_card(card))

    with tab_intuition:
        st.markdown(f"💡 {card.get('intuition', '')}")
        st.markdown(f"\n**Effect size:** {card.get('effect_size', '')}")
        if "followup" in card:
            st.warning(f"⚠️ **Next step:** {card['followup']}")


def effect_label_from_card(card: dict) -> str:
    return f"\n**Effect size:** {card.get('effect_size', '')}"


def _show_landing():
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem 2rem">
        <div style="font-size:4rem;margin-bottom:1rem">🔬</div>
        <h1 style="font-family:'Space Grotesk';font-size:2.2rem;font-weight:700;
                   background:linear-gradient(135deg,#4F8EF7,#A259FF);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:0.5rem">StatSense</h1>
        <p style="color:#8899b0;font-size:1.05rem;max-width:520px;margin:0 auto 2rem">
            Answer three plain-English questions about your data
            and get a clear, jargon-free statistical analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📂", "Load data", "Upload a CSV, enter numbers, or pick a built-in example."),
        ("🧭", "Answer 3 questions", "Tell us what you're trying to find out — in plain English."),
        ("📊", "Read your results", "Get a clear verdict, visualizations, and optional deep dives."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:1.4rem;text-align:center;height:140px">
                <div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>
                <div style="font-weight:600;color:#c8d0e0;font-size:0.95rem;margin-bottom:0.3rem">{title}</div>
                <div style="color:#8899b0;font-size:0.82rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#8899b0;font-size:0.82rem">
        Uses your data to choose between: t-tests · ANOVA · Mann-Whitney · Wilcoxon ·
        Kruskal-Wallis · Chi-square · Pearson/Spearman correlation
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS ACADEMY PAGE
if __name__ == "__main__":
    main()
