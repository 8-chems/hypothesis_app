"""
estimator_tab.py — renders the "Estimator" tab inside the StatSense results page.
Receives the estimator_context dict from estimator_guide.get_estimator_context().
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats as sci_stats


# ── Palette ───────────────────────────────────────────────────────────────────
C_BLUE   = "#4F8EF7"
C_PURPLE = "#A259FF"
C_RED    = "#FF4B6E"
C_AMBER  = "#FFB347"
C_GREEN  = "#34D399"
FONT     = "IBM Plex Mono"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_estimator_tab(ctx: dict):
    """
    ctx = output of estimator_guide.get_estimator_context()
    """
    profile   = ctx["profile"]
    regime    = ctx["regime"]
    narrative = ctx["narrative"]
    subs      = ctx["substitutions"]
    switched  = ctx["switched"]
    actual    = ctx["actual_test"]

    # ── 1. Header ─────────────────────────────────────────────────────────────
    _render_header(ctx)
    st.markdown("")

    # ── 2. Data regime panel ──────────────────────────────────────────────────
    _render_regime_panel(regime, narrative)
    st.markdown("")

    # ── 3. Estimator profile ──────────────────────────────────────────────────
    _render_estimator_profile(profile, switched)
    st.markdown("")

    # ── 4. Efficiency chart ───────────────────────────────────────────────────
    _render_efficiency_chart(regime, actual)
    st.markdown("")

    # ── 5. Substitution rules ─────────────────────────────────────────────────
    if subs:
        _render_substitutions(subs, regime)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_header(ctx: dict):
    switched = ctx["switched"]
    actual   = ctx["actual_test"]
    test_id  = ctx["test_id"]
    profile  = ctx["profile"]

    st.markdown("""
    <div style="margin-bottom:.5rem">
        <div style="font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;
                    color:#4F8EF7;text-transform:uppercase;margin-bottom:.4rem">
            Why this estimator — for your data
        </div>
    </div>
    """, unsafe_allow_html=True)

    if switched:
        st.info(
            f"**Auto-switched:** Your data triggered the non-parametric fallback. "
            f"The estimator shown below is for **{actual.replace('_',' ').title()}** — "
            f"the safer choice given your data's shape or sample size.",
            icon="🔄"
        )


def _render_regime_panel(regime: dict, narrative: dict):
    st.markdown("""
    <div style="font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;
                color:#4F8EF7;text-transform:uppercase;margin-bottom:.8rem">
        Your data regime
    </div>
    """, unsafe_allow_html=True)

    # Regime badges
    n = regime["n_min"]
    size_color  = C_GREEN if regime["is_large"] else C_AMBER if regime["is_medium"] else C_RED
    size_label  = f"Large (n={n})" if regime["is_large"] else f"Medium (n={n})" if regime["is_medium"] else f"Small (n={n})"
    shape_label = "Skewed" if regime["is_skewed"] else "Binary" if regime["is_binary"] else "Count data" if regime["is_count"] else "Continuous"
    shape_color = C_AMBER if regime["is_skewed"] else C_BLUE
    norm_label  = "Normal ✓" if regime["normality_ok"] else "Non-Normal ⚠"
    norm_color  = C_GREEN if regime["normality_ok"] else C_RED

    badges = [
        (size_label,  size_color,  "Sample size"),
        (shape_label, shape_color, "Data shape"),
        (norm_label,  norm_color,  "Normality"),
        ("Paired" if regime["is_paired"] else "Sequential" if regime["is_sequential"] else "Independent",
         C_PURPLE, "Observation type"),
    ]
    if regime["is_binary"]:
        rate_color = C_RED if regime["is_rare"] else C_AMBER if regime["event_rate"] < 0.2 else C_GREEN
        badges.append((f"Rate ≈ {regime['event_rate']*100:.1f}%", rate_color, "Event rate"))
    if regime["is_count"]:
        di_color = C_RED if regime["overdispersed"] else C_GREEN
        badges.append((f"Dispersion {regime['dispersion_index']:.2f}", di_color, "Var/Mean"))

    badge_html = "".join(f"""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
                border-left:3px solid {c};border-radius:8px;padding:.6rem .9rem;min-width:120px">
        <div style="font-family:'IBM Plex Mono';font-size:.62rem;color:#8899b0;
                    letter-spacing:.06em;margin-bottom:.2rem">{lbl.upper()}</div>
        <div style="font-size:.85rem;font-weight:600;color:{c}">{val}</div>
    </div>
    """ for val, c, lbl in badges)

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:.7rem;margin-bottom:1.2rem">
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

    # Narrative stories
    for key, icon in [("size_story", "📏"), ("frequency_story", "📊"), ("obs_story", "🔗"), ("estimator_choice_story", "🎯")]:
        story = narrative.get(key, "")
        if story:
            _concept_box(story, icon)

    # Warnings
    for w in narrative.get("warning_story", []):
        st.warning(w)


def _render_estimator_profile(profile: dict, switched: bool):
    st.markdown("""
    <div style="font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;
                color:#4F8EF7;text-transform:uppercase;margin-bottom:.8rem">
        Estimator profile
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(f"""
        <div style="background:rgba(79,142,247,.06);border:1px solid rgba(79,142,247,.2);
                    border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.9rem">
            <div style="font-family:'IBM Plex Mono';font-size:.68rem;color:#4F8EF7;
                        letter-spacing:.08em;margin-bottom:.4rem">ESTIMATOR</div>
            <div style="font-size:1.05rem;font-weight:600;color:#e8eef8;margin-bottom:.6rem">
                {profile['estimator']}
            </div>
            <div style="font-family:'IBM Plex Mono';font-size:.82rem;color:{C_AMBER};
                        background:rgba(255,179,71,.07);border:1px solid rgba(255,179,71,.15);
                        border-radius:6px;padding:.4rem .7rem;margin-bottom:.7rem">
                {profile['formula']}
            </div>
            <div style="font-size:.83rem;color:#c8d0e0;line-height:1.65">{profile['why']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Caveats
        if profile.get("caveats"):
            st.markdown("""
            <div style="font-family:'IBM Plex Mono';font-size:.65rem;color:#8899b0;
                        letter-spacing:.08em;margin-bottom:.5rem">WATCH OUT FOR</div>
            """, unsafe_allow_html=True)
            for c in profile["caveats"]:
                st.markdown(f"""
                <div style="display:flex;gap:.6rem;align-items:flex-start;
                            margin-bottom:.4rem;font-size:.82rem;color:#8899b0">
                    <span style="color:{C_AMBER};flex-shrink:0">⚠</span>
                    <span>{c}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_r:
        # CI and SE boxes
        for label, val, color in [
            ("Confidence Interval", profile.get("ci", "—"), C_BLUE),
            ("Standard Error",      profile.get("se", "—"), C_PURPLE),
            ("Key property",        profile.get("key_property", "—"), C_GREEN),
        ]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.08);
                        border-left:3px solid {color};border-radius:8px;
                        padding:.7rem 1rem;margin-bottom:.7rem">
                <div style="font-family:'IBM Plex Mono';font-size:.62rem;color:#8899b0;
                            letter-spacing:.08em;margin-bottom:.3rem">{label.upper()}</div>
                <div style="font-family:'IBM Plex Mono';font-size:.78rem;color:{color};
                            line-height:1.5">{val}</div>
            </div>
            """, unsafe_allow_html=True)

        # Family badge
        st.markdown(f"""
        <div style="background:rgba(162,89,255,.08);border:1px solid rgba(162,89,255,.2);
                    border-radius:8px;padding:.7rem 1rem">
            <div style="font-family:'IBM Plex Mono';font-size:.62rem;color:#8899b0;
                        letter-spacing:.08em;margin-bottom:.3rem">ESTIMATOR FAMILY</div>
            <div style="font-size:.82rem;color:{C_PURPLE}">{profile.get('family','—')}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_efficiency_chart(regime: dict, test_id: str):
    """
    Shows relative efficiency of the current estimator vs alternatives
    as a function of sample size.
    """
    st.markdown("""
    <div style="font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;
                color:#4F8EF7;text-transform:uppercase;margin-bottom:.8rem">
        Efficiency vs sample size — why n matters for estimator choice
    </div>
    """, unsafe_allow_html=True)

    ns = np.array([2, 5, 8, 10, 15, 20, 30, 50, 100, 200])

    # Parametric mean efficiency (approaches 100% as n grows and CLT kicks in)
    is_skewed = regime.get("is_skewed", False)
    if is_skewed:
        param_eff = np.clip(20 + ns * 1.8, 0, 97)
    else:
        param_eff = np.clip(15 + ns * 2.5, 0, 100)

    # Non-parametric (rank-based) — flat 95.5% of parametric for Normal, better for skewed
    if is_skewed:
        nonparam_eff = np.clip(65 + ns * 0.3, 0, 98)
    else:
        nonparam_eff = np.full(len(ns), 95.5)

    # Exact methods (always reliable but constant cost)
    exact_eff = np.full(len(ns), 90.0)

    # Bootstrap (valid only above n=15 or so)
    bootstrap_eff = np.where(ns >= 15, np.clip(60 + ns * 0.5, 0, 95), ns * 3.0)

    # Mark current n
    n_cur = regime["n_min"]

    fig = go.Figure()

    traces = [
        (param_eff,     C_BLUE,   "Parametric (mean-based)",   "solid"),
        (nonparam_eff,  C_PURPLE, "Non-parametric (rank-based)", "solid"),
        (bootstrap_eff, C_AMBER,  "Bootstrap CI",               "dash"),
        (exact_eff,     C_RED,    "Exact methods",              "dot"),
    ]
    for y, color, name, dash in traces:
        fig.add_trace(go.Scatter(
            x=ns, y=y, mode="lines+markers",
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=5),
            name=name,
        ))

    # Mark current n with a vertical line
    if n_cur <= 200:
        fig.add_vline(
            x=n_cur,
            line=dict(color=C_GREEN, width=2, dash="dot"),
            annotation_text=f"your n = {n_cur}",
            annotation_font=dict(color=C_GREEN, size=10, family=FONT),
            annotation_position="top right",
        )

    # CLT threshold
    fig.add_vline(
        x=30,
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        annotation_text="CLT ≥ 30",
        annotation_font=dict(color="#8899b0", size=9, family=FONT),
        annotation_position="top left",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,18,30,0.6)",
        height=280,
        margin=dict(l=50, r=20, t=35, b=40),
        font=dict(family=FONT, color="#8899b0"),
        legend=dict(
            font=dict(family=FONT, size=10, color="#c8d0e0"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        xaxis=dict(title="Sample size (n)", gridcolor="rgba(255,255,255,.06)", zeroline=False,
                   tickfont=dict(size=9)),
        yaxis=dict(title="Appropriateness / efficiency (%)", gridcolor="rgba(255,255,255,.06)",
                   zeroline=False, range=[0, 105], tickfont=dict(size=9),
                   ticksuffix="%"),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Parametric estimators gain reliability as n grows and the CLT applies. "
        "Non-parametric methods are robust at any n but sacrifice a small amount of efficiency when "
        "data is truly Normal. Exact methods are always valid but computationally heavier."
    )


def _render_substitutions(subs: list, regime: dict):
    st.markdown("""
    <div style="font-family:'IBM Plex Mono';font-size:.68rem;letter-spacing:.12em;
                color:#4F8EF7;text-transform:uppercase;margin-bottom:.8rem">
        When can one distribution's estimator stand in for another?
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "These are the substitution rules relevant to your test. "
        "Each one states the exact condition that makes the swap valid — "
        "and where it breaks down.",
        unsafe_allow_html=False,
    )
    st.markdown("")

    for sub in subs:
        # Check if condition is currently met (heuristic)
        met = _check_condition_met(sub, regime)
        border_color = C_GREEN if met else "rgba(255,255,255,0.1)"
        status_icon  = "✅ Condition met for your data" if met else "ℹ️ Condition not met — substitution not valid here"
        status_color = C_GREEN if met else "#8899b0"

        st.markdown(f"""
        <div style="background:rgba(255,255,255,.02);border:1px solid {border_color};
                    border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1rem">

            <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;flex-wrap:wrap">
                <span style="background:rgba(255,75,110,.1);border:1px solid rgba(255,75,110,.25);
                              border-radius:5px;padding:.2rem .6rem;font-family:'IBM Plex Mono';
                              font-size:.75rem;color:{C_RED}">{sub['from_dist']}</span>
                <span style="color:#8899b0;font-size:.85rem">→</span>
                <span style="background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.25);
                              border-radius:5px;padding:.2rem .6rem;font-family:'IBM Plex Mono';
                              font-size:.75rem;color:{C_BLUE}">{sub['to_dist']}</span>
                <span style="font-family:'IBM Plex Mono';font-size:.72rem;color:{status_color};
                              margin-left:auto">{status_icon}</span>
            </div>

            <div style="font-family:'IBM Plex Mono';font-size:.72rem;color:{C_AMBER};
                        background:rgba(255,179,71,.06);border:1px solid rgba(255,179,71,.15);
                        border-radius:6px;padding:.4rem .7rem;margin-bottom:.7rem">
                Condition: {sub['condition']}
            </div>

            <div style="font-size:.83rem;color:#c8d0e0;line-height:1.65;margin-bottom:.6rem">
                {sub['why']}
            </div>

            <div style="display:flex;gap:1rem;flex-wrap:wrap">
                <div style="flex:1;min-width:180px">
                    <div style="font-family:'IBM Plex Mono';font-size:.62rem;color:{C_RED};
                                letter-spacing:.06em;margin-bottom:.2rem">BREAKS WHEN</div>
                    <div style="font-size:.78rem;color:#8899b0">{sub['breaks_when']}</div>
                </div>
                <div style="flex:1;min-width:180px">
                    <div style="font-family:'IBM Plex Mono';font-size:.62rem;color:{C_GREEN};
                                letter-spacing:.06em;margin-bottom:.2rem">PRACTICAL TIP</div>
                    <div style="font-size:.78rem;color:#8899b0">{sub['practical_tip']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _concept_box(text: str, icon: str = "📌"):
    st.markdown(f"""
    <div style="background:rgba(79,142,247,.05);border:1px solid rgba(79,142,247,.15);
                border-left:3px solid #4F8EF7;border-radius:8px;
                padding:.8rem 1.1rem;margin-bottom:.7rem;
                font-size:.85rem;color:#c8d0e0;line-height:1.65">
        <span style="margin-right:.4rem">{icon}</span>{text}
    </div>
    """, unsafe_allow_html=True)


def _check_condition_met(sub: dict, regime: dict) -> bool:
    """Heuristic check of whether a substitution condition holds for the current data."""
    from_dist = sub["from_dist"]
    n = regime.get("n_min", 0)
    p = regime.get("event_rate", 0.5)
    di = regime.get("dispersion_index", 1.0)

    if "Binomial" in from_dist and "Normal" in sub["to_dist"]:
        return n * p >= 5 and n * (1 - p) >= 5
    if "Binomial" in from_dist and "Poisson" in sub["to_dist"]:
        return n >= 20 and p < 0.05
    if "Poisson" in from_dist and "Normal" in sub["to_dist"]:
        mu = regime.get("event_rate", 0) * n
        return mu >= 10
    if "t(df)" in from_dist:
        return n >= 30
    if "Chi-squared" in from_dist:
        return n >= 30
    if "Negative Binomial" in from_dist:
        return abs(di - 1.0) < 0.5
    if "Hypergeometric" in from_dist:
        return True  # assume large population
    if "Mann-Whitney" in from_dist:
        return regime.get("normality_ok", False) and n >= 30
    if "Wilcoxon" in from_dist:
        return regime.get("normality_ok", False) and n >= 15
    return False
