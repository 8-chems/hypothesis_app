"""
diagnostic_plots.py — rich data visualization panels:
  boxplots, histograms, QQ plots, scatter+regression, contingency heatmaps, diff plots.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

PALETTE = ["#4F8EF7", "#FF4B6E", "#A259FF", "#FFB347", "#34D399", "#F472B6"]
FONT    = "IBM Plex Mono"
BG_PLOT = "rgba(15,18,30,0.6)"
BG_PAPER= "rgba(0,0,0,0)"
GRID    = "rgba(255,255,255,0.07)"


def _base_layout(title="", height=350):
    return dict(
        title=dict(text=title, font=dict(family=FONT, size=13, color="#c8d0e0"), x=0.01),
        paper_bgcolor=BG_PAPER,
        plot_bgcolor=BG_PLOT,
        height=height,
        margin=dict(l=50, r=20, t=45, b=40),
        font=dict(family=FONT, color="#8899b0"),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        showlegend=True,
        legend=dict(font=dict(family=FONT, size=10, color="#c8d0e0"),
                    bgcolor="rgba(0,0,0,0)"),
    )


# ── 1. Grouped box + strip plot ───────────────────────────────────────────────
def group_box_plot(groups: list[np.ndarray], group_names: list[str],
                   value_label: str = "Value") -> go.Figure:
    fig = go.Figure()
    for i, (g, name) in enumerate(zip(groups, group_names)):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Box(
            y=g, name=name,
            marker_color=color,
            boxmean="sd",
            jitter=0.35, pointpos=0,
            marker=dict(size=4, opacity=0.5),
            line=dict(width=1.5),
        ))
    fig.update_layout(**_base_layout(f"Distribution of {value_label} by group"))
    fig.update_layout(yaxis_title=value_label)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 2. Overlapping histograms ─────────────────────────────────────────────────
def group_histogram(groups: list[np.ndarray], group_names: list[str],
                    value_label: str = "Value") -> go.Figure:
    fig = go.Figure()
    for i, (g, name) in enumerate(zip(groups, group_names)):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Histogram(
            x=g, name=name,
            opacity=0.55,
            marker_color=color,
            nbinsx=20,
            histnorm="probability density",
        ))
        # KDE overlay
        kde_x = np.linspace(g.min() - g.std(), g.max() + g.std(), 200)
        kde = stats.gaussian_kde(g)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde(kde_x), name=f"{name} (density)",
            line=dict(color=color, width=2),
            showlegend=False,
        ))
    fig.update_layout(**_base_layout(f"Shape of data — {value_label}"))
    fig.update_layout(barmode="overlay", xaxis_title=value_label, yaxis_title="Density")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 3. Paired / before-after plot ────────────────────────────────────────────
def paired_change_plot(g1: np.ndarray, g2: np.ndarray,
                       name1: str = "Before", name2: str = "After") -> go.Figure:
    fig = go.Figure()
    n = min(len(g1), len(g2))
    show_lines = n <= 60  # only draw lines if not too many points

    if show_lines:
        for i in range(n):
            color = "#34D399" if g2[i] >= g1[i] else "#FF4B6E"
            fig.add_trace(go.Scatter(
                x=[name1, name2], y=[g1[i], g2[i]],
                mode="lines+markers",
                line=dict(color=color, width=1),
                marker=dict(size=5),
                showlegend=False, hoverinfo="skip",
            ))

    # Mean ± SD bars
    for vals, name, color in [(g1, name1, PALETTE[0]), (g2, name2, PALETTE[1])]:
        fig.add_trace(go.Scatter(
            x=[name], y=[np.mean(vals)],
            error_y=dict(type="data", array=[np.std(vals, ddof=1)], visible=True,
                         color=color, thickness=2, width=8),
            mode="markers",
            marker=dict(size=12, color=color, symbol="diamond"),
            name=f"{name} mean ± SD",
        ))

    fig.update_layout(**_base_layout("Individual changes (each line = one observation)"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 4. QQ plot per group ──────────────────────────────────────────────────────
def qq_plots(groups: list[np.ndarray], group_names: list[str]) -> go.Figure:
    n_groups = len(groups)
    fig = make_subplots(rows=1, cols=n_groups,
                        subplot_titles=[f"Q-Q: {n}" for n in group_names])
    for i, (g, name) in enumerate(zip(groups, group_names)):
        (osm, osr), (slope, intercept, _) = stats.probplot(g)
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(
            x=osm, y=osr, mode="markers",
            marker=dict(size=5, color=color, opacity=0.7),
            name=name, showlegend=False,
        ), row=1, col=i+1)
        # reference line
        x_line = np.array([min(osm), max(osm)])
        fig.add_trace(go.Scatter(
            x=x_line, y=slope * x_line + intercept,
            mode="lines", line=dict(color="#FFB347", width=1.5, dash="dot"),
            showlegend=False,
        ), row=1, col=i+1)

    fig.update_layout(
        title=dict(text="Q-Q plots (points near the line = normal shape)",
                   font=dict(family=FONT, size=13, color="#c8d0e0")),
        paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT,
        height=300, margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family=FONT, color="#8899b0"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 5. Scatter + regression line (correlation) ───────────────────────────────
def scatter_regression(x: np.ndarray, y: np.ndarray,
                       x_label: str = "X", y_label: str = "Y",
                       r: float = 0.0) -> go.Figure:
    fig = go.Figure()

    # Points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=7, color=PALETTE[0], opacity=0.65,
                    line=dict(width=0.5, color="white")),
        name="Data points",
    ))

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b, mode="lines",
        line=dict(color=PALETTE[1], width=2.5),
        name=f"Trend line  (r = {r:.3f})",
    ))

    # 95% CI band
    n = len(x)
    se_band = np.std(y - (m * x + b), ddof=2) * np.sqrt(
        1/n + (x_line - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
    )
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([m*x_line+b + 1.96*se_band,
                          (m*x_line+b - 1.96*se_band)[::-1]]),
        fill="toself", fillcolor="rgba(255,75,110,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% confidence band", hoverinfo="skip",
    ))

    fig.update_layout(**_base_layout(f"Relationship between {x_label} and {y_label}"))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 6. Differences histogram (paired) ────────────────────────────────────────
def differences_histogram(diffs: np.ndarray, name1="Before", name2="After") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs, nbinsx=20,
        marker_color=PALETTE[2], opacity=0.75,
        histnorm="probability density",
        name=f"{name2} − {name1}",
    ))
    # KDE
    if len(diffs) > 3:
        kde = stats.gaussian_kde(diffs)
        xr = np.linspace(diffs.min() - diffs.std(), diffs.max() + diffs.std(), 200)
        fig.add_trace(go.Scatter(x=xr, y=kde(xr), mode="lines",
            line=dict(color=PALETTE[2], width=2), name="Density", showlegend=False))
    # Zero line
    fig.add_vline(x=0, line=dict(color="#FFB347", width=1.5, dash="dash"),
        annotation_text="no change", annotation_font_color="#FFB347", annotation_font_size=10)
    fig.add_vline(x=np.mean(diffs), line=dict(color="#34D399", width=1.5),
        annotation_text=f"mean diff = {np.mean(diffs):.2f}",
        annotation_font_color="#34D399", annotation_font_size=10)

    fig.update_layout(**_base_layout(f"Distribution of changes ({name2} − {name1})"))
    fig.update_layout(xaxis_title=f"Change ({name2} − {name1})", yaxis_title="Density")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 7. Contingency heatmap ────────────────────────────────────────────────────
def contingency_heatmap(ct: pd.DataFrame, title: str = "Observed counts") -> go.Figure:
    pct = ct.div(ct.sum().sum()) * 100
    text = [[f"{int(ct.iloc[i,j])}<br>({pct.iloc[i,j]:.1f}%)"
             for j in range(ct.shape[1])]
            for i in range(ct.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=ct.values,
        x=[str(c) for c in ct.columns],
        y=[str(r) for r in ct.index],
        text=text, texttemplate="%{text}",
        textfont=dict(family=FONT, size=11),
        colorscale=[[0, "rgba(15,18,30,0.8)"], [1, "#4F8EF7"]],
        showscale=True,
    ))
    fig.update_layout(**_base_layout(title))
    fig.update_layout(xaxis_title="", yaxis_title="")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ── 8. Effect size visual ─────────────────────────────────────────────────────
def effect_size_gauge(effect_size: float, label: str, thresholds: dict = None) -> go.Figure:
    """
    A horizontal bar showing where the effect size falls relative to benchmarks.
    thresholds: dict like {"small": 0.2, "medium": 0.5, "large": 0.8}
    """
    if thresholds is None:
        thresholds = {"small": 0.2, "medium": 0.5, "large": 0.8}

    d = abs(effect_size)
    max_val = max(thresholds["large"] * 1.5, d * 1.2, 1.0)

    fig = go.Figure()

    # Background bands
    boundaries = [0] + list(thresholds.values()) + [max_val]
    band_labels = ["negligible", "small", "medium", "large"]
    band_colors = ["rgba(79,142,247,0.08)", "rgba(79,142,247,0.15)",
                   "rgba(162,89,255,0.20)", "rgba(255,75,110,0.25)"]
    for i in range(len(band_labels)):
        x0, x1 = boundaries[i], boundaries[i+1]
        fig.add_vrect(x0=x0, x1=x1, fillcolor=band_colors[i],
                      layer="below", line_width=0,
                      annotation_text=band_labels[i],
                      annotation_position="top",
                      annotation_font=dict(family=FONT, size=9, color="#8899b0"))

    # Actual value marker
    fig.add_trace(go.Scatter(
        x=[d], y=[0], mode="markers+text",
        marker=dict(size=18, color="#FFB347", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        text=[f"{d:.3f}"], textposition="top center",
        textfont=dict(family=FONT, size=12, color="#FFB347"),
        name=label,
    ))

    fig.update_layout(**_base_layout(f"Effect size: {label}", height=180))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, max_val], title_text=label, gridcolor=GRID,
                     tickfont=dict(family=FONT, size=10))
    fig.update_yaxes(visible=False)
    return fig