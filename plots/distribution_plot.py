"""
distribution_plot.py — draws theoretical distributions with shaded rejection zones,
test statistic markers, and critical value lines.
"""
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# ── Color palette ─────────────────────────────────────────────────────────────
C_DIST   = "#4F8EF7"   # curve fill
C_REJECT = "#FF4B6E"   # rejection / p-value region
C_STAT   = "#FFB347"   # test statistic line
C_CRIT   = "#A259FF"   # critical value line
C_BG     = "rgba(0,0,0,0)"
C_GRID   = "rgba(255,255,255,0.07)"
FONT     = "IBM Plex Mono"


def _t_dist_plot(stat: float, df, alpha: float, tail: str) -> go.Figure:
    df_val = df if isinstance(df, (int, float)) and df > 0 else 30
    x = np.linspace(stats.t.ppf(0.0001, df_val), stats.t.ppf(0.9999, df_val), 500)
    y = stats.t.pdf(x, df_val)

    crit_two = stats.t.ppf(1 - alpha / 2, df_val)
    crit_one = stats.t.ppf(1 - alpha, df_val)

    fig = go.Figure()

    # Full curve
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
        line=dict(color=C_DIST, width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.10)",
        name="t distribution", hoverinfo="skip"))

    # Rejection regions
    if tail == "two":
        _shade_region(fig, x, y, x >= crit_two,  C_REJECT, "Rejection zone (right)")
        _shade_region(fig, x, y, x <= -crit_two, C_REJECT, "Rejection zone (left)")
        # Critical lines
        for cv in [crit_two, -crit_two]:
            fig.add_vline(x=cv, line=dict(color=C_CRIT, width=1.5, dash="dash"),
                annotation_text=f"critical = ±{crit_two:.3f}",
                annotation_font_size=10, annotation_font_color=C_CRIT)
    else:
        _shade_region(fig, x, y, x >= crit_one, C_REJECT, "Rejection zone")
        fig.add_vline(x=crit_one, line=dict(color=C_CRIT, width=1.5, dash="dash"),
            annotation_text=f"critical = {crit_one:.3f}",
            annotation_font_size=10, annotation_font_color=C_CRIT)

    # Test statistic
    stat_clipped = max(min(float(stat), x[-1]), x[0])
    fig.add_vline(x=stat_clipped, line=dict(color=C_STAT, width=2.5),
        annotation_text=f"your t = {stat:.3f}",
        annotation_font_size=11, annotation_font_color=C_STAT,
        annotation_position="top right")

    _style_fig(fig, "t distribution", f"df = {int(df_val)}")
    return fig


def _normal_dist_plot(stat: float, alpha: float, tail: str) -> go.Figure:
    x = np.linspace(-4.5, 4.5, 500)
    y = stats.norm.pdf(x)

    crit_two = stats.norm.ppf(1 - alpha / 2)
    crit_one = stats.norm.ppf(1 - alpha)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
        line=dict(color=C_DIST, width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.10)",
        name="Normal distribution", hoverinfo="skip"))

    if tail == "two":
        _shade_region(fig, x, y, x >= crit_two,  C_REJECT, "Rejection zone (right)")
        _shade_region(fig, x, y, x <= -crit_two, C_REJECT, "Rejection zone (left)")
        for cv in [crit_two, -crit_two]:
            fig.add_vline(x=cv, line=dict(color=C_CRIT, width=1.5, dash="dash"),
                annotation_text=f"z = ±{crit_two:.2f}",
                annotation_font_size=10, annotation_font_color=C_CRIT)
    else:
        _shade_region(fig, x, y, x >= crit_one, C_REJECT, "Rejection zone")
        fig.add_vline(x=crit_one, line=dict(color=C_CRIT, width=1.5, dash="dash"),
            annotation_text=f"z = {crit_one:.2f}",
            annotation_font_size=10, annotation_font_color=C_CRIT)

    stat_clipped = max(min(float(stat), 4.4), -4.4)
    fig.add_vline(x=stat_clipped, line=dict(color=C_STAT, width=2.5),
        annotation_text=f"your z = {stat:.3f}",
        annotation_font_size=11, annotation_font_color=C_STAT,
        annotation_position="top right")

    _style_fig(fig, "Normal (z) distribution", "")
    return fig


def _chi2_dist_plot(stat: float, df, alpha: float) -> go.Figure:
    df_val = max(int(df) if df else 1, 1)
    x_max = max(stats.chi2.ppf(0.999, df_val), float(stat) * 1.2, df_val * 3)
    x = np.linspace(0.001, x_max, 600)
    y = stats.chi2.pdf(x, df_val)

    crit = stats.chi2.ppf(1 - alpha, df_val)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
        line=dict(color=C_DIST, width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.10)",
        name="Chi-squared distribution", hoverinfo="skip"))

    _shade_region(fig, x, y, x >= crit, C_REJECT, "Rejection zone")
    fig.add_vline(x=crit, line=dict(color=C_CRIT, width=1.5, dash="dash"),
        annotation_text=f"critical = {crit:.3f}",
        annotation_font_size=10, annotation_font_color=C_CRIT)

    stat_clipped = min(float(stat), x_max * 0.98)
    fig.add_vline(x=stat_clipped, line=dict(color=C_STAT, width=2.5),
        annotation_text=f"your χ² = {stat:.3f}",
        annotation_font_size=11, annotation_font_color=C_STAT,
        annotation_position="top right")

    _style_fig(fig, "Chi-squared distribution", f"df = {df_val}")
    return fig


def _f_dist_plot(stat: float, df, alpha: float) -> go.Figure:
    df1, df2 = (df if isinstance(df, tuple) else (1, 30))
    x_max = max(stats.f.ppf(0.999, df1, df2), float(stat) * 1.3, 8)
    x = np.linspace(0.001, x_max, 600)
    y = stats.f.pdf(x, df1, df2)

    crit = stats.f.ppf(1 - alpha, df1, df2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
        line=dict(color=C_DIST, width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.10)",
        name="F distribution", hoverinfo="skip"))

    _shade_region(fig, x, y, x >= crit, C_REJECT, "Rejection zone")
    fig.add_vline(x=crit, line=dict(color=C_CRIT, width=1.5, dash="dash"),
        annotation_text=f"critical = {crit:.3f}",
        annotation_font_size=10, annotation_font_color=C_CRIT)

    stat_clipped = min(float(stat), x_max * 0.98)
    fig.add_vline(x=stat_clipped, line=dict(color=C_STAT, width=2.5),
        annotation_text=f"your F = {stat:.3f}",
        annotation_font_size=11, annotation_font_color=C_STAT,
        annotation_position="top right")

    _style_fig(fig, "F distribution", f"df₁={df1}, df₂={df2}")
    return fig


def _shade_region(fig, x, y, mask, color, name):
    x_region = x[mask]
    y_region = y[mask]
    if len(x_region) == 0:
        return
    x_fill = np.concatenate([[x_region[0]], x_region, [x_region[-1]]])
    y_fill = np.concatenate([[0], y_region, [0]])
    fig.add_trace(go.Scatter(
        x=x_fill, y=y_fill, fill="toself",
        fillcolor=f"rgba(255,75,110,0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        name=name, hoverinfo="skip",
        showlegend=False,
    ))


def _style_fig(fig, title, subtitle):
    label = f"{title}" + (f"  ·  {subtitle}" if subtitle else "")
    fig.update_layout(
        title=dict(text=label, font=dict(family=FONT, size=14, color="#c8d0e0"), x=0.01),
        paper_bgcolor=C_BG,
        plot_bgcolor="rgba(15,18,30,0.6)",
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False,
                   tickfont=dict(family=FONT, size=10, color="#8899b0"),
                   title_font=dict(family=FONT, size=11, color="#8899b0")),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False,
                   tickfont=dict(family=FONT, size=10, color="#8899b0"),
                   title_text="Probability density",
                   title_font=dict(family=FONT, size=11, color="#8899b0")),
        margin=dict(l=50, r=30, t=50, b=40),
        height=300,
        showlegend=False,
        font=dict(family=FONT),
    )


def make_distribution_plot(result: dict) -> go.Figure:
    """Main entry point — dispatches to the correct distribution plot."""
    dist = result.get("distribution", "t")
    stat = float(result.get("stat", 0))
    alpha = result.get("alpha", 0.05)
    tail = result.get("tail", "two")
    df = result.get("df", 30)

    if dist == "t":
        return _t_dist_plot(stat, df, alpha, tail)
    elif dist == "normal":
        return _normal_dist_plot(stat, alpha, tail)
    elif dist == "chi2":
        return _chi2_dist_plot(stat, df, alpha)
    elif dist == "F":
        return _f_dist_plot(stat, df, alpha)
    else:
        return _normal_dist_plot(stat, alpha, tail)
