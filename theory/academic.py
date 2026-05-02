"""
academic.py — Interactive academic statistics visualizations.
Covers: descriptive stats, distributions & their parameters,
estimators, variance/bias, CLT, sampling distributions.
Every figure is standalone — no Streamlit calls here, just go.Figure returns.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ── Shared style ──────────────────────────────────────────────────────────────
PALETTE = ["#4F8EF7", "#FF4B6E", "#A259FF", "#FFB347", "#34D399", "#F472B6", "#38BDF8"]
FONT    = "IBM Plex Mono"
BG_P    = "rgba(0,0,0,0)"
BG_PLOT = "rgba(15,18,30,0.65)"
GRID    = "rgba(255,255,255,0.06)"
TEXT    = "#c8d0e0"
MUTED   = "#8899b0"


def _L(title="", height=380, margin=None):
    """Base layout dict. Never includes xaxis/yaxis — callers set those explicitly."""
    m = margin or dict(l=55, r=25, t=48, b=42)
    return dict(
        title=dict(text=title, font=dict(family=FONT, size=13, color=TEXT), x=0.01),
        paper_bgcolor=BG_P, plot_bgcolor=BG_PLOT, height=height,
        margin=m, font=dict(family=FONT, color=MUTED),
        legend=dict(font=dict(family=FONT, size=10, color=TEXT), bgcolor="rgba(0,0,0,0)"),
    )


def _axis():
    """Standard axis style — use in update_xaxes/update_yaxes calls."""
    return dict(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DESCRIPTIVE STATISTICS — mean, median, variance, std on a real dataset
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats_visual(data: np.ndarray, label: str = "Value") -> go.Figure:
    """
    Histogram + KDE with mean, median, ±1 SD, ±2 SD overlaid.
    """
    mu   = np.mean(data)
    med  = np.median(data)
    sd   = np.std(data, ddof=1)
    var  = sd ** 2

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data, nbinsx=25, histnorm="probability density",
        marker_color="rgba(79,142,247,0.35)",
        marker_line=dict(color="rgba(79,142,247,0.6)", width=0.5),
        name="Data", showlegend=False,
    ))

    # KDE
    kde = stats.gaussian_kde(data)
    xr  = np.linspace(data.min() - sd, data.max() + sd, 400)
    fig.add_trace(go.Scatter(
        x=xr, y=kde(xr), mode="lines",
        line=dict(color=PALETTE[0], width=2.5),
        name="Density (KDE)", showlegend=False,
    ))

    # SD bands
    for k, alpha_fill, lbl in [(1, 0.18, "±1 SD  (≈68%)"), (2, 0.09, "±2 SD  (≈95%)")]:
        fig.add_vrect(x0=mu - k*sd, x1=mu + k*sd,
                      fillcolor=f"rgba(79,142,247,{alpha_fill})",
                      layer="below", line_width=0,
                      annotation_text=lbl,
                      annotation_position="top left",
                      annotation_font=dict(family=FONT, size=9, color=PALETTE[0]))

    # Mean line
    fig.add_vline(x=mu, line=dict(color=PALETTE[1], width=2),
                  annotation_text=f"mean = {mu:.2f}",
                  annotation_font=dict(family=FONT, size=10, color=PALETTE[1]),
                  annotation_position="top right")
    # Median line
    fig.add_vline(x=med, line=dict(color=PALETTE[2], width=1.5, dash="dot"),
                  annotation_text=f"median = {med:.2f}",
                  annotation_font=dict(family=FONT, size=10, color=PALETTE[2]),
                  annotation_position="bottom right")

    fig.update_layout(
        **_L(f"Descriptive statistics of {label}  ·  "
             f"mean={mu:.2f}  sd={sd:.2f}  var={var:.2f}  n={len(data)}"),
        xaxis_title=label, yaxis_title="Density",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DISTRIBUTION FAMILY EXPLORER — show shape vs parameters
# ═══════════════════════════════════════════════════════════════════════════════

def normal_parameter_explorer() -> go.Figure:
    """4 Normal curves with different μ and σ — shows how parameters shift/scale shape."""
    configs = [
        (0,  1,   PALETTE[0], "μ=0, σ=1  (standard)"),
        (2,  1,   PALETTE[1], "μ=2, σ=1  (shift right)"),
        (0,  0.5, PALETTE[2], "μ=0, σ=0.5  (narrower)"),
        (0,  2,   PALETTE[3], "μ=0, σ=2  (wider)"),
    ]
    fig = go.Figure()
    x = np.linspace(-7, 9, 600)
    for mu, sigma, color, name in configs:
        y = stats.norm.pdf(x, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=2.2),
            fill="tozeroy", fillcolor=color.replace(")", ",0.06)").replace("rgb", "rgba") if "rgb" in color else f"rgba(79,142,247,0.04)",
            name=name,
        ))
    fig.update_layout(
        **_L("Normal distribution  ·  effect of μ (location) and σ (scale)"),
        xaxis_title="x", yaxis_title="f(x)  —  probability density",
    )
    _add_formula(fig, "f(x) = (1/σ√2π) · exp(−(x−μ)²/2σ²)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


def t_distribution_df_explorer() -> go.Figure:
    """t distributions with varying df — shows convergence to Normal."""
    configs = [
        (1,  PALETTE[0], "df=1  (very heavy tails)"),
        (3,  PALETTE[1], "df=3"),
        (10, PALETTE[2], "df=10"),
        (30, PALETTE[3], "df=30  (≈ Normal)"),
    ]
    fig = go.Figure()
    x = np.linspace(-5, 5, 500)
    for df, color, name in configs:
        y = stats.t.pdf(x, df)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                  line=dict(color=color, width=2.2), name=name))
    # Reference Normal
    fig.add_trace(go.Scatter(
        x=x, y=stats.norm.pdf(x), mode="lines",
        line=dict(color="#ffffff", width=1.5, dash="dot"),
        name="Normal(0,1)  (limit)",
    ))
    fig.update_layout(
        **_L("Student's t distribution  ·  effect of degrees of freedom (df)"),
        xaxis_title="t", yaxis_title="f(t)",
    )
    _add_formula(fig, "f(t) = Γ((ν+1)/2) / (√νπ · Γ(ν/2)) · (1 + t²/ν)^(−(ν+1)/2)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


def chi2_df_explorer() -> go.Figure:
    """Chi-squared for various df."""
    configs = [
        (1,  PALETTE[0], "df=1"),
        (2,  PALETTE[1], "df=2"),
        (5,  PALETTE[2], "df=5"),
        (10, PALETTE[3], "df=10"),
        (20, PALETTE[4], "df=20"),
    ]
    fig = go.Figure()
    x = np.linspace(0.01, 40, 600)
    for df, color, name in configs:
        y = stats.chi2.pdf(x, df)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                  line=dict(color=color, width=2.2), name=name))
    fig.update_layout(
        **_L("Chi-squared distribution  ·  effect of degrees of freedom"),
        xaxis_title="χ²", yaxis_title="f(χ²)",
        xaxis_range=[0, 40],
    )
    _add_formula(fig, "f(x) = x^(k/2−1) · e^(−x/2) / (2^(k/2) · Γ(k/2))   where k = df")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


def f_distribution_explorer() -> go.Figure:
    """F distribution for various (df1, df2) pairs."""
    configs = [
        (1,  1,  PALETTE[0], "df₁=1, df₂=1"),
        (2,  5,  PALETTE[1], "df₁=2, df₂=5"),
        (5,  10, PALETTE[2], "df₁=5, df₂=10"),
        (10, 30, PALETTE[3], "df₁=10, df₂=30"),
    ]
    fig = go.Figure()
    x = np.linspace(0.01, 8, 600)
    for d1, d2, color, name in configs:
        y = stats.f.pdf(x, d1, d2)
        y = np.clip(y, 0, 4)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                  line=dict(color=color, width=2.2), name=name))
    fig.update_layout(
        **_L("F distribution  ·  ratio of two chi-squared variables"),
        xaxis_title="F", yaxis_title="f(F)",
        yaxis_range=[0, 4],
    )
    _add_formula(fig, "F = (χ²₁/df₁) / (χ²₂/df₂)   — used in ANOVA as the signal-to-noise ratio")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


def binomial_explorer() -> go.Figure:
    """Binomial PMF for different n and p."""
    configs = [
        (10, 0.5,  PALETTE[0], "n=10, p=0.5"),
        (10, 0.2,  PALETTE[1], "n=10, p=0.2  (skewed)"),
        (30, 0.5,  PALETTE[2], "n=30, p=0.5  (more symmetric)"),
        (30, 0.1,  PALETTE[3], "n=30, p=0.1"),
    ]
    fig = go.Figure()
    for n, p, color, name in configs:
        k = np.arange(0, n + 1)
        pmf = stats.binom.pmf(k, n, p)
        fig.add_trace(go.Bar(x=k, y=pmf, name=name,
                              marker_color=color, opacity=0.6,
                              marker_line=dict(color=color, width=1)))
    fig.update_layout(
        **_L("Binomial distribution  ·  effect of n (trials) and p (success probability)"),
        barmode="overlay", xaxis_title="k  (number of successes)", yaxis_title="P(X = k)",
    )
    _add_formula(fig, "P(X=k) = C(n,k) · pᵏ · (1−p)^(n−k)   mean = np,  variance = np(1−p)")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MEAN vs VARIANCE — interactive decomposition on actual data
# ═══════════════════════════════════════════════════════════════════════════════

def variance_decomposition(data: np.ndarray, label: str = "Value") -> go.Figure:
    """
    Shows each observation's squared deviation from the mean — builds intuition
    for why variance = average squared deviation.
    """
    mu  = np.mean(data)
    sd  = np.std(data, ddof=1)
    var = sd ** 2
    n   = len(data)

    # Sample up to 40 points to keep readable
    rng  = np.random.default_rng(0)
    idx  = rng.choice(n, min(40, n), replace=False)
    samp = data[idx]
    devs = samp - mu

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Deviations from the mean", "Squared deviations (= variance building blocks)"
    ])

    # Left: raw deviations as lollipops
    for i, (xi, di) in enumerate(zip(samp, devs)):
        color = PALETTE[0] if di >= 0 else PALETTE[1]
        fig.add_trace(go.Scatter(
            x=[i, i], y=[mu, xi], mode="lines",
            line=dict(color=color, width=1.5), showlegend=False,
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(samp))), y=samp, mode="markers",
        marker=dict(size=7, color=PALETTE[0]), name="Observations", showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=mu, line=dict(color=PALETTE[1], width=2, dash="dash"),
                  annotation_text=f"mean = {mu:.2f}", row=1, col=1)

    # Right: squared deviations as bars
    sq_devs = devs ** 2
    colors  = [PALETTE[0] if d >= 0 else PALETTE[1] for d in devs]
    fig.add_trace(go.Bar(
        x=list(range(len(samp))), y=sq_devs,
        marker_color=colors, opacity=0.7,
        name="(xᵢ − x̄)²", showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=var, line=dict(color="#FFB347", width=2, dash="dash"),
                  annotation_text=f"variance = {var:.2f}  (mean of bars)",
                  annotation_font=dict(color="#FFB347"), row=1, col=2)

    fig.update_layout(
        **_L(f"Variance decomposition  ·  s² = Σ(xᵢ−x̄)²/(n−1) = {var:.3f}", height=400),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ESTIMATORS — bias, variance, MSE illustrated
# ═══════════════════════════════════════════════════════════════════════════════

def estimator_bias_variance(true_mean: float = 5.0, true_std: float = 2.0,
                             n_samples: int = 30, n_reps: int = 500) -> go.Figure:
    """
    Simulates repeated sampling and shows how x̄ and s² behave as estimators.
    Compares biased vs unbiased variance estimator.
    """
    rng = np.random.default_rng(42)
    means, var_biased, var_unbiased = [], [], []
    for _ in range(n_reps):
        samp = rng.normal(true_mean, true_std, n_samples)
        means.append(np.mean(samp))
        var_biased.append(np.var(samp, ddof=0))
        var_unbiased.append(np.var(samp, ddof=1))

    means       = np.array(means)
    var_biased  = np.array(var_biased)
    var_unbiased= np.array(var_unbiased)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"x̄ as estimator of μ={true_mean}  (n={n_samples})",
            f"s² estimators of σ²={true_std**2:.1f}  —  biased vs unbiased",
        ]
    )

    # Left: sampling distribution of x̄
    fig.add_trace(go.Histogram(
        x=means, nbinsx=30, histnorm="probability density",
        marker_color="rgba(79,142,247,0.55)", name="x̄ distribution",
        showlegend=True,
    ), row=1, col=1)
    sem = true_std / np.sqrt(n_samples)
    xr  = np.linspace(means.min(), means.max(), 300)
    fig.add_trace(go.Scatter(
        x=xr, y=stats.norm.pdf(xr, true_mean, sem), mode="lines",
        line=dict(color=PALETTE[0], width=2.5),
        name=f"N(μ, σ/√n)  SE={sem:.3f}", showlegend=True,
    ), row=1, col=1)
    fig.add_vline(x=true_mean, line=dict(color=PALETTE[1], width=2, dash="dash"),
                  annotation_text="true μ", row=1, col=1)
    fig.add_vline(x=np.mean(means), line=dict(color=PALETTE[4], width=1.5, dash="dot"),
                  annotation_text=f"E[x̄]={np.mean(means):.3f}", row=1, col=1)

    # Right: biased vs unbiased variance
    for arr, color, name in [
        (var_biased,   PALETTE[1], f"s²_biased  (÷n)   mean={np.mean(var_biased):.2f}"),
        (var_unbiased, PALETTE[4], f"s²_unbiased (÷n−1) mean={np.mean(var_unbiased):.2f}"),
    ]:
        fig.add_trace(go.Histogram(
            x=arr, nbinsx=30, histnorm="probability density",
            marker_color=color, opacity=0.55, name=name,
        ), row=1, col=2)
    fig.add_vline(x=true_std**2, line=dict(color="white", width=2, dash="dash"),
                  annotation_text=f"true σ²={true_std**2:.1f}", row=1, col=2)

    fig.update_layout(
        **_L(f"Estimator properties  ·  {n_reps} simulated samples of n={n_samples}", height=400),
        barmode="overlay",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CENTRAL LIMIT THEOREM — convergence to Normal as n grows
# ═══════════════════════════════════════════════════════════════════════════════

def central_limit_theorem(population: str = "exponential", n_reps: int = 2000) -> go.Figure:
    """
    Shows sample mean distributions for n = 1, 5, 20, 50 from a non-normal population.
    """
    rng = np.random.default_rng(7)

    def draw(n):
        if population == "exponential":
            return rng.exponential(1.0, (n_reps, n)).mean(axis=1)
        elif population == "uniform":
            return rng.uniform(0, 1, (n_reps, n)).mean(axis=1)
        else:  # bimodal
            a = rng.normal(0, 0.5, (n_reps, n))
            b = rng.normal(3, 0.5, (n_reps, n))
            mask = rng.random((n_reps, n)) > 0.5
            return np.where(mask, a, b).mean(axis=1)

    ns     = [1, 5, 20, 50]
    colors = [PALETTE[1], PALETTE[3], PALETTE[2], PALETTE[0]]

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=[f"n = {n}" for n in ns])

    for col_i, (n, color) in enumerate(zip(ns, colors), start=1):
        sample_means = draw(n)
        mu_sm = np.mean(sample_means)
        sd_sm = np.std(sample_means, ddof=1)

        fig.add_trace(go.Histogram(
            x=sample_means, nbinsx=40, histnorm="probability density",
            marker_color=color, opacity=0.6,
            name=f"n={n}", showlegend=True,
        ), row=1, col=col_i)

        # Normal overlay
        xr = np.linspace(sample_means.min(), sample_means.max(), 200)
        fig.add_trace(go.Scatter(
            x=xr, y=stats.norm.pdf(xr, mu_sm, sd_sm), mode="lines",
            line=dict(color="white", width=1.5),
            showlegend=False,
        ), row=1, col=col_i)

    pop_names = {"exponential": "Exponential(λ=1) population",
                 "uniform": "Uniform(0,1) population",
                 "bimodal": "Bimodal population"}
    fig.update_layout(
        **_L(f"Central Limit Theorem  ·  {pop_names.get(population, population)}  ·  {n_reps} samples each", height=380),
        barmode="overlay",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CONFIDENCE INTERVALS — what "95% confident" actually means
# ═══════════════════════════════════════════════════════════════════════════════

def confidence_interval_simulation(true_mean: float = 5.0, true_std: float = 2.0,
                                     n: int = 30, n_intervals: int = 50,
                                     confidence: float = 0.95) -> go.Figure:
    """
    Simulates 50 CIs — shows that ~95% contain the true mean.
    """
    rng   = np.random.default_rng(99)
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    lowers, uppers, means_list, contains = [], [], [], []
    for _ in range(n_intervals):
        samp = rng.normal(true_mean, true_std, n)
        xbar = np.mean(samp)
        se   = np.std(samp, ddof=1) / np.sqrt(n)
        lo, hi = xbar - t_crit * se, xbar + t_crit * se
        lowers.append(lo); uppers.append(hi); means_list.append(xbar)
        contains.append(lo <= true_mean <= hi)

    hit  = sum(contains)
    miss = n_intervals - hit

    fig = go.Figure()

    for i, (lo, hi, xbar, ok) in enumerate(zip(lowers, uppers, means_list, contains)):
        color = PALETTE[4] if ok else PALETTE[1]
        # CI bar
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i], mode="lines",
            line=dict(color=color, width=1.5), showlegend=False,
        ))
        # Point estimate
        fig.add_trace(go.Scatter(
            x=[xbar], y=[i], mode="markers",
            marker=dict(size=6, color=color), showlegend=False,
        ))

    # True mean line
    fig.add_vline(x=true_mean, line=dict(color="#FFB347", width=2.5),
                  annotation_text=f"true μ = {true_mean}",
                  annotation_font=dict(color="#FFB347", size=11, family=FONT))

    # Legend proxies
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=PALETTE[4], width=2), name=f"Contains μ  ({hit}/{n_intervals})"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=PALETTE[1], width=2), name=f"Misses μ  ({miss}/{n_intervals})"))

    fig.update_layout(
        **_L(f"{int(confidence*100)}% Confidence intervals  ·  {hit}/{n_intervals} contain the true mean  (expected ≈{int(confidence*100)}%)", height=700),
        xaxis_title="Value",
        yaxis_title="Sample #",
    )
    fig.update_xaxes(**_axis())
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=9))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  STANDARD ERROR — how SE shrinks with n
# ═══════════════════════════════════════════════════════════════════════════════

def standard_error_vs_n(true_std: float = 10.0) -> go.Figure:
    """Shows SE = σ/√n as a function of sample size."""
    ns  = np.arange(2, 201)
    se  = true_std / np.sqrt(ns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=se, mode="lines",
        line=dict(color=PALETTE[0], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        name=f"SE = σ/√n  (σ={true_std})",
    ))

    # Annotations at key n values
    for n_mark in [5, 10, 30, 100]:
        se_mark = true_std / np.sqrt(n_mark)
        fig.add_trace(go.Scatter(
            x=[n_mark], y=[se_mark], mode="markers+text",
            marker=dict(size=10, color=PALETTE[1]),
            text=[f"n={n_mark}<br>SE={se_mark:.2f}"],
            textposition="top right",
            textfont=dict(family=FONT, size=9, color=PALETTE[1]),
            showlegend=False,
        ))

    fig.update_layout(
        **_L(f"Standard Error vs sample size  ·  SE = σ/√n  (σ={true_std})"),
        xaxis_title="Sample size (n)",
        yaxis_title="Standard Error",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  P-VALUE INTUITION — what p really means (simulation)
# ═══════════════════════════════════════════════════════════════════════════════

def pvalue_intuition(observed_t: float = 2.3, df: int = 28) -> go.Figure:
    """
    Shows the t-distribution and shades the area that represents the p-value,
    with annotation explaining what it means in probability terms.
    """
    x   = np.linspace(-5, 5, 600)
    y   = stats.t.pdf(x, df)
    p   = 2 * stats.t.sf(abs(observed_t), df)  # two-tailed

    fig = go.Figure()

    # Full curve
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=PALETTE[0], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        name=f"t distribution (df={df})", showlegend=False,
    ))

    # P-value region (both tails)
    for mask_fn in [lambda xi: xi >= abs(observed_t), lambda xi: xi <= -abs(observed_t)]:
        mask = mask_fn(x)
        x_r  = np.concatenate([[x[mask][0]], x[mask], [x[mask][-1]]])
        y_r  = np.concatenate([[0], y[mask], [0]])
        fig.add_trace(go.Scatter(
            x=x_r, y=y_r, fill="toself",
            fillcolor="rgba(255,75,110,0.40)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"p-value area = {p:.4f}", showlegend=True,
        ))

    # Observed statistic
    fig.add_vline(x=observed_t, line=dict(color="#FFB347", width=2.5),
                  annotation_text=f"observed t = {observed_t}",
                  annotation_font=dict(color="#FFB347", size=11, family=FONT),
                  annotation_position="top left")
    fig.add_vline(x=-observed_t, line=dict(color="#FFB347", width=2.5, dash="dot"),
                  annotation_text=f"−t = {-observed_t}",
                  annotation_font=dict(color="#FFB347", size=10, family=FONT),
                  annotation_position="top right")

    fig.update_layout(
        **_L(f"p-value intuition  ·  t={observed_t}, df={df}  →  p={p:.4f}"),
        xaxis_title="t statistic", yaxis_title="Probability density",
    )
    _add_formula(fig,
        f"p = P(|T| ≥ {abs(observed_t):.2f} | H₀ true) = {p:.4f}  "
        f"— the red area is the probability of seeing a result this extreme by chance alone")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  TYPE I & II ERROR VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def type_error_visual(effect_size: float = 1.5, alpha: float = 0.05, n: int = 30) -> go.Figure:
    """
    Shows two overlapping distributions (H₀ and H₁) with Type I and II error regions.
    """
    se     = 1.0 / np.sqrt(n)
    crit   = stats.norm.ppf(1 - alpha) * se  # one-tailed critical value

    x = np.linspace(-3*se, effect_size + 3*se, 600)
    y0 = stats.norm.pdf(x, 0, se)           # H₀ distribution
    y1 = stats.norm.pdf(x, effect_size, se) # H₁ distribution

    fig = go.Figure()

    # H₀ curve
    fig.add_trace(go.Scatter(x=x, y=y0, mode="lines",
        line=dict(color=PALETTE[0], width=2.5),
        fill="tozeroy", fillcolor="rgba(79,142,247,0.08)",
        name="H₀ distribution (no effect)"))
    # H₁ curve
    fig.add_trace(go.Scatter(x=x, y=y1, mode="lines",
        line=dict(color=PALETTE[4], width=2.5),
        fill="tozeroy", fillcolor="rgba(52,211,153,0.08)",
        name=f"H₁ distribution (effect = {effect_size})"))

    # Type I error (α) — reject H₀ when it's true
    mask_alpha = x >= crit
    xa = np.concatenate([[crit], x[mask_alpha], [x[mask_alpha][-1]]])
    ya = np.concatenate([[0], y0[mask_alpha], [0]])
    fig.add_trace(go.Scatter(x=xa, y=ya, fill="toself",
        fillcolor="rgba(255,75,110,0.45)", line=dict(color="rgba(0,0,0,0)"),
        name=f"Type I error (α = {alpha}) — false alarm"))

    # Type II error (β) — fail to reject H₀ when H₁ is true
    mask_beta = x <= crit
    xb = np.concatenate([[x[mask_beta][0]], x[mask_beta], [crit]])
    yb = np.concatenate([[0], y1[mask_beta], [0]])
    power = 1 - stats.norm.cdf(crit, effect_size, se)
    beta  = 1 - power
    fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself",
        fillcolor="rgba(255,179,71,0.45)", line=dict(color="rgba(0,0,0,0)"),
        name=f"Type II error (β ≈ {beta:.2f}) — missed effect"))

    fig.add_vline(x=crit, line=dict(color="white", width=1.5, dash="dash"),
                  annotation_text=f"critical value = {crit:.3f}",
                  annotation_font=dict(color="white", size=10, family=FONT))

    fig.update_layout(
        **_L(f"Type I & II errors  ·  n={n}, α={alpha}, effect={effect_size}, power≈{power:.2f}"),
        xaxis_title="Test statistic", yaxis_title="Density",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SKEWNESS & KURTOSIS
# ═══════════════════════════════════════════════════════════════════════════════

def skewness_kurtosis_visual() -> go.Figure:
    """Shows distributions with different skewness and kurtosis."""
    rng = np.random.default_rng(11)
    configs = [
        (rng.normal(0, 1, 3000),                     PALETTE[0], "Normal  (skew≈0, kurt≈3)"),
        (rng.exponential(1, 3000),                   PALETTE[1], "Exponential  (right-skewed)"),
        (-rng.exponential(1, 3000),                  PALETTE[2], "Reflected Exp  (left-skewed)"),
        (rng.standard_t(3, 3000),                    PALETTE[3], "t(df=3)  (heavy tails, high kurt)"),
        (rng.uniform(-2, 2, 3000),                   PALETTE[4], "Uniform  (low kurtosis)"),
    ]

    fig = go.Figure()
    for data, color, name in configs:
        sk   = stats.skew(data)
        ku   = stats.kurtosis(data, fisher=False)  # Pearson kurtosis
        kde  = stats.gaussian_kde(data)
        xr   = np.linspace(data.min(), data.max(), 400)
        fig.add_trace(go.Scatter(
            x=xr, y=kde(xr), mode="lines",
            line=dict(color=color, width=2.2),
            name=f"{name}  skew={sk:.2f} kurt={ku:.2f}",
        ))

    fig.update_layout(
        **_L("Skewness and kurtosis  ·  shape beyond mean and variance"),
        xaxis_title="x", yaxis_title="Density",
        xaxis_range=[-6, 6],
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(size=10))
    return fig


# ─── Helper ───────────────────────────────────────────────────────────────────

def _add_formula(fig: go.Figure, text: str):
    fig.add_annotation(
        x=0.01, y=-0.18, xref="paper", yref="paper",
        text=f"<i>{text}</i>",
        showarrow=False,
        font=dict(family=FONT, size=9, color=MUTED),
        align="left",
    )