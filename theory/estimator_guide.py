"""
estimator_guide.py — Explains which estimator is used for the current test,
why it's appropriate given the data regime (frequency, sample size, observation type),
and when another distribution's estimator can be substituted.

Called from the "Estimator" tab in the main results page.
Also used by the Academy page as a standalone explorer.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA REGIME CLASSIFIER
#     Given actual data arrays, infer the regime characteristics.
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(groups: list, result: dict) -> dict:
    """
    Analyse the actual data and return a regime dict:
      n_min, n_total, is_small, is_medium, is_large,
      is_binary, is_count, is_skewed, normality_ok,
      overdispersion, event_rate, is_rare, is_very_rare,
      is_sequential (inferred from test_id),
      is_paired, dispersion_index
    """
    test_id = result.get("test_id", "")
    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups]) if groups else np.array([])

    n_min   = min(len(g) for g in groups) if groups else 0
    n_total = sum(len(g) for g in groups) if groups else 0

    # Size thresholds
    is_small  = n_min < 10
    is_medium = 10 <= n_min < 30
    is_large  = n_min >= 30

    # Binary?
    unique_vals = np.unique(all_vals)
    is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})

    # Count data?
    is_count = (
        np.all(all_vals >= 0) and
        np.all(np.floor(all_vals) == all_vals) and
        not is_binary
    ) if len(all_vals) > 0 else False

    # Skewness
    skew_abs = abs(stats.skew(all_vals)) if len(all_vals) >= 3 else 0
    is_skewed = skew_abs > 1.0

    # Normality
    normality_ok = result.get("normality_ok", True)

    # Event rate (for binary/count data)
    event_rate = float(np.mean(all_vals)) if len(all_vals) > 0 else 0.5
    is_rare      = event_rate < 0.05
    is_very_rare = event_rate < 0.01

    # Dispersion index (variance / mean) — for count data
    mu  = float(np.mean(all_vals)) if len(all_vals) > 0 else 1
    var = float(np.var(all_vals, ddof=1)) if len(all_vals) > 1 else 1
    dispersion_index = var / mu if mu > 0 else 1.0
    overdispersed = dispersion_index > 1.5 and is_count

    # Sequential / paired from test_id
    is_sequential = test_id in ("ar", "arma")  # future-proof
    is_paired     = test_id in ("paired_t", "wilcoxon")

    return dict(
        n_min=n_min, n_total=n_total,
        is_small=is_small, is_medium=is_medium, is_large=is_large,
        is_binary=is_binary, is_count=is_count,
        is_skewed=is_skewed, normality_ok=normality_ok,
        event_rate=event_rate, is_rare=is_rare, is_very_rare=is_very_rare,
        overdispersed=overdispersed, dispersion_index=dispersion_index,
        is_sequential=is_sequential, is_paired=is_paired,
        skew_abs=skew_abs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ESTIMATOR EXPLANATION DATABASE
#     For each test_id, explain the estimator used, why it fits, and caveats.
# ══════════════════════════════════════════════════════════════════════════════

ESTIMATOR_PROFILES = {

    "independent_t": {
        "estimator":    "Sample mean (x̄)",
        "formula":      "x̄ = Σxᵢ / n",
        "distribution": "Normal (or CLT approximation)",
        "family":       "Parametric — moment estimator",
        "why": (
            "The sample mean is the **Minimum Variance Unbiased Estimator (MVUE)** "
            "for the Normal distribution — no other unbiased estimator has smaller variance. "
            "For non-Normal data, the **Central Limit Theorem** guarantees the sampling "
            "distribution of x̄ approaches Normal as n grows, justifying the t-test."
        ),
        "ci":  "x̄ ± t*(df) · s/√n",
        "se":  "SE = s / √n",
        "key_property": "MVUE for Normal; CLT-justified for non-Normal with large n",
        "caveats": [
            "Sensitive to outliers — one extreme value pulls x̄ significantly",
            "Assumes errors are additive and variance is finite",
            "With very small n (<10), skewed data can make x̄ misleading even as a point estimate",
        ]
    },

    "mann_whitney": {
        "estimator":    "Hodges-Lehmann estimator (HL)",
        "formula":      "HL = median { (xᵢ + yⱼ)/2 : all i,j pairs }",
        "distribution": "Distribution-free (rank-based)",
        "family":       "Non-parametric — rank estimator",
        "why": (
            "Mann-Whitney works on **ranks**, not raw values. "
            "The Hodges-Lehmann estimator is its natural point estimate — "
            "the median of all pairwise averages between the two groups. "
            "It is robust to outliers, has **95% asymptotic efficiency** relative to the mean "
            "for Normal data, and remains valid for any continuous distribution. "
            "Used when Normality fails or sample sizes are small."
        ),
        "ci":  "Rank-based confidence interval (exact or normal approximation)",
        "se":  "Derived from Wilcoxon U distribution",
        "key_property": "95% asymptotic efficiency vs mean; robust to any distribution shape",
        "caveats": [
            "Estimates a shift in distribution location, not strictly the mean difference",
            "For discrete data, ties require continuity corrections",
            "Less familiar than the mean — harder to communicate to non-technical audiences",
        ]
    },

    "paired_t": {
        "estimator":    "Mean of paired differences (d̄)",
        "formula":      "d̄ = Σ(x₂ᵢ − x₁ᵢ) / n   where dᵢ = x₂ᵢ − x₁ᵢ",
        "distribution": "t-distribution on differences",
        "family":       "Parametric — moment estimator on derived variable",
        "why": (
            "Paired observations are **correlated** — each person/item appears in both groups, "
            "sharing a baseline level. By computing differences dᵢ = after − before, "
            "we eliminate this individual-level noise. The result is a single-sample problem: "
            "is d̄ significantly different from zero? "
            "This design **amplifies power** when within-subject correlation is high — "
            "the higher the correlation, the bigger the power advantage over an independent t-test."
        ),
        "ci":  "d̄ ± t*(n−1) · s_d/√n",
        "se":  "SE = s_d / √n   (s_d = std of differences)",
        "key_property": "Removes between-subject variability; power ∝ within-subject correlation",
        "caveats": [
            "Pairs must be genuinely matched — otherwise an independent test is more appropriate",
            "The differences (not raw values) must satisfy Normality for small n",
            "If n is small and differences are skewed, switch to Wilcoxon signed-rank",
        ]
    },

    "wilcoxon": {
        "estimator":    "Median of paired differences",
        "formula":      "x̃_d = median(dᵢ)   where dᵢ = x₂ᵢ − x₁ᵢ",
        "distribution": "Wilcoxon signed-rank distribution",
        "family":       "Non-parametric — signed-rank estimator",
        "why": (
            "When paired differences are non-Normal or skewed, the mean of differences "
            "can be misleading. The Wilcoxon test ranks the **absolute values** of differences "
            "and checks whether positive ranks dominate negative ones. "
            "The natural point estimate is the **median difference** — robust to outliers "
            "and valid for any continuous symmetric distribution of differences."
        ),
        "ci":  "Walsh averages-based CI (exact or normal approximation for large n)",
        "se":  "Derived from signed-rank statistic W",
        "key_property": "Robust to non-Normal differences; valid for ordinal scales",
        "caveats": [
            "Assumes differences are symmetrically distributed around the median",
            "Cannot be used if differences include exact zeros (they are excluded)",
            "Very small n (<6 pairs) gives very few possible rank combinations — low power",
        ]
    },

    "anova": {
        "estimator":    "Group means (x̄ᵢ) + F-ratio",
        "formula":      "F = MS_between / MS_within = [Σnᵢ(x̄ᵢ−x̄)²/(k−1)] / [ΣΣ(xᵢⱼ−x̄ᵢ)²/(N−k)]",
        "distribution": "F distribution — ratio of two chi-squared variables",
        "family":       "Parametric — variance decomposition",
        "why": (
            "ANOVA decomposes total variance into **signal** (between groups) and "
            "**noise** (within groups). The F-ratio measures signal-to-noise. "
            "Under H₀, both numerator and denominator estimate the same σ² — "
            "so F ≈ 1. A large F is implausible under H₀. "
            "Each group mean x̄ᵢ is the MLE for that group's true mean μᵢ."
        ),
        "ci":  "Per-group: x̄ᵢ ± t*(N−k) · √(MSE/nᵢ)",
        "se":  "SE(x̄ᵢ) = √(MSE/nᵢ)   (MSE = pooled within-group variance)",
        "key_property": "Tests all groups simultaneously — controls Type I error unlike multiple t-tests",
        "caveats": [
            "A significant F only tells you *at least one* group differs — run post-hoc tests to find which",
            "Sensitive to outliers in any group — one bad observation inflates MSE or MS_between",
            "Assumes equal variances across groups (Welch's ANOVA relaxes this)",
        ]
    },

    "kruskal": {
        "estimator":    "Group medians + H statistic (rank-based F analog)",
        "formula":      "H = [12/N(N+1)] · Σ(Rᵢ²/nᵢ) − 3(N+1)",
        "distribution": "Chi-squared (df = k−1) — large-sample approximation",
        "family":       "Non-parametric — rank-based variance decomposition",
        "why": (
            "Kruskal-Wallis is ANOVA on **ranks**. All N observations are ranked together, "
            "then H measures whether groups occupy disproportionately high or low ranks. "
            "Used when ANOVA's Normality or equal-variance assumptions fail, "
            "or for ordinal data where arithmetic on raw values is inappropriate."
        ),
        "ci":  "Per-group medians with bootstrap or Dunn's post-hoc for pairwise comparison",
        "se":  "Asymptotic chi-squared distribution for H",
        "key_property": "Valid for any continuous distribution; handles ordinal data",
        "caveats": [
            "Only tells you distributions differ — not specifically that medians differ (unless distributions have same shape)",
            "Less powerful than ANOVA when data IS Normal",
            "Many ties reduce power — use continuity correction",
        ]
    },

    "pearson_r": {
        "estimator":    "Pearson correlation coefficient (r)",
        "formula":      "r = Σ(xᵢ−x̄)(yᵢ−ȳ) / [(n−1)·sₓ·s_y]   = Cov(X,Y) / (σₓ·σ_y)",
        "distribution": "t-distribution for testing r=0 (via t = r√(n−2)/√(1−r²))",
        "family":       "Parametric — MLE for bivariate Normal correlation",
        "why": (
            "r is the **MLE** for the correlation parameter ρ under bivariate Normality. "
            "It measures the strength and direction of **linear** association. "
            "r² (coefficient of determination) gives the proportion of variance in Y "
            "explained by X — a direct measure of practical effect size. "
            "The Fisher z-transformation (z = arctanh(r)) normalises the sampling distribution "
            "for CI construction."
        ),
        "ci":  "Fisher z-transform: CI_z = arctanh(r) ± 1.96/√(n−3)  →  back-transform",
        "se":  "SE(r) ≈ (1−r²)/√(n−1)   (asymptotic)",
        "key_property": "MLE for ρ under bivariate Normal; r² directly interpretable as explained variance",
        "caveats": [
            "Measures only LINEAR association — misses U-shaped or monotone nonlinear relationships",
            "Heavily influenced by outliers — one extreme (x,y) pair can change r by 0.3+",
            "Does not imply causation; confounding variables can create spurious correlations",
        ]
    },

    "spearman_r": {
        "estimator":    "Spearman rank correlation (ρ)",
        "formula":      "ρ = 1 − 6Σdᵢ² / [n(n²−1)]   where dᵢ = rank(xᵢ) − rank(yᵢ)",
        "distribution": "t-distribution via t = ρ√(n−2)/√(1−ρ²) for large n",
        "family":       "Non-parametric — rank correlation",
        "why": (
            "Spearman's ρ is Pearson's r applied to **ranks** instead of raw values. "
            "It measures **monotonic** association — whether one variable tends to increase "
            "when the other does, regardless of the functional form. "
            "Robust to outliers and valid for ordinal data. "
            "Has **91% asymptotic efficiency** relative to Pearson r when data is bivariate Normal."
        ),
        "ci":  "Fisher z-transform on ρ (approximate for n > 10)",
        "se":  "SE(ρ) ≈ 1/√(n−3)   (asymptotic, via z-transform)",
        "key_property": "Captures any monotone relationship; 91% efficient vs Pearson for Normal data",
        "caveats": [
            "Only detects monotone relationships — misses non-monotone patterns (e.g. quadratic)",
            "Many tied ranks reduce accuracy — use Kendall's τ for heavily tied ordinal data",
            "Less interpretable than r² — ρ² does not directly equal explained variance",
        ]
    },

    "chi2_indep": {
        "estimator":    "Cell proportions (p̂ᵢⱼ) + Chi-squared statistic",
        "formula":      "χ² = Σ (Observed − Expected)² / Expected   Expected_ij = (Rᵢ · Cⱼ) / N",
        "distribution": "Chi-squared (df = (r−1)(c−1))",
        "family":       "Non-parametric — goodness-of-fit on counts",
        "why": (
            "For categorical data, no mean exists. "
            "The chi-square test compares **observed cell counts** to the counts we'd "
            "expect if the two variables were completely independent. "
            "The test statistic measures total discrepancy from independence — "
            "large χ² is implausible under H₀. "
            "Cramér's V (= √(χ²/N·min(r−1,c−1))) converts this to an effect size on [0,1]."
        ),
        "ci":  "Per-cell: Wilson interval for proportions; Cramér's V has bootstrap CI",
        "se":  "Each cell proportion p̂ᵢⱼ has SE = √(p̂ᵢⱼ(1−p̂ᵢⱼ)/N)",
        "key_property": "Only test valid for purely categorical data; no distributional assumption on cells",
        "caveats": [
            "Requires expected cell count ≥ 5 in each cell — use Fisher's exact test otherwise",
            "Sensitive to sample size — large N makes tiny associations statistically significant",
            "Pearson chi-square tests independence, not strength — always report Cramér's V alongside",
        ]
    },
}

# Default profile for unknown test_id
_DEFAULT_PROFILE = {
    "estimator": "Context-dependent",
    "formula": "See test-specific formula",
    "distribution": "Varies",
    "family": "See test output",
    "why": "The estimator used depends on the test selected by the wizard.",
    "ci": "See result card",
    "se": "See result card",
    "key_property": "Check the test card for details",
    "caveats": [],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SUBSTITUTION RULES
#     When can one distribution's estimator stand in for another?
# ══════════════════════════════════════════════════════════════════════════════

SUBSTITUTIONS = [
    {
        "from_dist":  "Binomial(n, p)",
        "to_dist":    "Normal(np, np(1−p))",
        "condition":  "n·p ≥ 5  AND  n·(1−p) ≥ 5",
        "why": (
            "By the CLT, the sum of n Bernoulli(p) trials converges to Normal. "
            "Both tails need enough expected counts — if p is near 0 or 1, "
            "one tail is nearly empty and the Normal approximation collapses. "
            "Apply continuity correction (+0.5) for discrete→continuous mapping."
        ),
        "when_relevant": ["independent_t", "chi2_indep"],
        "breaks_when": "p < 0.05 or p > 0.95 even for large n",
        "practical_tip": "Wald CI for proportions uses this — replace with Wilson score when p is extreme.",
    },
    {
        "from_dist":  "Binomial(n, p)",
        "to_dist":    "Poisson(λ = np)",
        "condition":  "n → large,  p → 0,  n·p = λ fixed",
        "why": (
            "When trials are many and success is rare, most trials yield 0 — "
            "the process looks like random rare arrivals. Poisson simplifies to "
            "one parameter (λ = np) vs two (n, p). Classic use: "
            "defect counts, disease incidence, insurance claims."
        ),
        "when_relevant": ["chi2_indep"],
        "breaks_when": "p is not small (p > 0.1) — the Poisson approximation degrades",
        "practical_tip": "Rule of thumb: use Poisson approximation when n > 20 and p < 0.05.",
    },
    {
        "from_dist":  "Poisson(λ)",
        "to_dist":    "Normal(λ, λ)",
        "condition":  "λ ≥ 10  (λ ≥ 30 for tail accuracy)",
        "why": (
            "For large λ, Poisson is approximately Normal(λ, λ) — mean equals variance. "
            "Enables z-tests and closed-form CIs for count data. "
            "The √x transformation stabilises variance to ≈ 1 for all λ."
        ),
        "when_relevant": ["chi2_indep"],
        "breaks_when": "λ < 5 — tails are severely underestimated; use exact Poisson",
        "practical_tip": "For λ between 5 and 10, use the Freeman-Tukey transformation √x + √(x+1).",
    },
    {
        "from_dist":  "t(df)",
        "to_dist":    "Normal(0, 1)",
        "condition":  "df ≥ 30  (df ≥ 120 for tail accuracy at α = 0.001)",
        "why": (
            "t(df) → N(0,1) as df → ∞. The t-distribution is needed when σ is estimated "
            "from the data — for small n, this estimation adds uncertainty, inflating the tails. "
            "As n grows, s → σ and the extra tail weight disappears."
        ),
        "when_relevant": ["independent_t", "paired_t"],
        "breaks_when": "df < 30 — z-based critical values understate required evidence; false positives increase",
        "practical_tip": "Always use t, not z, when σ is estimated. The cost is negligible for large n.",
    },
    {
        "from_dist":  "Mann-Whitney (non-parametric)",
        "to_dist":    "t-test (parametric)",
        "condition":  "n ≥ 30  AND  Normality confirmed  AND  variances equal",
        "why": (
            "When all parametric assumptions hold, the t-test is more powerful "
            "than Mann-Whitney. The ARE (Asymptotic Relative Efficiency) of Mann-Whitney "
            "vs t-test for Normal data is π/3 ≈ 0.955 — only a 4.5% power loss. "
            "For non-Normal data, Mann-Whitney can be MORE powerful than the t-test."
        ),
        "when_relevant": ["mann_whitney", "independent_t"],
        "breaks_when": "Non-Normal data or outliers — Mann-Whitney is safer",
        "practical_tip": "The app switches automatically based on Shapiro-Wilk. You rarely need to choose manually.",
    },
    {
        "from_dist":  "Chi-squared(k)",
        "to_dist":    "Normal(k, 2k)",
        "condition":  "k (df) ≥ 30",
        "why": (
            "χ²(k) is a sum of k squared Normal variables — CLT applies. "
            "For large df, goodness-of-fit and independence tests can use z-approximations. "
            "Wilson-Hilferty cube-root transformation gives better Normal approximation "
            "for moderate k: (χ²/k)^(1/3) ≈ N(1 − 2/(9k), 2/(9k))."
        ),
        "when_relevant": ["chi2_indep"],
        "breaks_when": "Small df or sparse cells — exact methods (Fisher) needed",
        "practical_tip": "For 2×2 tables with any expected cell < 5, always use Fisher's exact test.",
    },
    {
        "from_dist":  "Negative Binomial(r, p)",
        "to_dist":    "Poisson(λ)",
        "condition":  "Dispersion index ≈ 1  (variance / mean ≈ 1)",
        "why": (
            "Negative Binomial reduces to Poisson when the dispersion parameter r → ∞. "
            "In practice: if variance/mean ≈ 1, Poisson is sufficient. "
            "If variance >> mean (overdispersion > 1.5), Negative Binomial is required "
            "— Poisson will underestimate uncertainty and give false precision."
        ),
        "when_relevant": ["chi2_indep"],
        "breaks_when": "Overdispersion index > 1.5 — use Negative Binomial or quasi-Poisson",
        "practical_tip": "Run a likelihood ratio test between Poisson and NB to decide formally.",
    },
    {
        "from_dist":  "Wilcoxon signed-rank (non-parametric)",
        "to_dist":    "Paired t-test (parametric)",
        "condition":  "Differences are Normal  AND  n ≥ 15",
        "why": (
            "The ARE of Wilcoxon signed-rank vs paired t-test for Normal differences is "
            "π/3 ≈ 0.955 — almost no power loss. When differences ARE Normal, "
            "the paired t-test is marginally more powerful. "
            "For non-Normal or skewed differences, Wilcoxon dominates."
        ),
        "when_relevant": ["wilcoxon", "paired_t"],
        "breaks_when": "Non-Normal or skewed differences — Wilcoxon is the safer choice",
        "practical_tip": "The app checks Normality of differences and switches automatically.",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REGIME-AWARE NARRATIVE
#     Generate a plain-English explanation tied to the actual data regime.
# ══════════════════════════════════════════════════════════════════════════════

def regime_narrative(regime: dict, test_id: str) -> dict:
    """
    Returns dict with:
      size_story, frequency_story, obs_story, estimator_choice_story, warning_story
    All plain English, specific to the observed regime.
    """
    n = regime["n_min"]
    stories = {}

    # ── Sample size story ──────────────────────────────────────────────────────
    if regime["is_large"]:
        stories["size_story"] = (
            f"With **n = {n}** observations per group, the Central Limit Theorem is firmly "
            f"in effect. The sampling distribution of the mean is approximately Normal "
            f"regardless of your data's shape — parametric estimators are well-justified."
        )
    elif regime["is_medium"]:
        stories["size_story"] = (
            f"With **n = {n}** observations per group, the CLT is beginning to apply "
            f"but isn't guaranteed. The t-distribution's heavier tails account for this "
            f"uncertainty. If your data is visibly non-Normal, the non-parametric fallback "
            f"is the safer choice."
        )
    else:
        stories["size_story"] = (
            f"With only **n = {n}** observations per group, the CLT does not apply. "
            f"No distributional assumptions can be reliably verified from this little data. "
            f"Non-parametric or exact methods are used — but treat all results as preliminary. "
            f"Consider collecting more data."
        )

    # ── Event frequency story ──────────────────────────────────────────────────
    if regime["is_binary"]:
        p = regime["event_rate"]
        if regime["is_very_rare"]:
            stories["frequency_story"] = (
                f"Your outcome is **binary** with a very rare event rate (≈ {p*100:.1f}%). "
                f"Standard Normal approximations for proportions (Wald interval) fail completely here — "
                f"they can produce CIs that include negative probabilities. "
                f"The **exact Binomial (Clopper-Pearson)** or **Wilson score** interval is required."
            )
        elif regime["is_rare"]:
            stories["frequency_story"] = (
                f"Your outcome is **binary** with a rare event rate (≈ {p*100:.1f}%). "
                f"The Wald interval undercovers at this rate. "
                f"The **Wilson score interval** provides near-nominal coverage and is preferred."
            )
        else:
            stories["frequency_story"] = (
                f"Your outcome is **binary** (proportion ≈ {p*100:.1f}%). "
                f"Both tails are well-populated, so the Normal approximation is valid "
                f"and the standard Wald interval performs well."
            )
    elif regime["is_count"]:
        di = regime["dispersion_index"]
        if regime["overdispersed"]:
            stories["frequency_story"] = (
                f"Your data is **count-based** with a dispersion index of {di:.2f} "
                f"(variance / mean > 1.5). This indicates **overdispersion** — the Poisson "
                f"assumption of mean = variance is violated. A Negative Binomial model "
                f"or quasi-Poisson is more appropriate."
            )
        else:
            stories["frequency_story"] = (
                f"Your data is **count-based** with a dispersion index of {di:.2f} "
                f"(variance ≈ mean). Poisson assumptions hold — the MLE λ̂ = x̄ is "
                f"both unbiased and the UMVUE for the Poisson rate."
            )
    else:
        stories["frequency_story"] = ""

    # ── Observation type story ─────────────────────────────────────────────────
    if regime["is_paired"]:
        stories["obs_story"] = (
            "Observations are **paired** — the same subject appears in both conditions. "
            "The estimator works on the *differences* dᵢ = after − before, "
            "not the raw values. This removes between-subject variability "
            "and increases power when within-subject correlation is high."
        )
    elif regime["is_sequential"]:
        stories["obs_story"] = (
            "Observations are **time-ordered**. Using a simple mean ignores autocorrelation, "
            "which inflates the effective sample size and makes standard errors too small. "
            "An autoregressive estimator is needed."
        )
    else:
        stories["obs_story"] = (
            "Observations are **independent** across groups — "
            "the standard assumption that justifies all classical estimators."
        )

    # ── Estimator choice story ─────────────────────────────────────────────────
    switched = not regime["normality_ok"] or regime["is_small"]
    if switched:
        stories["estimator_choice_story"] = (
            "Because the Normality assumption was not met (or the sample is small), "
            "the app switched to a **rank-based non-parametric estimator**. "
            "This trades a small amount of efficiency for robustness — "
            "it remains valid regardless of the underlying distribution shape."
        )
    else:
        stories["estimator_choice_story"] = (
            "All parametric assumptions were met. "
            "The **parametric estimator** (based on means and the t or F distribution) "
            "is used — it is more efficient than rank-based alternatives when assumptions hold, "
            "meaning it can detect smaller true effects with the same sample size."
        )

    # ── Warning story ──────────────────────────────────────────────────────────
    warnings = []
    if regime["is_skewed"] and not switched:
        warnings.append(
            f"⚠️ Skewness is {regime['skew_abs']:.2f} — your data has an asymmetric shape. "
            "The mean may not represent the typical value well. Consider reporting the median alongside."
        )
    if regime["is_small"]:
        warnings.append(
            f"⚠️ n = {n} is very small. Even the non-parametric test has very low power — "
            "a real effect may exist but go undetected. Treat results as exploratory only."
        )
    stories["warning_story"] = warnings

    return stories


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN API — called from the Streamlit tab
# ══════════════════════════════════════════════════════════════════════════════

def get_estimator_context(result: dict, assumption_result: dict) -> dict:
    """
    Full context dict passed to the Streamlit renderer.
    """
    test_id = result.get("test_id", "")
    actual  = assumption_result.get("actual_test", test_id)
    groups  = result.get("_groups_raw", [])

    profile  = ESTIMATOR_PROFILES.get(actual, _DEFAULT_PROFILE)
    regime   = classify_regime(groups, result)
    narrative = regime_narrative(regime, actual)

    # Relevant substitutions for this test
    relevant_subs = [
        s for s in SUBSTITUTIONS
        if test_id in s["when_relevant"] or actual in s["when_relevant"]
    ]

    return dict(
        test_id=test_id,
        actual_test=actual,
        switched=assumption_result.get("switched", False),
        profile=profile,
        regime=regime,
        narrative=narrative,
        substitutions=relevant_subs,
    )
