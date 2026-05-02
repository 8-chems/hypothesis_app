"""
assumption_checks.py — Shapiro-Wilk normality, Levene variance, sample size checks.
Returns plain-English verdicts + recommends parametric vs non-parametric.
"""
import numpy as np
import pandas as pd
from scipy import stats


def check_assumptions(groups: list[np.ndarray]) -> dict:
    """
    Given a list of arrays (one per group), returns a dict with:
    - use_parametric: bool
    - checks: list of dicts with {label, status, message, technical}
    - normality_ok: bool
    - variance_ok: bool
    - small_sample: bool
    """
    checks = []
    normality_ok = True
    variance_ok = True
    small_sample = False

    # ── Sample size ───────────────────────────────────────────────────────────
    min_n = min(len(g) for g in groups)
    if min_n < 10:
        small_sample = True
        checks.append({
            "label": "Sample size",
            "status": "warn",
            "message": f"Very small dataset (smallest group has {min_n} observations). Results should be treated as very preliminary.",
            "technical": f"n={min_n} — consider collecting more data before drawing conclusions.",
        })
    elif min_n < 30:
        small_sample = True
        checks.append({
            "label": "Sample size",
            "status": "warn",
            "message": f"Small dataset ({min_n} observations per group). Results are usable but treat with some caution.",
            "technical": f"n={min_n} — Central Limit Theorem may not fully apply; non-parametric tests may be safer.",
        })
    else:
        checks.append({
            "label": "Sample size",
            "status": "ok",
            "message": f"Good sample size ({min_n}+ observations per group). Results should be reliable.",
            "technical": f"n={min_n} — CLT applies; parametric tests are appropriate.",
        })

    # ── Normality (Shapiro-Wilk per group) ───────────────────────────────────
    norm_failures = []
    for i, g in enumerate(groups):
        g_clean = g[~np.isnan(g)]
        if len(g_clean) < 3:
            continue
        if len(g_clean) > 5000:
            # Shapiro unreliable for very large n; use D'Agostino
            stat, p = stats.normaltest(g_clean)
        else:
            stat, p = stats.shapiro(g_clean)
        if p < 0.05:
            norm_failures.append(i + 1)

    if norm_failures:
        normality_ok = False
        label = f"Group{'s' if len(norm_failures) > 1 else ''} {', '.join(map(str, norm_failures))}"
        checks.append({
            "label": "Data shape (normality)",
            "status": "warn",
            "message": f"Your data has an unusual shape — {label} " + ("don't look" if len(norm_failures)>1 else "doesn't look") + " bell-curve-shaped. We'll automatically use a safer test.",
            "technical": f"Shapiro-Wilk p < 0.05 for group(s) {norm_failures}. Non-parametric test recommended.",
        })
    else:
        checks.append({
            "label": "Data shape (normality)",
            "status": "ok",
            "message": "Your data has a roughly normal (bell-curve) shape — that's good.",
            "technical": "Shapiro-Wilk p ≥ 0.05 for all groups. Parametric tests are appropriate.",
        })

    # ── Equal variances (Levene, only for 2+ groups) ──────────────────────────
    if len(groups) >= 2:
        groups_clean = [g[~np.isnan(g)] for g in groups]
        if all(len(g) >= 2 for g in groups_clean):
            lev_stat, lev_p = stats.levene(*groups_clean)
            if lev_p < 0.05:
                variance_ok = False
                checks.append({
                    "label": "Spread similarity (equal variance)",
                    "status": "warn",
                    "message": "The groups have very different spreads. This has been automatically accounted for in the test.",
                    "technical": f"Levene's test p={lev_p:.4f} < 0.05. Welch correction applied for t-test; Kruskal-Wallis used for ANOVA.",
                })
            else:
                checks.append({
                    "label": "Spread similarity (equal variance)",
                    "status": "ok",
                    "message": "Both groups have similar spread — that's what we expect.",
                    "technical": f"Levene's test p={lev_p:.4f} ≥ 0.05. Equal variances assumed.",
                })

    use_parametric = normality_ok and not small_sample

    return {
        "use_parametric": use_parametric,
        "normality_ok": normality_ok,
        "variance_ok": variance_ok,
        "small_sample": small_sample,
        "checks": checks,
    }


def extract_groups_from_df(df: pd.DataFrame, group_col: str, value_col: str) -> list[np.ndarray]:
    groups = []
    for name in df[group_col].unique():
        g = df[df[group_col] == name][value_col].dropna().values
        groups.append(g)
    return groups
