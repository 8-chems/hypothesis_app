"""
nonparametric.py — Mann-Whitney, Wilcoxon, Kruskal-Wallis, Chi-square, Spearman.
Each function returns the same standardized result dict as parametric.py.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _r_from_z(z: float, n: int) -> float:
    return z / np.sqrt(n) if n > 0 else 0.0


def mann_whitney(g1: np.ndarray, g2: np.ndarray, alpha: float, tail: str) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    u_stat, p = stats.mannwhitneyu(g1, g2, alternative=alt)
    
    n1, n2 = len(g1), len(g2)
    # r effect size
    z = stats.norm.ppf(p / 2) if p > 0 else 0
    r = abs(_r_from_z(z, n1 + n2))
    
    r_word = "very weak" if r < 0.1 else "weak" if r < 0.3 else "moderate" if r < 0.5 else "large"
    
    return {
        "test_id": "mann_whitney",
        "stat": u_stat,
        "stat_label": "U",
        "p": p,
        "effect_size": r,
        "effect_label": "r (rank-biserial)",
        "effect_size_word": r_word,
        "ci": None,
        "df": None,
        "reject": p < alpha,
        "alpha": alpha,
        "tail": tail,
        "group_means": [np.median(g1), np.median(g2)],
        "group_stds":  [np.std(g1, ddof=1), np.std(g2, ddof=1)],
        "group_ns":    [n1, n2],
        "distribution": "normal",
        "note": "Medians shown (Mann-Whitney is rank-based)",
    }


def wilcoxon_signed_rank(g1: np.ndarray, g2: np.ndarray, alpha: float, tail: str) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    diffs = g2 - g1
    diffs_nonzero = diffs[diffs != 0]
    
    if len(diffs_nonzero) < 3:
        # Fall back gracefully
        return {
            "test_id": "wilcoxon",
            "stat": 0, "stat_label": "W", "p": 1.0,
            "effect_size": 0, "effect_label": "r", "effect_size_word": "very weak",
            "ci": None, "df": None, "reject": False, "alpha": alpha, "tail": tail,
            "group_means": [np.median(g1), np.median(g2)],
            "group_stds": [0, 0], "group_ns": [len(g1), len(g2)],
            "distribution": "normal", "diffs": diffs,
            "note": "Too few non-zero differences to compute test.",
        }
    
    w_stat, p = stats.wilcoxon(diffs_nonzero, alternative=alt)
    
    n = len(diffs_nonzero)
    z = stats.norm.ppf(max(p / 2, 1e-10))
    r = abs(_r_from_z(z, n))
    r_word = "very weak" if r < 0.1 else "weak" if r < 0.3 else "moderate" if r < 0.5 else "large"
    
    return {
        "test_id": "wilcoxon",
        "stat": w_stat,
        "stat_label": "W",
        "p": p,
        "effect_size": r,
        "effect_label": "r (rank correlation)",
        "effect_size_word": r_word,
        "ci": None,
        "df": None,
        "reject": p < alpha,
        "alpha": alpha,
        "tail": tail,
        "group_means": [np.median(g1), np.median(g2)],
        "group_stds":  [np.std(g1, ddof=1), np.std(g2, ddof=1)],
        "group_ns":    [len(g1), len(g2)],
        "distribution": "normal",
        "diffs": diffs,
        "note": "Medians shown (Wilcoxon is rank-based)",
    }


def kruskal_wallis(groups: list[np.ndarray], group_names: list[str], alpha: float) -> dict:
    h_stat, p = stats.kruskal(*groups)
    
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (h_stat - k + 1) / (n_total - k)  # epsilon-squared approx
    eta2 = max(0, eta2)
    
    effect_word = "very weak" if eta2 < 0.01 else "weak" if eta2 < 0.06 else "moderate" if eta2 < 0.14 else "large"
    
    return {
        "test_id": "kruskal",
        "stat": h_stat,
        "stat_label": "H",
        "p": p,
        "effect_size": eta2,
        "effect_label": "ε² (epsilon-squared)",
        "effect_size_word": effect_word,
        "ci": None,
        "df": k - 1,
        "reject": p < alpha,
        "alpha": alpha,
        "tail": "two",
        "group_means": [np.median(g) for g in groups],
        "group_stds":  [np.std(g, ddof=1) for g in groups],
        "group_ns":    [len(g) for g in groups],
        "group_names": group_names,
        "distribution": "chi2",
        "note": "Medians shown (Kruskal-Wallis is rank-based)",
    }


def chi2_independence(df: pd.DataFrame, col1: str, col2: str, alpha: float) -> dict:
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    
    n = ct.values.sum()
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
    
    v_word = "very weak" if cramers_v < 0.1 else "weak" if cramers_v < 0.3 else "moderate" if cramers_v < 0.5 else "strong"
    
    return {
        "test_id": "chi2_indep",
        "stat": chi2,
        "stat_label": "χ²",
        "p": p,
        "effect_size": cramers_v,
        "effect_label": "Cramér's V",
        "effect_size_word": v_word,
        "ci": None,
        "df": dof,
        "reject": p < alpha,
        "alpha": alpha,
        "tail": "two",
        "distribution": "chi2",
        "contingency_table": ct,
        "expected": pd.DataFrame(expected, index=ct.index, columns=ct.columns),
        "col1": col1,
        "col2": col2,
    }


def spearman_correlation(x: np.ndarray, y: np.ndarray, alpha: float, tail: str) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    r, p = stats.spearmanr(x, y, alternative=alt)
    
    n = len(x)
    r_word = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong" if abs(r) < 0.7 else "very strong"
    
    return {
        "test_id": "spearman_r",
        "stat": r,
        "stat_label": "ρ",
        "p": p,
        "effect_size": r,
        "effect_label": "Spearman ρ",
        "effect_size_word": r_word,
        "ci": None,
        "df": n - 2,
        "reject": p < alpha,
        "alpha": alpha,
        "tail": tail,
        "r_squared": r**2,
        "distribution": "t",
        "x": x,
        "y": y,
    }
