"""
parametric.py — t-tests, ANOVA, Pearson correlation.
Each function returns a standardized result dict.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _effect_size_label(d: float) -> str:
    d = abs(d)
    if d < 0.2:
        return "very small"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "moderate"
    return "large"


def independent_t_test(g1: np.ndarray, g2: np.ndarray, alpha: float, tail: str, equal_var: bool = True) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alt)
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)
    d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0.0
    
    # 95% CI on difference of means
    se = np.sqrt(np.var(g1, ddof=1)/len(g1) + np.var(g2, ddof=1)/len(g2))
    diff = np.mean(g1) - np.mean(g2)
    ci = (diff - 1.96*se, diff + 1.96*se)
    
    df = len(g1) + len(g2) - 2
    reject = p < alpha
    
    return {
        "test_id": "independent_t",
        "stat": t_stat,
        "stat_label": "t",
        "p": p,
        "effect_size": d,
        "effect_label": "Cohen's d",
        "effect_size_word": _effect_size_label(d),
        "ci": ci,
        "df": df,
        "reject": reject,
        "alpha": alpha,
        "tail": tail,
        "group_means": [np.mean(g1), np.mean(g2)],
        "group_stds":  [np.std(g1, ddof=1), np.std(g2, ddof=1)],
        "group_ns":    [len(g1), len(g2)],
        "distribution": "t",
        "welch": not equal_var,
    }


def paired_t_test(g1: np.ndarray, g2: np.ndarray, alpha: float, tail: str) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    diffs = g2 - g1
    t_stat, p = stats.ttest_rel(g1, g2, alternative=alt)
    
    d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0
    se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    ci = (np.mean(diffs) - 1.96*se, np.mean(diffs) + 1.96*se)
    df = len(diffs) - 1
    reject = p < alpha
    
    return {
        "test_id": "paired_t",
        "stat": t_stat,
        "stat_label": "t",
        "p": p,
        "effect_size": d,
        "effect_label": "Cohen's d (paired)",
        "effect_size_word": _effect_size_label(d),
        "ci": ci,
        "df": df,
        "reject": reject,
        "alpha": alpha,
        "tail": tail,
        "group_means": [np.mean(g1), np.mean(g2)],
        "group_stds":  [np.std(g1, ddof=1), np.std(g2, ddof=1)],
        "group_ns":    [len(g1), len(g2)],
        "mean_diff": np.mean(diffs),
        "distribution": "t",
        "diffs": diffs,
    }


def one_way_anova(groups: list[np.ndarray], group_names: list[str], alpha: float) -> dict:
    f_stat, p = stats.f_oneway(*groups)
    
    # eta-squared
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total   = sum(np.sum((x - grand_mean)**2) for g in groups for x in g)
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    
    n_total  = sum(len(g) for g in groups)
    df_between = len(groups) - 1
    df_within  = n_total - len(groups)
    reject = p < alpha
    
    return {
        "test_id": "anova",
        "stat": f_stat,
        "stat_label": "F",
        "p": p,
        "effect_size": eta2,
        "effect_label": "η² (eta-squared)",
        "effect_size_word": _effect_size_label(eta2 / 0.06),  # rescale for label
        "ci": None,
        "df": (df_between, df_within),
        "reject": reject,
        "alpha": alpha,
        "tail": "two",
        "group_means": [np.mean(g) for g in groups],
        "group_stds":  [np.std(g, ddof=1) for g in groups],
        "group_ns":    [len(g) for g in groups],
        "group_names": group_names,
        "distribution": "F",
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray, alpha: float, tail: str) -> dict:
    alt = "two-sided" if tail == "two" else "greater"
    r, p = stats.pearsonr(x, y)
    
    n = len(x)
    se = np.sqrt((1 - r**2) / (n - 2)) if n > 2 else 1.0
    z = np.arctanh(r)
    ci_z = (z - 1.96/np.sqrt(n-3), z + 1.96/np.sqrt(n-3))
    ci = (np.tanh(ci_z[0]), np.tanh(ci_z[1]))
    
    reject = p < alpha
    r_word = "very weak" if abs(r) < 0.1 else "weak" if abs(r) < 0.3 else "moderate" if abs(r) < 0.5 else "strong" if abs(r) < 0.7 else "very strong"
    
    return {
        "test_id": "pearson_r",
        "stat": r,
        "stat_label": "r",
        "p": p,
        "effect_size": r,
        "effect_label": "Pearson r",
        "effect_size_word": r_word,
        "ci": ci,
        "df": n - 2,
        "reject": reject,
        "alpha": alpha,
        "tail": tail,
        "r_squared": r**2,
        "distribution": "t",
        "x": x,
        "y": y,
    }
