"""
test_runner.py — Dispatches to the correct test based on wizard config + assumption results.
Handles column selection UI and data extraction.
"""
import numpy as np
import pandas as pd
import streamlit as st

from core.assumption_checks import check_assumptions, extract_groups_from_df
from core.parametric import independent_t_test, paired_t_test, one_way_anova, pearson_correlation
from core.nonparametric import (
    mann_whitney, wilcoxon_signed_rank, kruskal_wallis,
    chi2_independence, spearman_correlation,
)

NONPARAM_FALLBACK = {
    "independent_t": "mann_whitney",
    "paired_t": "wilcoxon",
    "anova": "kruskal",
    "pearson_r": "spearman_r",
}


def select_columns(df: pd.DataFrame, config: dict) -> dict | None:
    """
    Shows column-selection UI appropriate for the test type.
    Returns a dict with the needed column refs, or None if not ready.
    """
    test_id = config["test_id"]
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("### 📂 Map your columns")
    st.caption("Tell me which column contains what.")

    if test_id in ("independent_t", "mann_whitney", "anova", "kruskal"):
        if len(cols) < 2:
            st.error("Need at least 2 columns: one for groups, one for values.")
            return None
        group_col = st.selectbox(
            "Which column has the **group labels**? (e.g. 'Treatment' or 'Group')",
            options=cat_cols if cat_cols else cols,
        )
        value_col = st.selectbox(
            "Which column has the **numbers you want to compare**?",
            options=numeric_cols if numeric_cols else cols,
        )
        return {"group_col": group_col, "value_col": value_col}

    elif test_id in ("paired_t", "wilcoxon"):
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns (before and after).")
            return None
        col1 = st.selectbox("Which column is the **first measurement** (e.g. Before)?", numeric_cols)
        col2 = st.selectbox("Which column is the **second measurement** (e.g. After)?",
                            [c for c in numeric_cols if c != col1])
        return {"col1": col1, "col2": col2}

    elif test_id in ("pearson_r", "spearman_r"):
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns.")
            return None
        x_col = st.selectbox("Which column is the **first variable** (x-axis)?", numeric_cols)
        y_col = st.selectbox("Which column is the **second variable** (y-axis)?",
                             [c for c in numeric_cols if c != x_col])
        return {"x_col": x_col, "y_col": y_col}

    elif test_id in ("chi2_indep", "mcnemar"):
        if len(cols) < 2:
            st.error("Need at least 2 columns.")
            return None
        col1 = st.selectbox("Which column is the **first category**?", cols)
        col2 = st.selectbox("Which column is the **second category**?",
                            [c for c in cols if c != col1])
        return {"col1": col1, "col2": col2}

    return None


def run_test(df: pd.DataFrame, config: dict, col_map: dict) -> tuple[dict, dict]:
    """
    Runs assumption checks then the appropriate test.
    Returns (result_dict, assumption_dict).
    """
    test_id = config["test_id"]
    alpha = config["alpha"]
    tail = config["tail"]

    # ── Extract arrays ────────────────────────────────────────────────────────
    if test_id in ("independent_t", "mann_whitney", "anova", "kruskal"):
        group_col = col_map["group_col"]
        value_col = col_map["value_col"]
        groups_raw = []
        group_names = []
        for name in df[group_col].unique():
            g = df[df[group_col] == name][value_col].dropna().values
            groups_raw.append(g)
            group_names.append(str(name))

    elif test_id in ("paired_t", "wilcoxon"):
        g1 = df[col_map["col1"]].dropna().values
        g2 = df[col_map["col2"]].dropna().values
        min_len = min(len(g1), len(g2))
        g1, g2 = g1[:min_len], g2[:min_len]
        groups_raw = [g1, g2]
        group_names = [col_map["col1"], col_map["col2"]]

    elif test_id in ("pearson_r", "spearman_r"):
        sub = df[[col_map["x_col"], col_map["y_col"]]].dropna()
        x = sub[col_map["x_col"]].values
        y = sub[col_map["y_col"]].values
        groups_raw = [x, y]
        group_names = [col_map["x_col"], col_map["y_col"]]

    elif test_id in ("chi2_indep", "mcnemar"):
        groups_raw = []
        group_names = [col_map["col1"], col_map["col2"]]

    # ── Assumption checks (where applicable) ─────────────────────────────────
    assumption_result = {"use_parametric": True, "checks": [], "normality_ok": True,
                         "variance_ok": True, "small_sample": False}

    if test_id not in ("chi2_indep", "mcnemar") and groups_raw:
        assumption_result = check_assumptions(groups_raw)

    # ── Auto-switch to non-parametric if needed ───────────────────────────────
    actual_test_id = test_id
    switched = False
    if not assumption_result["use_parametric"] and test_id in NONPARAM_FALLBACK:
        actual_test_id = NONPARAM_FALLBACK[test_id]
        switched = True

    assumption_result["switched"] = switched
    assumption_result["original_test"] = test_id
    assumption_result["actual_test"] = actual_test_id

    # ── Run test ──────────────────────────────────────────────────────────────
    result = {}

    if actual_test_id == "independent_t":
        result = independent_t_test(
            groups_raw[0], groups_raw[1], alpha, tail,
            equal_var=assumption_result["variance_ok"],
        )
        result["group_names"] = group_names[:2]

    elif actual_test_id == "mann_whitney":
        result = mann_whitney(groups_raw[0], groups_raw[1], alpha, tail)
        result["group_names"] = group_names[:2]

    elif actual_test_id == "paired_t":
        result = paired_t_test(groups_raw[0], groups_raw[1], alpha, tail)
        result["group_names"] = group_names[:2]

    elif actual_test_id == "wilcoxon":
        result = wilcoxon_signed_rank(groups_raw[0], groups_raw[1], alpha, tail)
        result["group_names"] = group_names[:2]

    elif actual_test_id == "anova":
        result = one_way_anova(groups_raw, group_names, alpha)

    elif actual_test_id == "kruskal":
        result = kruskal_wallis(groups_raw, group_names, alpha)

    elif actual_test_id in ("pearson_r",):
        result = pearson_correlation(x, y, alpha, tail)
        result["col_names"] = group_names

    elif actual_test_id == "spearman_r":
        result = spearman_correlation(x, y, alpha, tail)
        result["col_names"] = group_names

    elif actual_test_id in ("chi2_indep", "mcnemar"):
        result = chi2_independence(df, col_map["col1"], col_map["col2"], alpha)

    # Attach raw groups for plotting
    result["_groups_raw"] = groups_raw
    result["_group_names"] = group_names
    if test_id in ("independent_t", "mann_whitney", "anova", "kruskal"):
        result["_group_col"] = col_map.get("group_col")
        result["_value_col"] = col_map.get("value_col")

    return result, assumption_result
