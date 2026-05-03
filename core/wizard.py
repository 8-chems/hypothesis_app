from typing import Optional, Tuple, List, Dict
"""
wizard.py — Guided question flow that determines the correct statistical test.
The user never sees test names unless they want to.
"""
import streamlit as st

# ─── Test lookup table ────────────────────────────────────────────────────────
# Keys: (goal, same_or_different, data_type)
# Values: (test_id, friendly_name, technical_name)

TEST_MAP = {
    # goal="2_groups", same/diff, numeric/categorical
    ("2_groups", "different", "numeric"):  ("independent_t",    "Comparing two separate groups",      "Independent samples t-test"),
    ("2_groups", "same",      "numeric"):  ("paired_t",          "Comparing before & after",           "Paired samples t-test"),
    ("2_groups", "different", "categorical"): ("chi2_indep",     "Comparing two groups' choices",      "Chi-square independence test"),
    ("2_groups", "same",      "categorical"): ("mcnemar",        "Comparing paired yes/no answers",    "McNemar test"),

    # goal="3plus_groups"
    ("3plus_groups", "different", "numeric"):  ("anova",         "Comparing three or more groups",     "One-way ANOVA"),
    ("3plus_groups", "same",      "numeric"):  ("repeated_anova","Comparing same group at 3+ times",   "Repeated measures ANOVA"),
    ("3plus_groups", "different", "categorical"): ("chi2_indep",  "Comparing groups' category choices","Chi-square independence test"),

    # goal="over_time" (treated as paired)
    ("over_time", "same", "numeric"):  ("paired_t",              "Measuring change over time",          "Paired samples t-test"),
    ("over_time", "same", "categorical"): ("mcnemar",            "Tracking yes/no change over time",    "McNemar test"),

    # goal="relationship"
    ("relationship", "n/a", "numeric"):    ("pearson_r",         "Finding a relationship between numbers","Pearson correlation"),
    ("relationship", "n/a", "categorical"):("chi2_indep",        "Finding a link between categories",   "Chi-square independence test"),
    ("relationship", "n/a", "mixed"):      ("pearson_r",         "Finding a relationship",              "Correlation analysis"),
}

# Fallback non-parametric equivalents
NONPARAM_FALLBACK = {
    "independent_t": ("mann_whitney",   "Mann-Whitney U test"),
    "paired_t":       ("wilcoxon",       "Wilcoxon signed-rank test"),
    "anova":          ("kruskal",        "Kruskal-Wallis test"),
    "pearson_r":      ("spearman_r",     "Spearman rank correlation"),
}

CONFIDENCE_LABELS = {
    90: ("Exploratory", "I'm just getting a first look — being wrong 10% of the time is OK"),
    95: ("Standard",    "The level used in most published research — 5% chance of a false alarm"),
    99: ("Very cautious","I need to be really sure — only 1% chance of a false alarm"),
}


def run_wizard(dataset_type_hint: Optional[str]) -> dict:
    """
    Runs the 3-question wizard and returns a config dict with:
    - test_id, friendly_name, technical_name
    - alpha, confidence_level
    - tail
    """
    st.markdown("### 🧭 Tell me about your question")
    st.caption("Answer in plain English — I'll figure out the right analysis.")

    # ── Q1: goal ──────────────────────────────────────────────────────────────
    goal_map = {
        "Are two groups different from each other?":        "2_groups",
        "Are three or more groups different?":              "3plus_groups",
        "Did the same group change over time?":             "over_time",
        "Is there a relationship between two things?":      "relationship",
    }
    goal_label = st.radio(
        "**1 · What are you trying to find out?**",
        list(goal_map.keys()),
    )
    goal = goal_map[goal_label]

    # ── Q2: same/different (skip for relationship) ────────────────────────────
    if goal in ("2_groups", "3plus_groups"):
        same_map = {
            "Different groups of people/items (no overlap)":    "different",
            "The same people/items measured more than once":     "same",
        }
        same_label = st.radio(
            "**2 · Are these the same people/items or different ones?**",
            list(same_map.keys()),
        )
        same_or_diff = same_map[same_label]
    elif goal == "over_time":
        same_or_diff = "same"
    else:
        same_or_diff = "n/a"

    # ── Q3: data type ─────────────────────────────────────────────────────────
    dtype_map = {
        "Numbers (like height, score, temperature, time)":  "numeric",
        "Categories (like yes/no, color, type, grade)":      "categorical",
        "One of each (one number column, one category)":     "mixed",
    }
    dtype_label = st.radio(
        "**3 · What kind of values do you have?**",
        list(dtype_map.keys()),
    )
    dtype = dtype_map[dtype_label]

    # ── Confidence ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎚️ How cautious do you want to be?")
    confidence = st.select_slider(
        "If there's really no difference, how often am I allowed to wrongly say there is?",
        options=[90, 95, 99],
        value=95,
        format_func=lambda v: f"{v}% confident — {CONFIDENCE_LABELS[v][0]}",
    )
    st.caption(f"💬 {CONFIDENCE_LABELS[confidence][1]}")
    alpha = (100 - confidence) / 100

    # ── Tail ──────────────────────────────────────────────────────────────────
    with st.expander("⚙️ Advanced: one-tailed or two-tailed? (most people leave this as-is)"):
        tail_map = {
            "Two-tailed — I just want to know if they're different (default)": "two",
            "One-tailed — I expect one specific direction (e.g., treatment > control)": "one",
        }
        tail_label = st.radio("Direction", list(tail_map.keys()))
        tail = tail_map[tail_label]

    # ── Look up test ──────────────────────────────────────────────────────────
    key = (goal, same_or_diff, dtype)
    # try mixed -> numeric fallback for relationship
    if key not in TEST_MAP and dtype == "mixed":
        key = (goal, same_or_diff, "numeric")
    if key not in TEST_MAP:
        # best-effort fallback
        key = ("2_groups", "different", "numeric")

    test_id, friendly_name, technical_name = TEST_MAP[key]

    return {
        "test_id": test_id,
        "friendly_name": friendly_name,
        "technical_name": technical_name,
        "alpha": alpha,
        "confidence": confidence,
        "tail": tail,
        "goal": goal,
        "same_or_diff": same_or_diff,
        "dtype": dtype,
    }
