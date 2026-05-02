"""
data_loader.py — handles CSV upload, manual entry, and built-in sample datasets.
"""
import numpy as np
import pandas as pd
import streamlit as st


# ─── Built-in sample datasets ────────────────────────────────────────────────

SAMPLE_DATASETS = {
    "Drug vs Placebo (blood pressure)": {
        "description": "60 patients split into two groups. Does the drug lower blood pressure?",
        "type": "two_groups_numeric",
        "columns": ["Group", "BloodPressure"],
        "generator": lambda: _drug_placebo(),
    },
    "Exam scores: before vs after tutoring": {
        "description": "30 students tested before and after a tutoring program. Did scores improve?",
        "type": "paired_numeric",
        "columns": ["StudentID", "Before", "After"],
        "generator": lambda: _before_after(),
    },
    "Three fertilizers on plant growth": {
        "description": "90 plants assigned to 3 fertilizer types. Which grows tallest?",
        "type": "three_groups_numeric",
        "columns": ["Fertilizer", "Height_cm"],
        "generator": lambda: _three_groups(),
    },
    "Survey: preference vs age group": {
        "description": "200 people. Is product preference related to age group?",
        "type": "categorical",
        "columns": ["AgeGroup", "Preference"],
        "generator": lambda: _categorical(),
    },
    "Study hours vs exam score": {
        "description": "50 students. Is there a relationship between study time and grade?",
        "type": "correlation",
        "columns": ["StudyHours", "ExamScore"],
        "generator": lambda: _correlation(),
    },
}


def _drug_placebo():
    rng = np.random.default_rng(42)
    drug = rng.normal(118, 12, 30)
    placebo = rng.normal(128, 14, 30)
    df = pd.DataFrame({
        "Group": ["Drug"] * 30 + ["Placebo"] * 30,
        "BloodPressure": np.concatenate([drug, placebo]),
    })
    return df


def _before_after():
    rng = np.random.default_rng(7)
    before = rng.normal(65, 10, 30)
    after = before + rng.normal(8, 5, 30)
    df = pd.DataFrame({
        "StudentID": range(1, 31),
        "Before": before,
        "After": after,
    })
    return df


def _three_groups():
    rng = np.random.default_rng(99)
    a = rng.normal(22, 3, 30)
    b = rng.normal(25, 3, 30)
    c = rng.normal(20, 4, 30)
    df = pd.DataFrame({
        "Fertilizer": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "Height_cm": np.concatenate([a, b, c]),
    })
    return df


def _categorical():
    rng = np.random.default_rng(13)
    age_groups = rng.choice(["18-30", "31-50", "51+"], 200, p=[0.4, 0.35, 0.25])
    prefs = []
    for ag in age_groups:
        if ag == "18-30":
            prefs.append(rng.choice(["Product A", "Product B", "Product C"], p=[0.5, 0.3, 0.2]))
        elif ag == "31-50":
            prefs.append(rng.choice(["Product A", "Product B", "Product C"], p=[0.3, 0.4, 0.3]))
        else:
            prefs.append(rng.choice(["Product A", "Product B", "Product C"], p=[0.2, 0.3, 0.5]))
    return pd.DataFrame({"AgeGroup": age_groups, "Preference": prefs})


def _correlation():
    rng = np.random.default_rng(55)
    hours = rng.uniform(1, 10, 50)
    score = 40 + 5 * hours + rng.normal(0, 8, 50)
    score = np.clip(score, 0, 100)
    return pd.DataFrame({"StudyHours": hours, "ExamScore": score})


# ─── Main loader ─────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (dataframe, dataset_type_hint) or (None, None).
    dataset_type_hint is one of: two_groups_numeric, paired_numeric,
    three_groups_numeric, categorical, correlation, custom
    """
    source = st.radio(
        "Where is your data?",
        ["Use a built-in example", "Upload a CSV file", "Enter numbers manually"],
        help="Choose how to bring your data into the app.",
    )

    if source == "Use a built-in example":
        name = st.selectbox("Pick an example dataset", list(SAMPLE_DATASETS.keys()))
        info = SAMPLE_DATASETS[name]
        st.caption(f"📋 {info['description']}")
        df = info["generator"]()
        return df, info["type"]

    elif source == "Upload a CSV file":
        uploaded = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns")
                return df, "custom"
            except Exception as e:
                st.error(f"Could not read file: {e}")
        return None, None

    else:  # manual entry
        st.markdown("**Enter your data below** — one number per line for each group.")
        col1, col2 = st.columns(2)
        with col1:
            raw1 = st.text_area("Group 1 values", "12\n15\n14\n18\n11\n16\n13\n17")
        with col2:
            raw2 = st.text_area("Group 2 values (leave blank if not needed)", "9\n11\n8\n12\n10\n7\n13")
        try:
            g1 = [float(x.strip()) for x in raw1.strip().splitlines() if x.strip()]
            g2_raw = raw2.strip().splitlines()
            if g2_raw:
                g2 = [float(x.strip()) for x in g2_raw if x.strip()]
                df = pd.DataFrame({
                    "Group": ["A"] * len(g1) + ["B"] * len(g2),
                    "Value": g1 + g2,
                })
            else:
                df = pd.DataFrame({"Value": g1})
            return df, "custom"
        except ValueError:
            st.error("Please enter only numbers, one per line.")
            return None, None
