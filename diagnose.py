"""
diagnose.py — run this to find exactly what is broken before launching the app.
Usage:  python diagnose.py
No Streamlit needed — pure Python.
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\u2705"
FAIL = "\u274c"
WARN = "\u26a0\ufe0f"

results = []

def check(label, fn):
    try:
        fn()
        results.append((PASS, label, None))
        print(f"  {PASS}  {label}")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((FAIL, label, tb))
        print(f"  {FAIL}  {label}")
        print(f"       {type(e).__name__}: {e}")

print()
print("=" * 60)
print("  StatSense — Diagnostic Report")
print("=" * 60)

# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n[1] Python environment")
print(f"  Python {sys.version}")
check("Python >= 3.10", lambda: None if sys.version_info >= (3, 10)
      else (_ for _ in ()).throw(RuntimeError(f"Need Python 3.10+, got {sys.version}")))

# ── 2. Required packages ──────────────────────────────────────────────────────
print("\n[2] Required packages")
packages = [
    ("streamlit",    "streamlit"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("scipy",        "scipy"),
    ("plotly",       "plotly"),
]
for display_name, import_name in packages:
    def _check(n=import_name):
        import importlib
        m = importlib.import_module(n)
        v = getattr(m, "__version__", "unknown")
        print(f"       version: {v}", end="")
    check(f"import {display_name}", _check)
    print()

# ── 3. Core modules ───────────────────────────────────────────────────────────
print("\n[3] Core modules")

check("core.data_loader", lambda: __import__("core.data_loader"))
check("core.wizard", lambda: __import__("core.wizard"))
check("core.assumption_checks", lambda: __import__("core.assumption_checks"))
check("core.parametric", lambda: __import__("core.parametric"))
check("core.nonparametric", lambda: __import__("core.nonparametric"))
check("core.interpreter", lambda: __import__("core.interpreter"))
check("core.test_runner", lambda: __import__("core.test_runner"))

# ── 4. Plot modules ───────────────────────────────────────────────────────────
print("\n[4] Plot modules")

check("plots.distribution_plot", lambda: __import__("plots.distribution_plot"))
check("plots.diagnostic_plots", lambda: __import__("plots.diagnostic_plots"))

# ── 5. Theory modules ─────────────────────────────────────────────────────────
print("\n[5] Theory modules")

check("theory.cards", lambda: __import__("theory.cards"))
check("theory.academic", lambda: __import__("theory.academic"))
check("theory.academy_page", lambda: __import__("theory.academy_page"))

# ── 6. App.py syntax ──────────────────────────────────────────────────────────
print("\n[6] app.py syntax check")
import ast

def check_syntax(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    ast.parse(src)

for pyfile in [
    "app.py",
    "core/data_loader.py",
    "core/wizard.py",
    "core/assumption_checks.py",
    "core/parametric.py",
    "core/nonparametric.py",
    "core/interpreter.py",
    "core/test_runner.py",
    "plots/distribution_plot.py",
    "plots/diagnostic_plots.py",
    "theory/cards.py",
    "theory/academic.py",
    "theory/academy_page.py",
]:
    check(f"syntax: {pyfile}", lambda p=pyfile: check_syntax(p))

# ── 7. End-to-end figure generation ──────────────────────────────────────────
print("\n[7] Figure generation (no Streamlit needed)")

import numpy as np

def make_test_data():
    rng = np.random.default_rng(0)
    g1 = rng.normal(10, 2, 40)
    g2 = rng.normal(12, 2, 40)
    return g1, g2

check("distribution_plot: t", lambda: (
    lambda r: __import__("plots.distribution_plot", fromlist=["make_distribution_plot"])
    .make_distribution_plot({"distribution": "t", "stat": 2.3, "alpha": 0.05, "tail": "two", "df": 28})
)(None))

check("distribution_plot: chi2", lambda: (
    __import__("plots.distribution_plot", fromlist=["make_distribution_plot"])
    .make_distribution_plot({"distribution": "chi2", "stat": 6.0, "alpha": 0.05, "tail": "two", "df": 3})
))

check("distribution_plot: F", lambda: (
    __import__("plots.distribution_plot", fromlist=["make_distribution_plot"])
    .make_distribution_plot({"distribution": "F", "stat": 4.0, "alpha": 0.05, "tail": "two", "df": (2, 27)})
))

def _diag_figures():
    import numpy as np, pandas as pd
    from plots.diagnostic_plots import (
        group_box_plot, group_histogram, paired_change_plot,
        qq_plots, scatter_regression, differences_histogram,
        contingency_heatmap, effect_size_gauge,
    )
    rng = np.random.default_rng(1)
    g1 = rng.normal(10, 2, 40)
    g2 = rng.normal(12, 2, 40)
    group_box_plot([g1, g2], ["A", "B"], "Val")
    group_histogram([g1, g2], ["A", "B"], "Val")
    paired_change_plot(g1, g2)
    qq_plots([g1, g2], ["A", "B"])
    scatter_regression(g1, g2, "X", "Y", 0.4)
    differences_histogram(g2 - g1)
    ct = pd.crosstab(pd.Series(list("abab") * 5), pd.Series(list("xxyy") * 5))
    contingency_heatmap(ct)
    effect_size_gauge(0.6, "Cohen d")

check("diagnostic_plots: all 8 figures", _diag_figures)

def _academic_figures():
    from theory.academic import (
        descriptive_stats_visual, variance_decomposition,
        normal_parameter_explorer, t_distribution_df_explorer,
        chi2_df_explorer, f_distribution_explorer, binomial_explorer,
        estimator_bias_variance, central_limit_theorem,
        confidence_interval_simulation, standard_error_vs_n,
        pvalue_intuition, type_error_visual, skewness_kurtosis_visual,
    )
    import numpy as np
    d = np.random.default_rng(2).normal(50, 10, 150)
    descriptive_stats_visual(d, "Test")
    variance_decomposition(d, "Test")
    normal_parameter_explorer()
    t_distribution_df_explorer()
    chi2_df_explorer()
    f_distribution_explorer()
    binomial_explorer()
    estimator_bias_variance()
    central_limit_theorem("exponential")
    central_limit_theorem("uniform")
    central_limit_theorem("bimodal")
    confidence_interval_simulation()
    standard_error_vs_n()
    pvalue_intuition(2.3, 28)
    type_error_visual(1.5, 0.05, 30)
    skewness_kurtosis_visual()

check("academic figures: all 16 figures", _academic_figures)

# ── 8. End-to-end test pipeline ───────────────────────────────────────────────
print("\n[8] End-to-end test pipeline")

def _e2e_test():
    import numpy as np
    from core.data_loader import _drug_placebo, _before_after, _three_groups, _categorical, _correlation
    from core.assumption_checks import check_assumptions
    from core.parametric import independent_t_test, paired_t_test, one_way_anova, pearson_correlation
    from core.nonparametric import mann_whitney, chi2_independence
    from core.interpreter import interpret

    # Independent t-test
    df = _drug_placebo()
    g1 = df[df.Group == "Drug"].BloodPressure.values
    g2 = df[df.Group == "Placebo"].BloodPressure.values
    assum = check_assumptions([g1, g2])
    res = independent_t_test(g1, g2, 0.05, "two", equal_var=assum["variance_ok"])
    config = {"test_id": "independent_t", "confidence": 95, "tail": "two", "alpha": 0.05}
    interp = interpret(res, config)
    assert "verdict" in interp, "Missing verdict"
    assert res["p"] < 0.05, f"Expected p < 0.05, got {res['p']}"

    # Paired t-test
    df2 = _before_after()
    res2 = paired_t_test(df2.Before.values[:30], df2.After.values[:30], 0.05, "two")
    assert res2["reject"], "Paired t-test should reject H0"

    # ANOVA
    df3 = _three_groups()
    groups = [df3[df3.Fertilizer == f].Height_cm.values for f in ["A", "B", "C"]]
    res3 = one_way_anova(groups, ["A", "B", "C"], 0.05)
    assert res3["reject"], "ANOVA should reject H0"

    # Pearson
    df4 = _correlation()
    res4 = pearson_correlation(df4.StudyHours.values, df4.ExamScore.values, 0.05, "two")
    assert abs(res4["stat"]) > 0.5, "Correlation should be strong"

    # Chi-square
    df5 = _categorical()
    res5 = chi2_independence(df5, "AgeGroup", "Preference", 0.05)
    assert "stat" in res5

check("full pipeline: 5 test types", _e2e_test)

# ── 9. Port availability ──────────────────────────────────────────────────────
print("\n[9] Network")

def _check_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("localhost", 8501))
        if result == 0:
            raise RuntimeError(
                "Port 8501 is already in use — something is already running there. "
                "Stop it first or change the port in docker-compose.yml."
            )

check("Port 8501 is free", _check_port)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  SUMMARY: {passed} passed, {failed} failed")

if failed > 0:
    print()
    print("  FAILURES (full tracebacks):")
    print()
    for icon, label, tb in results:
        if icon == FAIL:
            print(f"  --- {label} ---")
            print(tb)
else:
    print()
    print("  All checks passed. Run the app with:")
    print()
    print("    docker compose up --build")
    print()
    print("  Then open: http://localhost:8501")

print("=" * 60)
print()
