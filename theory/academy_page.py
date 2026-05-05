"""
academy_page.py — renders the Statistics Academy Streamlit page.
Separated from app.py to keep string literals clean.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm as sci_norm

from theory.estimator_guide import ESTIMATOR_PROFILES, SUBSTITUTIONS, classify_regime
from theory.estimator_tab   import render_estimator_tab, _concept_box, _check_condition_met
from theory.academic import (
    descriptive_stats_visual, variance_decomposition,
    normal_parameter_explorer, t_distribution_df_explorer,
    chi2_df_explorer, f_distribution_explorer, binomial_explorer,
    estimator_bias_variance, central_limit_theorem,
    confidence_interval_simulation, standard_error_vs_n,
    pvalue_intuition, type_error_visual, skewness_kurtosis_visual,
)


# ── Shared UI helpers ─────────────────────────────────────────────────────────

def _academy_header(title: str, subtitle: str):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <h2 style="font-family:'Space Grotesk';font-size:1.5rem;font-weight:700;
                   color:#e8eef8;margin:0 0 0.4rem">{title}</h2>
        <p style="color:#8899b0;font-size:0.92rem;line-height:1.6;margin:0">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def _concept_box(title: str, body: str, icon: str = "📌"):
    st.markdown(f"""
    <div style="background:rgba(79,142,247,0.06);border:1px solid rgba(79,142,247,0.18);
                border-left:3px solid #4F8EF7;border-radius:8px;
                padding:1rem 1.2rem;margin-bottom:0.9rem">
        <div style="font-family:'IBM Plex Mono';font-size:0.78rem;font-weight:600;
                    color:#4F8EF7;letter-spacing:0.06em;margin-bottom:0.4rem">{icon} {title.upper()}</div>
        <div style="color:#c8d0e0;font-size:0.88rem;line-height:1.65;white-space:pre-line">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def _formula_box(latex_text: str):
    st.markdown(f"""
    <div style="background:rgba(162,89,255,0.07);border:1px solid rgba(162,89,255,0.2);
                border-radius:8px;padding:0.8rem 1.2rem;margin-bottom:0.9rem;
                font-family:'IBM Plex Mono';font-size:0.9rem;color:#c8d0e0;text-align:center">
        {latex_text}
    </div>
    """, unsafe_allow_html=True)


# ── Main page entry ───────────────────────────────────────────────────────────

def render_academy_page():
    # ── Back button + header ──────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 6])
    with col_back:
        if st.button("← Back"):
            st.session_state["show_academy"] = False
            st.rerun()
    with col_title:
        st.markdown("""
        <h1 style="font-family:'Space Grotesk';font-size:1.9rem;font-weight:700;
                   background:linear-gradient(135deg,#4F8EF7,#A259FF);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0">🎓 Statistics Academy</h1>
        <p style="color:#8899b0;font-size:0.88rem;margin:0.2rem 0 0">
            Interactive visual lessons on the foundations behind every hypothesis test
        </p>""", unsafe_allow_html=True)

    st.markdown("---")

    CHAPTERS = {
        "📐 1 · Descriptive Statistics":             "desc",
        "🔔 2 · Distributions & Their Parameters":   "dists",
        "📊 3 · Variance, Spread & Shape":            "variance",
        "🔭 4 · Estimators & Sampling":               "estimators",
        "🎯 5 · The Central Limit Theorem":           "clt",
        "🪟 6 · Confidence Intervals":                "ci",
        "⚖️ 7 · p-values & Hypothesis Testing":      "pval",
        "❌ 8 · Type I & II Errors, Power":           "errors",
        "🎰 9 · Estimator Explorer":                    "estimators_deep",
    }

    nav_col, content_col = st.columns([1, 3])

    with nav_col:
        st.markdown('<div style="font-family:\'IBM Plex Mono\';font-size:0.68rem;'
                    'color:#4F8EF7;letter-spacing:0.1em;margin-bottom:0.6rem">CHAPTERS</div>',
                    unsafe_allow_html=True)
        chapter_label = st.radio("", list(CHAPTERS.keys()), label_visibility="collapsed")
        chapter = CHAPTERS[chapter_label]

    with content_col:
        _render_chapter(chapter)


# ── Chapter dispatcher ────────────────────────────────────────────────────────

def _render_chapter(chapter: str):

    # ── 1. Descriptive Statistics ─────────────────────────────────────────────
    if chapter == "desc":
        _academy_header(
            "Descriptive Statistics",
            "Before testing anything, we summarise data. The mean, median, variance, and "
            "standard deviation are the building blocks of every inferential method.",
        )

        _concept_box("Mean (x̄)",
            "The arithmetic average: add all values, divide by n.\n\n"
            "x\u0304 = (x\u2081 + x\u2082 + \u2026 + x\u2099) / n\n\n"
            "Sensitive to outliers \u2014 one extreme value pulls it hard.", "📌")

        _concept_box("Median",
            "The middle value when data is sorted. Half the values are above, half below.\n\n"
            "Robust to outliers \u2014 an extreme value barely moves it.", "📌")

        _concept_box("Variance (s\u00b2)",
            "Average squared deviation from the mean:\n\n"
            "s\u00b2 = \u03a3(x\u1d62 \u2212 x\u0304)\u00b2 / (n\u22121)\n\n"
            "Squaring removes negatives and penalises large deviations more. "
            "Dividing by n\u22121 (Bessel\u2019s correction) makes s\u00b2 unbiased.", "📐")

        _concept_box("Standard Deviation (s)",
            "Square root of variance \u2014 same units as the data.\n\n"
            "s = \u221as\u00b2\n\n"
            "Rule of thumb for bell-shaped data: \u224868\u0025 of values fall within 1 SD of the mean, "
            "\u224895\u0025 within 2 SD, \u224899.7\u0025 within 3 SD (the empirical rule).", "📐")

        groups = st.session_state.get("result", {}).get("_groups_raw", [])
        gnames = st.session_state.get("result", {}).get("_group_names", [])
        if groups:
            all_data = np.concatenate(groups)
            label = gnames[0] if gnames else "Value"
            st.plotly_chart(descriptive_stats_visual(all_data, label), use_container_width=True)
            st.caption("Visualisation uses your actual dataset. "
                       "Mean (red), median (purple), \u00b11 and \u00b12 SD bands shown.")
        else:
            rng = np.random.default_rng(0)
            demo = rng.normal(50, 12, 200)
            st.plotly_chart(descriptive_stats_visual(demo, "Demo data (Normal, \u03bc=50, \u03c3=12)"),
                            use_container_width=True)
            st.caption("Run an analysis first to see your own data here.")

    # ── 2. Distributions ──────────────────────────────────────────────────────
    elif chapter == "dists":
        _academy_header(
            "Distributions & Their Parameters",
            "A probability distribution describes the likelihood of every possible outcome. "
            "Parameters like \u03bc and \u03c3 control location and shape.",
        )

        dist_choice = st.radio("Choose a distribution:",
            ["Normal", "Student\u2019s t", "Chi-squared (\u03c7\u00b2)", "F distribution", "Binomial"],
            horizontal=True)

        if dist_choice == "Normal":
            _concept_box("The Normal (Gaussian) distribution",
                "Defined by two parameters:\n"
                "  \u2022 \u03bc (mu) = mean \u2014 shifts the curve left or right\n"
                "  \u2022 \u03c3 (sigma) = standard deviation \u2014 controls width\n\n"
                "The standard Normal N(0,1) has \u03bc=0 and \u03c3=1. "
                "Any Normal can be standardised with z = (x\u2212\u03bc)/\u03c3.", "\U0001f514")
            _formula_box("f(x) = (1/\u03c3\u221a2\u03c0) \u00b7 exp(\u2212(x\u2212\u03bc)\u00b2 / 2\u03c3\u00b2)")
            st.plotly_chart(normal_parameter_explorer(), use_container_width=True)
            _concept_box("Why it appears everywhere",
                "The Central Limit Theorem (Chapter 5) guarantees that sample means from any "
                "distribution converge to Normal as n grows. "
                "That\u2019s why Normal-based tests work even when raw data isn\u2019t Normal.", "\U0001f4a1")

        elif dist_choice == "Student\u2019s t":
            _concept_box("Student\u2019s t distribution",
                "Like Normal but with heavier tails \u2014 more probability in the extremes.\n\n"
                "Governed by one parameter: df (degrees of freedom) = n\u22121 for a one-sample test.\n\n"
                "As df \u2192 \u221e it converges to N(0,1). With small n (low df), heavier tails make it "
                "harder to get a significant result \u2014 which is correct, since small samples give less certainty.", "📊")
            _formula_box("f(t) = \u0393((df+1)/2) / (\u221adf\u03c0 \u00b7 \u0393(df/2)) \u00b7 (1 + t\u00b2/df)^(\u2212(df+1)/2)")
            st.plotly_chart(t_distribution_df_explorer(), use_container_width=True)
            _concept_box("Historical note",
                "William Gosset derived this in 1908 while working at Guinness Brewery. "
                "He published under the pseudonym \u2018Student\u2019 because Guinness didn\u2019t allow "
                "employees to publish. The t-test is why small-batch quality control became rigorous.", "\U0001f4a1")

        elif dist_choice == "Chi-squared (\u03c7\u00b2)":
            _concept_box("Chi-squared distribution",
                "Sum of k squared standard Normal variables:\n\n"
                "\u03c7\u00b2(k) = Z\u2081\u00b2 + Z\u2082\u00b2 + \u2026 + Z\u2096\u00b2\n\n"
                "k = df controls the shape. Always positive, right-skewed for small df, "
                "increasingly symmetric for large df.", "📐")
            _formula_box("f(x) = x^(k/2\u22121) \u00b7 e^(\u2212x/2) / (2^(k/2) \u00b7 \u0393(k/2))")
            st.plotly_chart(chi2_df_explorer(), use_container_width=True)

        elif dist_choice == "F distribution":
            _concept_box("F distribution",
                "Ratio of two independent chi-squared variables divided by their df:\n\n"
                "F = (\u03c7\u00b2\u2081/df\u2081) / (\u03c7\u00b2\u2082/df\u2082)\n\n"
                "In ANOVA: F = variance between groups / variance within groups. "
                "A large F means the signal (group differences) dominates the noise.", "\u2696\ufe0f")
            st.plotly_chart(f_distribution_explorer(), use_container_width=True)

        elif dist_choice == "Binomial":
            _concept_box("Binomial distribution",
                "Number of successes in n independent yes/no trials, each with probability p.\n\n"
                "P(X=k) = C(n,k) \u00b7 p\u1d4f \u00b7 (1\u2212p)^(n\u2212k)\n\n"
                "Mean = np, Variance = np(1\u2212p). "
                "For large n and moderate p, converges to Normal by the CLT.", "\U0001f3b2")
            _formula_box("P(X=k) = C(n,k) \u00b7 p\u1d4f \u00b7 (1\u2212p)^(n\u2212k)   \u00b7   mean = np,   var = np(1\u2212p)")
            st.plotly_chart(binomial_explorer(), use_container_width=True)

    # ── 3. Variance & Shape ───────────────────────────────────────────────────
    elif chapter == "variance":
        _academy_header(
            "Variance, Spread & Shape",
            "Variance captures spread. Skewness and kurtosis describe shape \u2014 "
            "things the mean and variance alone can\u2019t tell you.",
        )

        st.markdown("#### Building intuition for variance")
        _concept_box("Why square the deviations?",
            "If we averaged (x\u1d62 \u2212 x\u0304) directly, positive and negative deviations cancel to zero \u2014 useless. "
            "Squaring ensures all deviations contribute positively and penalises "
            "large deviations disproportionately.", "\U0001f914")
        _concept_box("Why divide by n\u22121, not n?",
            "We estimate x\u0304 from the same data, which \u2018uses up\u2019 one degree of freedom. "
            "Dividing by n\u22121 (Bessel\u2019s correction) compensates, making s\u00b2 an unbiased estimator: "
            "E[s\u00b2] = \u03c3\u00b2 on average across many samples.", "📐")

        groups = st.session_state.get("result", {}).get("_groups_raw", [])
        gnames = st.session_state.get("result", {}).get("_group_names", [])
        data_var = np.concatenate(groups) if groups else np.random.default_rng(1).normal(50, 12, 120)
        label_var = (gnames[0] if gnames else "Demo data")
        st.plotly_chart(variance_decomposition(data_var, label_var), use_container_width=True)
        st.caption("Left: raw deviations from the mean (some positive, some negative). "
                   "Right: squared deviations. Variance = mean height of the right bars (adjusted for n\u22121).")

        st.markdown("#### Skewness & Kurtosis")
        _concept_box("Skewness",
            "Measures asymmetry of the distribution.\n"
            "  \u2022 Positive skew = long right tail (most data on left, few very high values)\n"
            "  \u2022 Negative skew = long left tail\n"
            "  \u2022 Normal distribution: skewness = 0\n\n"
            "Income, reaction times, and city populations are typically right-skewed.", "📊")
        _concept_box("Kurtosis",
            "Measures tail heaviness.\n"
            "  \u2022 Normal distribution: kurtosis = 3 (mesokurtic)\n"
            "  \u2022 Heavier tails (more extremes): kurtosis > 3 (leptokurtic)\n"
            "  \u2022 Flatter / lighter tails: kurtosis < 3 (platykurtic)\n\n"
            "Excess kurtosis = kurtosis \u2212 3, so Normal has excess kurtosis = 0.", "📊")
        st.plotly_chart(skewness_kurtosis_visual(), use_container_width=True)

    # ── 4. Estimators ─────────────────────────────────────────────────────────
    elif chapter == "estimators":
        _academy_header(
            "Estimators & Sampling",
            "We never observe a whole population \u2014 only a sample. "
            "An estimator is a formula that computes a guess of a population parameter from a sample. "
            "A good estimator is unbiased and has low variance.",
        )

        _concept_box("Bias",
            "An estimator \u03b8\u0302 is unbiased if E[\u03b8\u0302] = \u03b8.\n\n"
            "  \u2022 x\u0304 is an unbiased estimator of \u03bc\n"
            "  \u2022 s\u00b2 with (n\u22121) is unbiased for \u03c3\u00b2\n"
            "  \u2022 s\u00b2 with (n) systematically underestimates \u03c3\u00b2", "\U0001f3af")
        _concept_box("Variance of an estimator",
            "Even an unbiased estimator varies from sample to sample. "
            "The standard deviation of this sampling distribution is the Standard Error (SE).\n\n"
            "For the sample mean:   SE = \u03c3 / \u221an\n\n"
            "Larger samples \u2192 smaller SE \u2192 more precise estimates.", "📐")
        _concept_box("Bias\u2013Variance tradeoff",
            "MSE = Bias\u00b2 + Variance\n\n"
            "Sometimes accepting a little bias dramatically reduces variance (e.g., ridge regression, shrinkage). "
            "For classical hypothesis tests we use unbiased estimators because we want valid p-values.", "\u2696\ufe0f")

        st.plotly_chart(estimator_bias_variance(), use_container_width=True)
        st.caption("Simulation of 500 repeated samples. Left: x\u0304 is centred on the true mean \u2014 unbiased. "
                   "Right: s\u00b2(\u00f7n) underestimates \u03c3\u00b2; s\u00b2(\u00f7n\u22121) is correctly centred on the true value.")

        st.markdown("#### Standard Error vs Sample Size")
        _concept_box("The square-root law",
            "SE = \u03c3 / \u221an\n\n"
            "Doubling precision requires quadrupling n. "
            "Going from n=10 to n=40 halves the SE. "
            "This is why large studies are needed to detect small effects reliably.", "\U0001f4a1")
        st.plotly_chart(standard_error_vs_n(), use_container_width=True)

    # ── 5. CLT ────────────────────────────────────────────────────────────────
    elif chapter == "clt":
        _academy_header(
            "The Central Limit Theorem (CLT)",
            "Perhaps the most important theorem in statistics. It explains why Normal-based "
            "tests work even when the raw data is not Normal.",
        )

        _concept_box("The theorem",
            "If you draw repeated samples of size n from any population with mean \u03bc and "
            "finite variance \u03c3\u00b2, then as n grows, the distribution of the sample mean x\u0304 "
            "approaches Normal(\u03bc, \u03c3\u00b2/n) regardless of the population\u2019s original shape.\n\n"
            "For most populations, n \u2265 30 is sufficient. "
            "For heavily skewed distributions, larger n may be needed.", "\U0001f3db\ufe0f")
        _formula_box("x\u0304 \u223c Normal(\u03bc, \u03c3\u00b2/n)   as   n \u2192 \u221e")

        pop_choice = st.radio("Population to sample from:",
            ["Exponential (right-skewed)", "Uniform (flat)", "Bimodal (two humps)"],
            horizontal=True)
        pop_map = {"Exponential (right-skewed)": "exponential",
                   "Uniform (flat)": "uniform",
                   "Bimodal (two humps)": "bimodal"}
        st.plotly_chart(central_limit_theorem(pop_map[pop_choice]), use_container_width=True)
        st.caption("Each panel shows 2000 sample means for that n. "
                   "The white overlay is a Normal curve fit. Watch the histogram become bell-shaped as n grows.")

        _concept_box("Why this justifies t-tests on non-Normal data",
            "t-tests compare means. Even if the raw data is skewed, the sampling distribution "
            "of the mean is approximately Normal for n \u2265 30. "
            "That\u2019s the distribution the test statistic is built from \u2014 so the test is valid.", "\U0001f4a1")

    # ── 6. Confidence Intervals ───────────────────────────────────────────────
    elif chapter == "ci":
        _academy_header(
            "Confidence Intervals",
            "A 95% CI gives a range of plausible values for the population parameter. "
            "It does NOT mean \u2018there is a 95% chance the parameter is in this interval\u2019.",
        )

        _concept_box("Construction of a 95% CI for the mean",
            "CI = x\u0304 \u00b1 t*(df) \u00d7 SE\n\n"
            "where t*(df) is the critical value from the t distribution and SE = s/\u221an.\n\n"
            "Width depends on:\n"
            "  \u2022 Sample size (larger n \u2192 narrower)\n"
            "  \u2022 Variability (larger s \u2192 wider)\n"
            "  \u2022 Confidence level (99% CI is wider than 95%)", "📐")

        c1, c2, c3 = st.columns(3)
        with c1:
            conf_level = st.select_slider("Confidence level", [90, 95, 99], value=95)
        with c2:
            n_ci = st.slider("Sample size (n)", 5, 100, 30)
        with c3:
            n_intervals_ci = st.slider("Intervals to simulate", 20, 100, 50)

        st.plotly_chart(
            confidence_interval_simulation(
                true_mean=5.0, true_std=2.0,
                n=n_ci, n_intervals=n_intervals_ci,
                confidence=conf_level / 100,
            ),
            use_container_width=True,
        )
        st.caption("Each bar is one CI from a simulated sample. "
                   "Green = contains the true mean. Red = misses it. "
                   "About (100\u2212confidence)\u0025 should miss.")

        _concept_box("The correct interpretation",
            "\u274c  \u2018There is a 95% probability the true \u03bc lies in this specific interval.\u2019\n\n"
            "\u2705  \u2018This interval was produced by a procedure that captures the true \u03bc 95% of the time.\u2019\n\n"
            "After computing an interval, the parameter is either in it or not \u2014 probability no longer applies. "
            "The 95% refers to the long-run behaviour of the procedure.", "\u26a0\ufe0f")

    # ── 7. p-values ───────────────────────────────────────────────────────────
    elif chapter == "pval":
        _academy_header(
            "p-values & Hypothesis Testing",
            "The p-value is the probability of observing a test statistic at least as extreme as "
            "the one we got, assuming H\u2080 is true. It is NOT the probability that H\u2080 is true.",
        )

        _concept_box("Formal definition",
            "p = P(|T| \u2265 |t_observed| | H\u2080 true)\n\n"
            "A small p means: if there were truly no effect, seeing a result this extreme would be rare. "
            "We then doubt H\u2080 and tentatively accept H\u2081.", "📐")

        _concept_box("What the p-value is NOT",
            "\u274c  The probability that H\u2080 is true\n"
            "\u274c  The probability the result is due to chance\n"
            "\u274c  The probability of making a mistake\n"
            "\u274c  A measure of the size or importance of an effect\n\n"
            "A p-value of 0.04 and a p-value of 0.0001 both lead to the same decision at \u03b1=0.05 \u2014 "
            "but tell you very different things about the evidence.", "\u26a0\ufe0f")

        c1, c2 = st.columns(2)
        with c1:
            obs_t = st.slider("Observed t statistic", -5.0, 5.0, 2.3, 0.1)
        with c2:
            df_pval = st.slider("Degrees of freedom", 3, 200, 28)
        st.plotly_chart(pvalue_intuition(obs_t, df_pval), use_container_width=True)
        st.caption("Red area = p-value. Move the slider to see how the p-value changes as the "
                   "test statistic moves closer to or further from zero.")

        _concept_box("Significance threshold \u03b1",
            "\u03b1 is the maximum p at which we call a result significant.\n"
            "Choosing \u03b1 = 0.05 means: I\u2019m willing to make a Type I error (false alarm) "
            "at most 5\u0025 of the time when H\u2080 is true.\n\n"
            "This is a design choice made BEFORE running the test, not adjusted after seeing the data.", "\U0001f3da\ufe0f")

    # ── 8. Errors & Power ─────────────────────────────────────────────────────
    elif chapter == "errors":
        _academy_header(
            "Type I & Type II Errors, and Statistical Power",
            "Every hypothesis test can make two kinds of mistakes. "
            "Understanding them \u2014 and the power to avoid the second \u2014 is essential for study design.",
        )

        col1, col2 = st.columns(2)
        with col1:
            _concept_box("Type I error (False Positive, \u03b1)",
                "Rejecting H\u2080 when it\u2019s actually true.\n"
                "Probability = \u03b1 (you set this).\n\n"
                "Example: concluding a drug works when it doesn\u2019t.", "\U0001f534")
        with col2:
            _concept_box("Type II error (False Negative, \u03b2)",
                "Failing to reject H\u2080 when H\u2081 is true.\n"
                "Probability = \u03b2.\n\n"
                "Example: concluding a drug doesn\u2019t work when it does.", "\U0001f7e1")

        _concept_box("Statistical Power (1 \u2212 \u03b2)",
            "The probability of correctly detecting a real effect.\n\n"
            "Power increases with:\n"
            "  \u2022 Larger sample size (n)\n"
            "  \u2022 Larger true effect size\n"
            "  \u2022 Larger \u03b1 (but raises Type I error risk)\n"
            "  \u2022 Lower data variability\n\n"
            "Conventional target: power \u2265 0.80 (80\u0025). "
            "Running an underpowered study is wasteful \u2014 you\u2019ll likely miss real effects.", "\u26a1")

        c1, c2, c3 = st.columns(3)
        with c1:
            eff = st.slider("True effect size", 0.2, 3.0, 1.5, 0.1)
        with c2:
            alph = st.select_slider("\u03b1 level", [0.01, 0.05, 0.10], value=0.05)
        with c3:
            n_err = st.slider("Sample size (n)", 5, 200, 30)

        st.plotly_chart(type_error_visual(eff, alph, n_err), use_container_width=True)
        st.caption("Blue = H\u2080 distribution. Green = H\u2081 distribution (true effect). "
                   "Red = Type I error (\u03b1). Orange = Type II error (\u03b2). "
                   "Adjust sliders to feel the tradeoffs live.")

        st.markdown("#### Power table — two-tailed t-test, \u03b1=0.05")
        rows = []
        for d in [0.2, 0.5, 0.8, 1.2]:
            row = {"Cohen\u2019s d": d,
                   "Size": {0.2: "small", 0.5: "medium", 0.8: "large", 1.2: "very large"}[d]}
            for n_pow in [10, 20, 30, 50, 100, 200]:
                se_pow   = 1 / np.sqrt(n_pow / 2)
                crit_pow = sci_norm.ppf(0.975)
                power_val = sci_norm.cdf(d / se_pow - crit_pow)
                cell = f"{power_val:.2f}"
                row[f"n={n_pow}"] = cell
            rows.append(row)
        df_power = pd.DataFrame(rows)
        st.dataframe(df_power, use_container_width=True, hide_index=True)
        st.caption("Values \u2265 0.80 indicate adequate power. "
                   "Small effects (d=0.2) need n>200 to reach 80\u0025 power.")

    elif chapter == "estimators_deep":
        _academy_header(
            "Estimator Navigator",
            "Given data characteristics — frequency, observation type, sample size, and shape — "
            "which estimator fits best? And when can one distribution's tools stand in for another's?",
        )

        # ── 4 dimension selectors ─────────────────────────────────────────────
        st.markdown("#### Describe your data")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            freq_choice = st.selectbox(
                "Event frequency",
                ["Common (>20%)", "Moderate (5–20%)", "Rare (<5%)", "Very rare (<1%)"],
            )
        with c2:
            obs_choice = st.selectbox(
                "Observation type",
                ["Independent draws", "Sequential / time-ordered",
                 "Paired (same subject twice)", "Counts in intervals"],
            )
        with c3:
            n_val = st.slider("Sample size (n)", 2, 500, 30)
        with c4:
            shape_choice = st.selectbox(
                "Data shape",
                ["Unknown / unchecked", "Bell-shaped (Normal)",
                 "Skewed / asymmetric", "Binary (0/1)"],
            )

        # ── Derive flags ──────────────────────────────────────────────────────
        is_rare      = "Rare" in freq_choice or "Very" in freq_choice
        is_very_rare = "Very" in freq_choice
        is_seq       = "Sequential" in obs_choice
        is_paired_   = "Paired" in obs_choice
        is_count     = "Counts" in obs_choice
        is_binary_   = "Binary" in shape_choice
        is_skewed_   = "Skewed" in shape_choice
        is_normal_   = "Bell" in shape_choice
        is_large_    = n_val >= 30
        is_medium_   = 10 <= n_val < 30
        is_small_    = n_val < 10

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # PRIMARY RECOMMENDATION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### Recommended estimator")

        if is_seq:
            _concept_box(
                "AR(1) / ARMA — Autoregressive estimator",
                "Formula:  x\u209c = \u03bc + \u03c6(x\u209c\u208b\u2081 \u2212 \u03bc) + \u03b5\u209c     \u03c6\u0302 via Yule-Walker or MLE\n\n"
                "Sequential observations violate independence. Using x\u0304 on time-series data inflates the "
                "effective sample size because consecutive values carry redundant information \u2014 a dangerous "
                "mistake that produces overconfident intervals.\n\n"
                "The AR(1) model estimates the autocorrelation \u03c6 alongside the mean. Yule-Walker "
                "provides a moment estimator; MLE is more efficient. Check stationarity first with "
                "an ADF test before fitting. If you only need the mean, use x\u0304 with Newey-West "
                "HAC standard errors to correct for autocorrelation without modelling the full dynamics.",
                "\U0001f4c8"
            )
            _concept_box(
                "Why plain x\u0304 fails on sequential data",
                "Var(x\u0304) = \u03c3\u00b2/n assumes independence. With autocorrelation \u03c6, the true variance is "
                "\u03c3\u00b2/n \u00b7 (1+\u03c6)/(1\u2212\u03c6) \u2014 which can be many times larger. "
                "For \u03c6=0.8 the true SE is 3\u00d7 larger than the naive estimate. "
                "The effective sample size is n\u2009\u00b7\u2009(1\u2212\u03c6)/(1+\u03c6).",
                "\u26a0\ufe0f"
            )

        elif is_paired_:
            _concept_box(
                "Mean of paired differences (d\u0304)",
                "Formula:  d\u1d62 = x\u2082\u1d62 \u2212 x\u2081\u1d62     d\u0304 = \u03a3d\u1d62/n     CI: d\u0304 \u00b1 t*(n\u22121) \u00b7 s_d/\u221an\n\n"
                "Paired observations share a subject-level baseline. Computing differences removes "
                "this noise, leaving only the change signal. The variance of d\u0304 is:\n\n"
                "Var(d\u0304) = [Var(X\u2082) + Var(X\u2081) \u2212 2\u00b7Cov(X\u2081,X\u2082)] / n\n\n"
                "When within-subject correlation is high, Cov is large, and Var(d\u0304) is far smaller "
                "than comparing two independent groups. This is why clinical trials use paired designs: "
                "the same n gives much higher power.",
                "\U0001f504"
            )
            if not is_large_:
                _concept_box(
                    "Small n paired \u2014 use Wilcoxon signed-rank instead",
                    "With n=" + str(n_val) + " < 30, the CLT on d\u1d62 may not hold. "
                    "The Wilcoxon signed-rank test works on ranks of |d\u1d62| without assuming Normality. "
                    "Hodges-Lehmann pseudo-median (HL = median of all (d\u1d62+d\u2c7c)/2) is the "
                    "matching point estimator with 95.5% asymptotic efficiency.",
                    "\U0001f4d0"
                )

        elif is_count:
            if is_very_rare or is_rare:
                _concept_box(
                    "Negative Binomial MLE \u2014 overdispersed rare counts",
                    "\u03bc\u0302 = x\u0304     r\u0302 = x\u0304\u00b2 / (s\u00b2 \u2212 x\u0304)     (requires s\u00b2 > x\u0304)\n\n"
                    "Rare counts often show overdispersion: Var >> Mean, because events cluster "
                    "(contagion, heterogeneity across units). Poisson assumes Mean = Variance \u2014 "
                    "if violated, it underestimates uncertainty and produces overconfident p-values.\n\n"
                    "Negative Binomial adds a dispersion parameter r, fitting mean and variance independently. "
                    "Diagnostic: compute dispersion index s\u00b2/x\u0304. If \u2248 1, Poisson is fine. "
                    "If > 1.5, use NB. Use a likelihood ratio test to decide formally.",
                    "\U0001f9ee"
                )
            else:
                _concept_box(
                    "Poisson MLE: \u03bb\u0302 = x\u0304",
                    "\u03bb\u0302 = x\u0304     Var(\u03bb\u0302) = \u03bb/n     Check: s\u00b2/x\u0304 \u2248 1\n\n"
                    "Counts of independent events in fixed intervals follow Poisson(\u03bb) "
                    "when mean \u2248 variance. The MLE is the sample mean \u2014 it is also the "
                    "UMVUE (uniformly minimum variance unbiased estimator) by the "
                    "Lehmann-Scheff\u00e9 theorem: no unbiased estimator exists with smaller variance.\n\n"
                    "Key diagnostic: dispersion index s\u00b2/x\u0304. If it exceeds 1.5, "
                    "switch to Negative Binomial.",
                    "\U0001f4ca"
                )

        elif is_binary_:
            if is_very_rare:
                _concept_box(
                    "Exact Binomial CI (Clopper-Pearson)",
                    "p\u0302 = k/n     CI: [Beta(\u03b1/2; k, n\u2212k+1),  Beta(1\u2212\u03b1/2; k+1, n\u2212k)]\n\n"
                    "Very rare events + small n: the Wald interval (p\u0302 \u00b1 z\u221a(p\u0302(1\u2212p\u0302)/n)) "
                    "can produce negative lower bounds and severe undercoverage. "
                    "The exact method inverts the Binomial CDF using Beta quantiles. "
                    "It is conservative (actual coverage \u2265 nominal) but always valid.",
                    "\U0001f3af"
                )
            elif is_rare:
                _concept_box(
                    "Wilson score interval for p",
                    "p\u0302 = k/n\n"
                    "CI centre: (p\u0302 + z\u00b2/2n) / (1 + z\u00b2/n)\n"
                    "CI half-width: z\u221a(p\u0302(1\u2212p\u0302)/n + z\u00b2/4n\u00b2) / (1 + z\u00b2/n)\n\n"
                    "Wald fails for rare events (p < 0.05) even at n=200 \u2014 it has "
                    "systematic undercoverage near p\u22480. Wilson recentres the interval "
                    "toward 0.5 and achieves near-nominal coverage across [0,1]. "
                    "Preferred by the American Statistical Association for proportions.",
                    "\U0001f3af"
                )
            else:
                _concept_box(
                    "Sample proportion p\u0302 + Wilson CI",
                    "p\u0302 = k/n     Wald CI: p\u0302 \u00b1 z\u221a(p\u0302(1\u2212p\u0302)/n)\n\n"
                    "With np\u0302 \u2265 5 and n(1\u2212p\u0302) \u2265 5, the Normal approximation holds and Wald performs well. "
                    "Wilson is still slightly preferable \u2014 it reduces to Wald asymptotically "
                    "but has better coverage at moderate n. "
                    "p\u0302 is the MLE and the UMVUE for the Binomial success probability.",
                    "\U0001f3af"
                )

        elif is_skewed_ and is_small_:
            _concept_box(
                "Hodges-Lehmann estimator + Wilcoxon CI",
                "HL = median\u007b(x\u1d62 + x\u2c7c)/2 : all i \u2264 j\u007d\n\n"
                "Small n=" + str(n_val) + " and skewed data: the mean is pulled toward the tail and "
                "the CLT has not activated. Hodges-Lehmann estimates the pseudo-median robustly "
                "without any distributional assumption.\n\n"
                "Asymptotic relative efficiency (ARE) vs. mean = 95.5% under Normality \u2014 "
                "almost as good as the mean when data is Normal, and much better under skew. "
                "The CI is obtained by inverting the Wilcoxon signed-rank test, giving exact coverage.",
                "\U0001f4d0"
            )

        elif is_skewed_ and is_large_:
            _concept_box(
                "Sample Median + Bootstrap CI",
                "x\u0303 = middle value     CI via B = 1000 resamples\n\n"
                "Skewed data: the mean is pulled toward the tail and is not representative. "
                "The median has 63.7% ARE vs. the mean for Normal data but dominates under skew.\n\n"
                "With n=" + str(n_val) + " \u2265 30, the non-parametric bootstrap gives a valid CI "
                "without any distributional assumption \u2014 resample with replacement B times, "
                "compute the median each time, take the 2.5th and 97.5th percentiles.",
                "\U0001f4c9"
            )

        else:
            if is_large_:
                _concept_box(
                    "Sample Mean x\u0304 + t-interval (CLT fully valid)",
                    "x\u0304 = \u03a3x\u1d62/n     CI: x\u0304 \u00b1 t*(n\u22121) \u00b7 s/\u221an\n\n"
                    "n=" + str(n_val) + " \u2265 30: the CLT guarantees the sampling distribution of x\u0304 "
                    "is approximately Normal regardless of the population's shape.\n\n"
                    "x\u0304 is the MVUE (minimum variance unbiased estimator) for the Normal "
                    "distribution \u2014 proven by Rao-Blackwell. Bessel's correction (n\u22121) makes s\u00b2 "
                    "unbiased for \u03c3\u00b2. At n=" + str(n_val) + " the correction is " +
                    f"{100/(n_val-1):.1f}%.",
                    "\U0001f4cf"
                )
            elif is_medium_:
                _concept_box(
                    "Sample Mean x\u0304 + t-interval (CLT partial \u2014 check shape)",
                    "x\u0304 = \u03a3x\u1d62/n     CI: x\u0304 \u00b1 t*(n\u22121) \u00b7 s/\u221an\n\n"
                    "n=" + str(n_val) + " is in the 10\u201329 range. The CLT is partially active: "
                    "x\u0304 is approximately Normal for symmetric data but can be skewed for "
                    "heavily non-Normal populations.\n\n"
                    "Recommended: check shape with a QQ plot before trusting the CI. "
                    "If data is skewed, consider the Hodges-Lehmann estimator or bootstrap instead.",
                    "\U0001f914"
                )
            else:
                _concept_box(
                    "Sample Mean x\u0304 + t-interval (use with caution \u2014 very small n)",
                    "x\u0304 = \u03a3x\u1d62/n     CI: x\u0304 \u00b1 t*(n\u22121) \u00b7 s/\u221an\n\n"
                    "n=" + str(n_val) + " is very small. The CLT has not activated. The t-interval "
                    "assumes Normality of the raw data \u2014 not just the sampling distribution.\n\n"
                    "Strongly recommended: use the Hodges-Lehmann estimator + Wilcoxon CI instead. "
                    "It requires no distributional assumption and has 95.5% ARE under Normality.",
                    "\u26a0\ufe0f"
                )

        # ══════════════════════════════════════════════════════════════════════
        # RELIABILITY CHART
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### Estimator reliability as n grows")
        st.caption("How appropriate each estimator becomes with more data, for your current settings.")

        import plotly.graph_objects as go
        # np already imported at module level

        ns_plot = [2, 5, 8, 10, 15, 20, 30, 50, 100, 200, 500]
        x_idx    = list(range(11))
        x_labels = [f"n={n}" for n in ns_plot]

        def clamp(v): return min(100, max(0, v))

        mean_r   = [clamp(30 + n*0.7 if is_skewed_ else 15 + n*1.8) for n in ns_plot]
        t_r      = [clamp(8 + n*2.0) for n in ns_plot]
        median_r = [clamp(55 + n*0.45 if is_skewed_ else 25 + n*0.9) for n in ns_plot]
        boot_r   = [clamp(75 + n*0.08 if n >= 20 else n*3.2) for n in ns_plot]
        exact_r  = [88] * len(ns_plot)

        fig_rel = go.Figure()
        for label, vals, color, dash in [
            ("Sample mean (x\u0304)",  mean_r,   "#4F8EF7", "solid"),
            ("t-interval",            t_r,      "#34D399", "solid"),
            ("Median",                median_r, "#A259FF", "solid"),
            ("Bootstrap CI",          boot_r,   "#FFB347", "solid"),
            ("Exact methods",         exact_r,  "#FF4B6E", "dash"),
        ]:
            fig_rel.add_trace(go.Scatter(
                x=x_idx, y=vals,
                mode="lines+markers", name=label,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=5),
            ))

        # Mark current n
        n_mark = n_val
        closest_idx = min(range(len(ns_plot)), key=lambda ii: abs(ns_plot[ii] - n_val))
        fig_rel.add_vline(
            x=closest_idx,
            line=dict(color="#FFB347", width=2, dash="dot"),
            annotation_text=f"your n={n_val}",
            annotation_font=dict(color="#FFB347", size=10),
            annotation_position="top right",
        )
        fig_rel.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,18,30,0.6)",
            height=280, margin=dict(l=45, r=20, t=30, b=40),
            font=dict(family="IBM Plex Mono", color="#8899b0"),
            legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            yaxis_title="Appropriateness (%)",
        )
        fig_rel.update_xaxes(
            gridcolor="rgba(255,255,255,.06)", zeroline=False,
            tickmode="array", tickvals=x_idx, ticktext=x_labels,
            tickfont=dict(size=9),
        )
        fig_rel.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False,
                             range=[0, 105],
                             ticksuffix="%")
        st.plotly_chart(fig_rel, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════
        # SUBSTITUTION TABLE — full 8-entry table, condition met badge
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### When one distribution can substitute for another")
        st.caption(
            "Each row shows the substitution, the mathematical condition that makes it valid, "
            "the justification, and whether it applies to your current n and scenario."
        )

        FULL_SUBS = [
            {
                "from_": "Binomial(n, p)",
                "to":    "Normal(np,\u00a0np(1\u2212p))",
                "cond":  "n\u00b7p \u2265 5  AND  n\u00b7(1\u2212p) \u2265 5",
                "why":   "CLT: sum of n Bernoulli(p) trials \u2192 Normal. Good for 0.1 < p < 0.9. "
                         "Apply continuity correction (+0.5) for integer comparisons.",
                "limit": "Fails when p \u2248 0 or p \u2248 1 even for large n",
                "check": lambda: is_binary_ and not is_very_rare and n_val * 0.1 >= 5,
            },
            {
                "from_": "Binomial(n, p)",
                "to":    "Poisson(\u03bb = np)",
                "cond":  "n \u2192 \u221e,  p \u2192 0,  n\u00b7p = \u03bb fixed",
                "why":   "Large n + tiny p: most trials yield 0 \u2014 looks like rare random arrivals. "
                         "Poisson is simpler (1 parameter). Classic use: defect counts, epidemiology.",
                "limit": "Only valid when both n and 1/p are simultaneously large",
                "check": lambda: is_binary_ and is_rare and n_val >= 30,
            },
            {
                "from_": "Poisson(\u03bb)",
                "to":    "Normal(\u03bb,\u00a0\u03bb)",
                "cond":  "\u03bb \u2265 10  (\u03bb \u2265 30 for tail accuracy)",
                "why":   "For large \u03bb, Poisson \u2248 Normal(\u03bb,\u03bb). Enables z-tests and closed-form CIs. "
                         "The \u221ax transformation stabilises variance first.",
                "limit": "Tails underestimated for \u03bb < 10",
                "check": lambda: is_count and not is_rare,
            },
            {
                "from_": "t(df)",
                "to":    "Normal(0, 1)",
                "cond":  "df \u2265 30  (df \u2265 120 for tail accuracy at \u03b1 = 0.001)",
                "why":   "t(df) \u2192 N(0,1) as df \u2192 \u221e. For df \u2265 30 the difference in critical values "
                         "is < 5% at \u03b1 = 0.05. t is needed when \u03c3 is estimated from data.",
                "limit": "At \u03b1 = 0.001, convergence needs df > 200",
                "check": lambda: not is_binary_ and not is_count and n_val >= 30,
            },
            {
                "from_": "Chi-squared(k)",
                "to":    "Normal(k,\u00a02k)",
                "cond":  "k \u2265 30",
                "why":   "\u03c7\u00b2(k) is a sum of k squared Normals \u2014 CLT applies. "
                         "Wilson-Hilferty cube-root gives even better Normal approximation for moderate k.",
                "limit": "Tails (p < 0.01) need larger k",
                "check": lambda: n_val >= 30,
            },
            {
                "from_": "Hypergeometric",
                "to":    "Binomial(n,\u00a0K/N)",
                "cond":  "n/N \u2264 0.05  (sample < 5% of population)",
                "why":   "Small sample relative to population: each draw barely changes remaining proportion "
                         "\u2014 effectively sampling with replacement. Binomial is simpler and sufficient.",
                "limit": "If n/N > 10%, apply finite population correction \u221a((N\u2212n)/(N\u22121))",
                "check": lambda: is_binary_ and is_large_,
            },
            {
                "from_": "Negative Binomial",
                "to":    "Poisson(\u03bb)",
                "cond":  "Dispersion index s\u00b2/x\u0304 \u2248 1",
                "why":   "NB reduces to Poisson as dispersion r \u2192 \u221e. "
                         "If variance/mean ratio is close to 1, Poisson is sufficient. "
                         "Use likelihood ratio test to decide formally.",
                "limit": "If dispersion index > 1.5, NB is clearly needed",
                "check": lambda: is_count and not is_rare,
            },
            {
                "from_": "Beta(\u03b1, \u03b2)",
                "to":    "Normal",
                "cond":  "\u03b1 \u2265 5  AND  \u03b2 \u2265 5",
                "why":   "Beta is conjugate prior for proportions (bounded [0,1]). "
                         "Large shape parameters \u2192 approximately Normal centred at \u03b1/(\u03b1+\u03b2). "
                         "Useful in Bayesian posterior reporting and A/B test analysis.",
                "limit": "Near boundaries (\u03b1 or \u03b2 < 2) Normal approximation fails badly",
                "check": lambda: is_binary_ and is_large_,
            },
        ]

        for s in FULL_SUBS:
            try:
                valid = s["check"]()
            except Exception:
                valid = False

            # Pre-compute all dynamic values — no logic inside the HTML string
            border_color  = "#34D399" if valid else "rgba(255,255,255,.12)"
            badge_color   = "#34D399" if valid else "#8899b0"
            badge_icon    = "✅" if valid else "ℹ️"
            badge_text    = "Valid for your scenario" if valid else "Conditions not met here"
            from_label    = s["from_"]
            to_label      = s["to"]
            cond_text     = s["cond"]
            why_text      = s["why"]
            limit_text    = s["limit"]

            html_card = (
                f'<div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.08);'
                f'border-left:3px solid {border_color};'
                f'border-radius:10px;padding:1rem 1.2rem;margin-bottom:.7rem">'
                f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.55rem;flex-wrap:wrap">'
                f'<span style="font-family:\'IBM Plex Mono\';font-size:.75rem;'
                f'background:rgba(255,75,110,.1);border:1px solid rgba(255,75,110,.25);'
                f'color:#FF4B6E;padding:.15rem .5rem;border-radius:4px">{from_label}</span>'
                f'<span style="color:#8899b0;font-size:.85rem">→</span>'
                f'<span style="font-family:\'IBM Plex Mono\';font-size:.75rem;'
                f'background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.25);'
                f'color:#4F8EF7;padding:.15rem .5rem;border-radius:4px">{to_label}</span>'
                f'<span style="font-family:\'IBM Plex Mono\';font-size:.7rem;'
                f'color:{badge_color};margin-left:auto">{badge_icon} {badge_text}</span>'
                f'</div>'
                f'<div style="font-family:\'IBM Plex Mono\';font-size:.72rem;'
                f'color:#FFB347;margin-bottom:.4rem">Condition: {cond_text}</div>'
                f'<div style="font-size:.8rem;color:#c8d0e0;line-height:1.6;margin-bottom:.3rem">{why_text}</div>'
                f'<div style="font-size:.75rem;color:rgba(255,179,71,.7);font-style:italic">⚠ {limit_text}</div>'
                f'</div>'
            )
            st.markdown(html_card, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # DATA REGIME SUMMARY LEGEND
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### Data regime reference")

        regimes = [
            ("#4F8EF7", "Large n (\u226530)",
             "CLT valid. Parametric estimators appropriate even for non-Normal data. z and t intervals reliable."),
            ("#A259FF", "Medium n (10\u201329)",
             "CLT partial. Use t-distribution. Verify shape with QQ plot. Bootstrap viable."),
            ("#FFB347", "Small n (<10)",
             "CLT not valid. Non-parametric or exact methods required. All estimates uncertain."),
            ("#34D399", "Rare events (p<5%)",
             "Wald interval fails. Use Wilson (binary) or exact Binomial. Poisson MLE for counts."),
            ("#FF4B6E", "Sequential data",
             "Independence violated. Autocorrelation inflates effective n. ARMA or HAC needed."),
            ("#F472B6", "Skewed / heavy tails",
             "Mean pulled by extremes. Median or Hodges-Lehmann more representative."),
        ]

        cols_reg = st.columns(3)
        for i, (r_color, r_name, r_desc) in enumerate(regimes):
            with cols_reg[i % 3]:
                html_regime = (
                    f'<div style="background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.07);'
                    f'border-radius:10px;padding:.85rem 1rem;margin-bottom:.7rem;'
                    f'display:flex;align-items:flex-start;gap:.6rem">'
                    f'<div style="width:9px;height:9px;border-radius:50%;background:{r_color};'
                    f'flex-shrink:0;margin-top:4px"></div>'
                    f'<div>'
                    f'<div style="font-size:.82rem;font-weight:600;color:#e8eef8;margin-bottom:.18rem">{r_name}</div>'
                    f'<div style="font-size:.74rem;color:#8899b0;line-height:1.5">{r_desc}</div>'
                    f'</div></div>'
                )
                st.markdown(html_regime, unsafe_allow_html=True)

