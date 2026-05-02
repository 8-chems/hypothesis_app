"""
academy_page.py — renders the Statistics Academy Streamlit page.
Separated from app.py to keep string literals clean.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm as sci_norm

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
