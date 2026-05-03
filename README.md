# StatSense · Statistical Hypothesis Testing

> Answer three plain-English questions about your data and get a rigorous,
> jargon-free statistical analysis — with visualisations, effect sizes,
> assumption checks, and full theoretical explanations.

---

## Table of contents

1. [What the app does](#what-the-app-does)
2. [How the wizard works — the decision logic](#how-the-wizard-works)
3. [Complete decision table](#complete-decision-table)
4. [Assumption checks & automatic fallbacks](#assumption-checks--automatic-fallbacks)
5. [The Statistics Academy](#the-statistics-academy)
6. [Running with Docker on Windows](#running-with-docker-on-windows)
7. [Diagnosing problems](#diagnosing-problems)
8. [Project structure](#project-structure)
9. [Extending the app](#extending-the-app)

---

## What the app does

StatSense guides a non-statistician through a 3-question wizard, selects the
correct statistical test automatically, runs assumption checks, falls back to
safer non-parametric tests when needed, and presents results in plain English
alongside interactive visualisations.

The user never needs to know the name of a test. The technical name is shown
as a small badge *after* the result, for those who want it.

---

## How the wizard works

The wizard asks three questions in sequence. Your answers are looked up in a
decision table and map to exactly one test. Below is every path explained —
what the app selects, and *why* that choice is statistically correct.

---

### Question 1 — What are you trying to find out?

---

#### Answer: "Are two groups different from each other?"

You have two distinct sets of observations and want to know whether a numeric
outcome differs between them, or whether the distribution of a categorical
outcome differs.

**Why this matters:** The key design question is whether the two groups are
*independent* (separate people) or *paired* (same people measured twice).
This changes the test entirely — pairing removes individual-level noise and
increases sensitivity.

---

#### Answer: "Are three or more groups different?"

You have three or more conditions or categories and want to know if at least
one group differs from the others on some outcome.

**Why not just run multiple t-tests?** Running k(k−1)/2 pairwise t-tests
inflates the false-positive rate (Type I error). ANOVA controls this by
testing all groups simultaneously in a single F-ratio.

---

#### Answer: "Did the same group change over time?"

You measured the same subjects at two or more time points and want to know
whether the outcome changed.

**Why it is treated as paired:** Because the same individual appears in both
measurements, observations are not independent. The paired design subtracts
out each person's baseline level, leaving only the *change* — which is the
signal you actually care about. This is equivalent to selecting "same
people/items measured more than once" in Q2.

---

#### Answer: "Is there a relationship between two things?"

You have two variables measured on the same subjects and want to know whether
they move together — as one increases, does the other tend to increase (or
decrease)?

**Why Q2 is skipped:** There is no "same vs. different" distinction for
correlation — you always have paired (x, y) observations on the same subjects.
The only choice is whether both variables are numeric or categorical, which
determines whether correlation or association is the right framework.

---

### Question 2 — Are these the same people/items or different ones?

*(Only asked for "two groups" and "three+ groups" goals)*

---

#### Answer: "Different groups of people/items (no overlap)"

The two groups are completely independent — no subject appears in both.
Classic examples: treatment vs. control, men vs. women, city A vs. city B.

**Statistical consequence:** Observations between groups are independent.
Tests exploit the *between-group* variance relative to the *within-group*
variance (noise). Sample sizes in the two groups can differ.

**→ App selects:** Independent samples t-test (or Mann-Whitney U if
assumptions fail) for two groups; one-way ANOVA (or Kruskal-Wallis) for
three or more.

---

#### Answer: "The same people/items measured more than once"

Each subject provides one observation per condition. Classic examples:
before/after a treatment, left eye vs. right eye, the same machine tested
under two settings.

**Statistical consequence:** Observations within a subject are correlated —
this correlation is *information*, not noise. The paired design leverages it
by computing per-subject *differences* and testing whether those differences
are centred on zero. This is more powerful than an independent test when
within-subject correlation is high.

**→ App selects:** Paired samples t-test (or Wilcoxon signed-rank) for two
measurements; repeated measures ANOVA for three or more time points.

---

### Question 3 — What kind of values do you have?

---

#### Answer: "Numbers (like height, score, temperature, time)"

Your outcome variable is continuous or at least interval-scaled — the
differences between values are meaningful.

**Statistical consequence:** Parametric tests (t-test, ANOVA, Pearson r) are
designed for numeric data. They model means and assume (to varying degrees)
that values follow a Normal distribution. Assumption checks are run
automatically, and non-parametric rank-based equivalents are used if the data
deviates too far from Normality.

---

#### Answer: "Categories (like yes/no, colour, type, grade)"

Your outcome is a nominal or ordinal category — values represent membership
in a group, not a quantity.

**Statistical consequence:** Parametric tests cannot be applied because there
is no meaningful mean of "red, green, blue". Instead, the app uses
count-based tests (chi-square, McNemar) that work with frequency tables.
No Normality assumption is needed.

---

#### Answer: "One of each (one number column, one category)"

Typically arises in the relationship goal — one variable is numeric (e.g.,
income) and one is categorical (e.g., education level). The app treats this
as a numeric relationship and uses Pearson/Spearman correlation.

**Note:** For a more rigorous group-comparison approach in this case,
re-run with goal = "Are the groups different?" instead.

---

## Complete decision table

The three answers combine into a single lookup key. The table below shows
every reachable path, the test selected, and the non-parametric fallback
activated automatically when the Normality or sample-size assumption is
violated.

| Goal | Same / Different | Data type | Primary test | Auto-fallback | Why this test |
|------|-----------------|-----------|--------------|---------------|---------------|
| 2 groups | Different | Numeric | **Independent t-test** | Mann-Whitney U | Compares means of two independent groups via t-ratio. Welch correction applied automatically if variances are unequal. |
| 2 groups | Same | Numeric | **Paired t-test** | Wilcoxon signed-rank | Tests whether the mean of within-subject differences is zero. More powerful than independent t when within-subject correlation is high. |
| 2 groups | Different | Categorical | **Chi-square independence** | *(non-parametric by default)* | Compares observed category counts to expected counts under independence. |
| 2 groups | Same | Categorical | **McNemar test** | *(non-parametric by default)* | Tests whether paired yes/no responses changed direction — checks whether discordant pairs are balanced. |
| 3+ groups | Different | Numeric | **One-way ANOVA** | Kruskal-Wallis | F-ratio = between-group variance / within-group variance. Rejects H₀ if signal-to-noise is implausibly high. |
| 3+ groups | Same | Numeric | **Repeated measures ANOVA** | *(planned)* | Like ANOVA but accounts for the correlation structure of repeated observations on the same subjects. |
| 3+ groups | Different | Categorical | **Chi-square independence** | *(non-parametric by default)* | Same as 2-group categorical, extended to k × m contingency tables. |
| Over time | Same | Numeric | **Paired t-test** | Wilcoxon signed-rank | Time-point 1 = "before", time-point 2 = "after". Identical to same-people numeric path. |
| Over time | Same | Categorical | **McNemar test** | *(non-parametric by default)* | Tracks whether yes/no classifications changed between two time points. |
| Relationship | — | Numeric | **Pearson correlation** | Spearman rank correlation | Tests r ≠ 0. r² gives the proportion of variance in Y explained by X. |
| Relationship | — | Categorical | **Chi-square independence** | *(non-parametric by default)* | Tests whether knowing one category gives information about the other. Cramér's V measures strength. |
| Relationship | — | Mixed | **Pearson correlation** | Spearman rank correlation | Treated as numeric; fallback to Spearman if Normality fails. |

---

## Assumption checks & automatic fallbacks

After the wizard, the app automatically checks three conditions before
running any parametric test.

### 1. Normality — Shapiro-Wilk test

Shapiro-Wilk is run on each group separately (p < 0.05 = flag). For n > 5000,
D'Agostino-Pearson is used instead (Shapiro-Wilk becomes unreliable at very
large n because it flags trivial deviations as significant).

**If Normality fails:** the parametric test is silently replaced by its
non-parametric equivalent. The user sees:
*"Your data has an unusual shape — we switched to a safer test automatically."*

**Why not always use non-parametric?** Parametric tests are more powerful
when their assumptions hold — they detect smaller effects with the same sample
size. Using non-parametric by default unnecessarily wastes statistical power.

### 2. Equal variances — Levene's test

Levene's test checks whether groups have similar spread (p < 0.05 = flag).
This matters for the independent t-test: if variances are unequal, the
pooled-variance formula is biased.

**If variances are unequal:** Welch's t-test is used instead of Student's
t-test. Welch's version adjusts the degrees of freedom to compensate. The
user sees: *"Groups have very different spreads — this has been accounted for."*

For ANOVA with unequal variances, the fallback is Kruskal-Wallis.

### 3. Sample size warning

If the smallest group has n < 30, a warning is shown. If n < 10, results are
labelled "very preliminary." Small samples trigger the non-parametric fallback
because rank-based tests make fewer distributional assumptions and are more
reliable before the Central Limit Theorem fully applies.

### The auto-switch banner

When the app switches from parametric to non-parametric, a blue banner reads:
*"Your data didn't fully meet the assumptions for the parametric test, so we
automatically switched to a safer non-parametric equivalent. The
interpretation is the same — just more robust."*

---

## The Statistics Academy

Accessible from the **🎓 Open Statistics Academy** button at the bottom of
any results page. Eight chapters, each combining concept explanations, formulas,
and live interactive Plotly figures.

| Chapter | What you learn | Key visualisation |
|---------|---------------|-------------------|
| 1 · Descriptive Statistics | Mean, median, variance, SD, the empirical rule | Histogram + KDE on your actual data with ±1σ and ±2σ bands |
| 2 · Distributions & Parameters | Normal, t, χ², F, Binomial — what each parameter controls | Four overlapping curves per distribution showing the effect of changing μ, σ, or df |
| 3 · Variance, Spread & Shape | Why we square deviations, Bessel's correction, skewness, kurtosis | Deviation lollipop chart + squared-deviation bars building variance bar by bar |
| 4 · Estimators & Sampling | Bias, variance of an estimator, MSE, standard error, the square-root law | 500-sample simulation proving x̄ is unbiased and s²(÷n) underestimates σ² |
| 5 · Central Limit Theorem | Why Normal-based tests work on non-Normal data | Interactive: choose Exponential / Uniform / Bimodal; watch n=1,5,20,50 converge to Normal |
| 6 · Confidence Intervals | What "95% confident" really means; the correct vs. incorrect interpretation | Live simulation of 50 CIs — green = contains true mean, red = misses; sliders for n and level |
| 7 · p-values | Formal definition, what p is NOT, choosing α before seeing data | Interactive t-distribution — drag the test statistic and watch the p-value area change in real time |
| 8 · Type I & II Errors, Power | False positives, missed effects, power analysis, minimum sample size | Two overlapping H₀/H₁ distributions with live α/β/n sliders + full power table across d and n |

---

## Deploying to Streamlit Community Cloud

### Prerequisites

- A free account at https://streamlit.io/cloud
- The project pushed to a **public GitHub repository** (or a private repo on a paid plan)

### Steps

1. Push the project to GitHub — the root of the repo must contain `app.py`
2. Go to https://share.streamlit.io → **New app**
3. Select your repository, branch, and set **Main file path** to `app.py`
4. Click **Deploy** — Streamlit Cloud reads `requirements.txt` and `.python-version` automatically

The app will be live at `https://<your-app-name>.streamlit.app` in ~2 minutes.

### What the Cloud config files do

| File | Purpose |
|------|---------|
| `requirements.txt` | Lists the 5 packages to install — only what the app actually imports |
| `.python-version` | Pins Python 3.11 — avoids type-hint issues on older Cloud defaults |
| `.streamlit/config.toml` | Sets dark theme, disables CORS warnings, turns off telemetry |

### Important: do NOT push these files to GitHub

The following are for local Docker use only — add them to `.gitignore`:

```
Dockerfile
docker-compose.yml
.dockerignore
```

---

## Running with Docker on Windows (local development)

### Prerequisites (one-time setup)

1. **Install Docker Desktop for Windows**
   - Download: https://www.docker.com/products/docker-desktop/
   - Run the installer and restart your PC when prompted
   - After restart, look for the whale icon in the taskbar — ready when it
     says **"Engine running"**

2. **WSL 2** — Docker Desktop installs and enables this automatically.
   If prompted during installation, click "Install" to allow it.

### Starting the app

Open **PowerShell** or **Command Prompt**, navigate to the project folder:

```powershell
cd path\to\hypothesis_app
docker compose up --build
```

Wait for:

```
You can now view your Streamlit app in your browser.
URL: http://0.0.0.0:8501
```

Then open **http://localhost:8501** in any browser.

### Stopping the app

```powershell
docker compose down
```

Or press `Ctrl+C` in the terminal.

### Alternative — plain Docker (no Compose)

```powershell
docker build -t statsense .
docker run -p 8501:8501 statsense
```

### Live editing without rebuilding

The `volumes` entry in `docker-compose.yml` maps your local folder into the
container:

```yaml
volumes:
  - .:/app
```

Edit any `.py` file → save → refresh the browser. Streamlit auto-reloads.
A full `--build` is only needed after changing `requirements.txt`.

### Changing the port

Edit `docker-compose.yml`:

```yaml
ports:
  - "8502:8501"   # change the left number (host port) only
```

Then open **http://localhost:8502**.

---

## Diagnosing problems

Run the self-diagnostic script — it checks every dependency, every import,
every figure, and the full test pipeline without needing the app to be running:

```powershell
# Inside Docker (no Python needed on host)
docker compose run --rm statsense python diagnose.py

# With local Python
python diagnose.py
```

Output example (all passing):

```
[1] Python environment
  ✅  Python >= 3.10

[2] Required packages
  ✅  import streamlit       version: 1.32.0
  ✅  import numpy           version: 1.26.4
  ...

[7] Figure generation
  ✅  distribution_plot: t
  ✅  diagnostic_plots: all 8 figures
  ✅  academic figures: all 16 figures

[8] End-to-end pipeline
  ✅  full pipeline: 5 test types

SUMMARY: 35 passed, 0 failed
```

### Common problems and fixes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `docker: command not found` | Docker Desktop not running | Open Docker Desktop, wait for "Engine running" |
| `Cannot connect to the Docker daemon` | Same | Same fix |
| Port 8501 already in use | Another service on that port | Change left port in `docker-compose.yml` to e.g. `8502` |
| `ModuleNotFoundError` | Dependencies not installed | Run `docker compose up --build` to force reinstall |
| `SyntaxError` in a `.py` file | Corrupted file | Re-extract the zip |
| Blank page in browser | Browser cache | Hard-refresh: `Ctrl+Shift+R` |
| Figures missing | Plotly version mismatch | `docker compose up --build` to reinstall |
| "Cannot connect" right after start | Streamlit still booting | Wait 15 s and refresh |

To save full Docker logs for sharing:

```powershell
docker compose up --build 2>&1 | Tee-Object -FilePath docker_log.txt
```

---

## Project structure

```
hypothesis_app/
│
├── app.py                      ← Streamlit entry point, page layout, CSS
├── diagnose.py                 ← Self-diagnostic script
├── Dockerfile                  ← Python 3.11-slim image
├── docker-compose.yml          ← One-command start with live-reload volume
├── requirements.txt
│
├── core/
│   ├── data_loader.py          ← CSV upload, manual entry, 5 built-in datasets
│   ├── wizard.py               ← 3-question flow → test lookup table
│   ├── assumption_checks.py    ← Shapiro-Wilk, Levene, sample-size checks
│   ├── parametric.py           ← Independent t, paired t, one-way ANOVA, Pearson r
│   ├── nonparametric.py        ← Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Chi², Spearman
│   ├── test_runner.py          ← Column selection UI + test dispatch
│   └── interpreter.py          ← Plain-English verdict generator
│
├── plots/
│   ├── distribution_plot.py    ← t / χ² / F / Normal curves with rejection zones
│   └── diagnostic_plots.py     ← Boxplots, histograms, QQ plots, scatter, heatmaps, effect gauge
│
└── theory/
    ├── cards.py                ← Per-test theory cards (H₀/H₁, assumptions, errors, intuition)
    ├── academic.py             ← All Academy Plotly figures (16 interactive charts)
    └── academy_page.py         ← Streamlit layout for the 8-chapter Academy page
```

---

## Extending the app

### Adding a new statistical test

1. **Implement** the test function in `core/parametric.py` or
   `core/nonparametric.py`. Return the standardised dict with keys:
   `test_id, stat, stat_label, p, effect_size, effect_label,
   effect_size_word, ci, df, reject, alpha, tail, distribution`.

2. **Register** the answer combination in `core/wizard.py` under `TEST_MAP`.

3. **Handle column selection** in `core/test_runner.py` under
   `select_columns()` and dispatch the test in `run_test()`.

4. **Add interpretation** in `core/interpreter.py` under the `if reject:`
   and `else:` branches.

5. **Add a theory card** in `theory/cards.py` under `CARDS`.

### Adding a new Academy chapter

Add an entry to `CHAPTERS` in `theory/academy_page.py` and implement the
corresponding `elif chapter == "your_key":` block. Use `_academy_header()`,
`_concept_box()`, `_formula_box()`, then call figures from `theory/academic.py`.
Add any new figures to `theory/academic.py` following the `_L()` / `update_xaxes()`
pattern (no axis keys inside `_L()` — pass them via `update_xaxes/yaxes` to
avoid the duplicate-keyword Plotly error).
