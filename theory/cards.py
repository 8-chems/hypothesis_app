"""
theory_cards.py — educational content for each test:
H₀/H₁, assumptions, Type I/II errors, power, visual intuition descriptions.
"""

CARDS = {
    "independent_t": {
        "name": "Independent samples t-test",
        "emoji": "⚖️",
        "when_to_use": "You have two separate groups and want to know if their **average values** are different.",
        "examples": [
            "Does a drug reduce blood pressure compared to a placebo?",
            "Do students taught Method A score higher than those taught Method B?",
        ],
        "hypotheses": {
            "H0": "The two groups have the **same average** — any observed difference is just random chance.",
            "H1": "The two groups have **different averages** — the difference is real.",
        },
        "assumptions": [
            ("Data shape", "Each group's values should roughly follow a bell curve (normal distribution)."),
            ("Independence", "The groups are completely separate — no person appears in both."),
            ("Similar spread", "Ideally, both groups have similar variance (though Welch's correction handles violations)."),
        ],
        "errors": {
            "type1": "You conclude there **is** a difference when there actually isn't. (False alarm) — controlled by α.",
            "type2": "You conclude there **isn't** a difference when there actually is. (Missed effect) — reduced by larger samples.",
        },
        "intuition": "Imagine weighing two bags of apples. Each bag has slightly different apples — some heavier, some lighter. You want to know if one bag is truly heavier *on average*, or if the difference you see is just because you happened to grab heavier apples by chance.",
        "effect_size": "**Cohen's d** measures how many standard deviations apart the two group means are. d = 0.2 is small, 0.5 is medium, 0.8 is large.",
    },

    "mann_whitney": {
        "name": "Mann-Whitney U test",
        "emoji": "🏅",
        "when_to_use": "Like the t-test, but safer when your data isn't bell-shaped or your samples are small. Compares **ranks** instead of means.",
        "examples": [
            "Does treatment A lead to better recovery scores than treatment B?",
            "Do customers in city X spend more than those in city Y?",
        ],
        "hypotheses": {
            "H0": "The two groups come from the **same distribution** — neither tends to have higher values.",
            "H1": "One group tends to have **systematically higher values** than the other.",
        },
        "assumptions": [
            ("Independence", "The two groups are completely separate."),
            ("Ordinal or numeric data", "Values can be ranked (ordered)."),
            ("Similar shape", "For comparing medians, both distributions should have roughly the same shape."),
        ],
        "errors": {
            "type1": "Concluding one group is higher when it's really not — controlled by α.",
            "type2": "Missing a real difference — less common with this test when distributions are skewed.",
        },
        "intuition": "Instead of comparing averages, imagine listing all values from both groups combined and ranking them 1st, 2nd, 3rd... If one group tends to claim the top spots, we have evidence it's genuinely higher.",
        "effect_size": "**Rank-biserial r**: 0 = no difference, 1 = complete separation between groups.",
    },

    "paired_t": {
        "name": "Paired samples t-test",
        "emoji": "🔄",
        "when_to_use": "You measure the **same people or items twice** (before/after, two conditions) and want to know if the change is real.",
        "examples": [
            "Did students improve after a training course?",
            "Did patients' symptoms decrease after treatment?",
        ],
        "hypotheses": {
            "H0": "The average **change** is zero — any differences are just random fluctuation.",
            "H1": "There is a real average change — the second measurement is genuinely different.",
        },
        "assumptions": [
            ("Paired data", "Each row in group 1 must match a corresponding row in group 2 (same person, same item)."),
            ("Normal differences", "The *differences* (not the raw values) should be roughly bell-shaped."),
            ("Independence", "Each pair is independent of every other pair."),
        ],
        "errors": {
            "type1": "Concluding something changed when it didn't — controlled by α.",
            "type2": "Missing a real improvement — reduced by having more pairs.",
        },
        "intuition": "By measuring the same person twice, you remove individual differences. If Person A was already tall, they'll be tall in both measurements — what matters is whether they grew. This makes the test much more sensitive than comparing two separate groups.",
        "effect_size": "**Cohen's d (paired)**: computed on the differences. Interpretation same as for the t-test.",
    },

    "wilcoxon": {
        "name": "Wilcoxon signed-rank test",
        "emoji": "📐",
        "when_to_use": "Like the paired t-test, but safer when the differences aren't bell-shaped.",
        "examples": [
            "Did pain scores (on a 1-10 scale) decrease after treatment?",
            "Did ranked performance improve in a small group?",
        ],
        "hypotheses": {
            "H0": "The median change is **zero** — no systematic improvement or decline.",
            "H1": "There is a real median change in one direction.",
        },
        "assumptions": [
            ("Paired data", "Same person/item measured twice."),
            ("Symmetric differences", "The distribution of differences should be roughly symmetric (not severely skewed)."),
            ("Ordinal or numeric", "Values can be meaningfully ordered."),
        ],
        "errors": {
            "type1": "False positive — controlled by α.",
            "type2": "Missed real change — more common in very small samples.",
        },
        "intuition": "Rank the *absolute* changes (ignoring direction), then check whether the positive changes or the negative changes claim the bigger ranks. If positive changes consistently dominate, something real happened.",
        "effect_size": "**r (rank correlation)**: effect based on the test statistic. 0.1 = small, 0.3 = medium, 0.5 = large.",
    },

    "anova": {
        "name": "One-way ANOVA",
        "emoji": "🧮",
        "when_to_use": "You have **three or more separate groups** and want to know if at least one is different from the others.",
        "examples": [
            "Does fertilizer type (A, B, C) affect plant height?",
            "Do three different teaching methods lead to different test scores?",
        ],
        "hypotheses": {
            "H0": "All group means are **equal** — any differences are random.",
            "H1": "At least **one group mean is different** from the others.",
        },
        "assumptions": [
            ("Normality", "Each group's values should be roughly bell-shaped."),
            ("Equal variance", "Groups should have similar spread (Levene's test checks this)."),
            ("Independence", "Observations in one group don't influence another."),
        ],
        "errors": {
            "type1": "Concluding groups differ when they don't — controlled by α.",
            "type2": "Missing a real group difference — reduced by larger samples or bigger true effects.",
        },
        "intuition": "ANOVA decomposes the total variation in your data into two buckets: variation *between* groups (signal) and variation *within* groups (noise). If the signal-to-noise ratio (the F statistic) is high enough, we conclude the groups are genuinely different.",
        "effect_size": "**η² (eta-squared)**: proportion of total variation explained by group membership. 0.01 = small, 0.06 = medium, 0.14 = large.",
        "followup": "A significant ANOVA tells you *at least one* group is different. To find *which* ones, you'd run post-hoc tests (Tukey HSD, Bonferroni) — consider this a next step.",
    },

    "kruskal": {
        "name": "Kruskal-Wallis test",
        "emoji": "📊",
        "when_to_use": "Like ANOVA but for non-normal data or ordinal values. Compares distributions across 3+ groups using ranks.",
        "examples": [
            "Do three neighborhoods have different housing prices (skewed data)?",
            "Do satisfaction ratings differ across three service providers?",
        ],
        "hypotheses": {
            "H0": "All groups come from the **same distribution** — no group tends to rank higher.",
            "H1": "At least one group tends to have **systematically different values**.",
        },
        "assumptions": [
            ("Independence", "Groups are separate."),
            ("Ordinal or numeric", "Values can be ranked."),
            ("Similar shape", "For median comparisons, shapes should be similar across groups."),
        ],
        "errors": {
            "type1": "False group difference detected — controlled by α.",
            "type2": "Missed real difference — larger samples help.",
        },
        "intuition": "Like ANOVA, but instead of using the raw numbers, we rank every observation from lowest to highest and then check if groups grab disproportionately high or low ranks.",
        "effect_size": "**ε² (epsilon-squared)**: proportion of variance explained. Same benchmarks as η².",
    },

    "pearson_r": {
        "name": "Pearson correlation",
        "emoji": "📈",
        "when_to_use": "You have two numeric variables and want to know if they move together — as one increases, does the other tend to increase (or decrease)?",
        "examples": [
            "Do students who study more hours get higher scores?",
            "Is height related to shoe size?",
        ],
        "hypotheses": {
            "H0": "There is **no linear relationship** between the two variables (r = 0).",
            "H1": "There **is** a linear relationship (r ≠ 0).",
        },
        "assumptions": [
            ("Both variables numeric", "Both columns must be numbers."),
            ("Linearity", "The relationship should be roughly linear (a straight line, not a curve)."),
            ("Normality", "Both variables should be roughly bell-shaped."),
            ("No major outliers", "Extreme outliers can distort r significantly."),
        ],
        "errors": {
            "type1": "Concluding a relationship exists when it doesn't — controlled by α.",
            "type2": "Missing a real relationship — more common with small samples.",
        },
        "intuition": "Imagine plotting all pairs of (X, Y) on a scatter plot. If the dots form a diagonal cloud going up-right, r is positive. If they go down-right, r is negative. If they're a circular blob, r is near zero.",
        "effect_size": "**r** is itself the effect size. r² (R-squared) tells you what percentage of variation in Y is explained by X. |r| = 0.1 small, 0.3 medium, 0.5 large.",
    },

    "spearman_r": {
        "name": "Spearman rank correlation",
        "emoji": "🔗",
        "when_to_use": "Like Pearson, but for non-normal data, ordinal variables, or when the relationship might be non-linear (but still monotonic).",
        "examples": [
            "Is class rank related to exam performance rank?",
            "Is customer satisfaction rating related to repurchase frequency?",
        ],
        "hypotheses": {
            "H0": "There is **no monotonic relationship** between the two variables.",
            "H1": "There **is** a monotonic relationship — as one increases, the other tends to increase (or decrease).",
        },
        "assumptions": [
            ("Ordinal or numeric", "Values can be ranked."),
            ("Monotonic relationship", "The relationship doesn't need to be linear, but should be consistently increasing or decreasing."),
        ],
        "errors": {
            "type1": "False relationship detected — controlled by α.",
            "type2": "Missing a real relationship — larger samples help.",
        },
        "intuition": "Instead of correlating the raw values, we correlate their *ranks*. This makes the result robust to outliers and non-normal distributions.",
        "effect_size": "**ρ (rho)** is interpreted like Pearson's r: 0.1 small, 0.3 medium, 0.5 large.",
    },

    "chi2_indep": {
        "name": "Chi-square independence test",
        "emoji": "🎲",
        "when_to_use": "You have two categorical variables and want to know if they are **related** or **independent** of each other.",
        "examples": [
            "Is product preference related to age group?",
            "Is political affiliation related to education level?",
        ],
        "hypotheses": {
            "H0": "The two categories are **independent** — knowing one tells you nothing about the other.",
            "H1": "The two categories are **associated** — one is related to the other.",
        },
        "assumptions": [
            ("Categorical data", "Both columns should contain categories (not continuous numbers)."),
            ("Adequate cell counts", "Each cell in the table should have an expected count ≥ 5. (Fisher's exact test is better for small tables.)"),
            ("Independence", "Each observation belongs to exactly one cell."),
        ],
        "errors": {
            "type1": "Concluding an association exists when categories are actually independent — controlled by α.",
            "type2": "Missing a real association — more common with small samples or weak effects.",
        },
        "intuition": "Build a table counting how many people fall into each combination of categories. Then ask: if there's no relationship, what counts would we *expect*? Chi-square measures the total gap between what we observe and what we'd expect under independence.",
        "effect_size": "**Cramér's V**: 0 = no association, 1 = perfect association. Benchmarks depend on table size.",
    },
}


def get_theory_card(test_id: str) -> dict:
    """Returns the theory card dict for a given test_id, with a fallback."""
    return CARDS.get(test_id, CARDS.get("independent_t"))


def render_theory_card(card: dict) -> str:
    """Renders the card as markdown for st.markdown()."""
    lines = []
    lines.append(f"## {card['emoji']} {card['name']}")
    lines.append(f"\n**When to use:** {card['when_to_use']}")

    lines.append("\n**Examples:**")
    for ex in card.get("examples", []):
        lines.append(f"- {ex}")

    h = card.get("hypotheses", {})
    lines.append(f"\n**Null hypothesis (H₀):** {h.get('H0', '')}")
    lines.append(f"\n**Alternative hypothesis (H₁):** {h.get('H1', '')}")

    lines.append("\n**Key assumptions:**")
    for name, desc in card.get("assumptions", []):
        lines.append(f"- **{name}:** {desc}")

    lines.append(f"\n💡 **Intuition:** {card.get('intuition', '')}")

    lines.append(f"\n**Effect size:** {card.get('effect_size', '')}")

    errs = card.get("errors", {})
    lines.append("\n**What can go wrong:**")
    lines.append(f"- **Type I error (false alarm):** {errs.get('type1', '')}")
    lines.append(f"- **Type II error (missed effect):** {errs.get('type2', '')}")

    if "followup" in card:
        lines.append(f"\n⚠️ **Next step:** {card['followup']}")

    return "\n".join(lines)
