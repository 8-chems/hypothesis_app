"""
interpreter.py — translates statistical results into plain English.
The user sees a two-sentence verdict, then can drill into details.
"""
import numpy as np


def interpret(result: dict, config: dict) -> dict:
    """
    Returns a dict with:
    - verdict:  "Yes" / "No clear difference"
    - headline: one bold sentence
    - detail:   one supporting sentence
    - effect_sentence: plain sentence about effect size
    - confidence_sentence
    - full_technical: for the "show me the numbers" expander
    """
    test_id = result.get("test_id", "")
    p = result.get("p", 1.0)
    alpha = result.get("alpha", 0.05)
    reject = result.get("reject", False)
    effect_size = result.get("effect_size", 0.0)
    effect_word = result.get("effect_size_word", "")
    effect_label = result.get("effect_label", "effect size")
    confidence = config.get("confidence", 95)
    tail = result.get("tail", "two")

    certainty_pct = round((1 - p) * 100, 1) if p < 1 else 99.9

    # ── Verdict ───────────────────────────────────────────────────────────────
    if reject:
        verdict = "✅ Yes — a real difference (or relationship) was found"
        if test_id in ("independent_t", "mann_whitney"):
            names = result.get("group_names", ["Group 1", "Group 2"])
            means = result.get("group_means", [0, 0])
            diff = abs(means[0] - means[1])
            headline = (
                f"**{names[0]}** and **{names[1]}** are genuinely different "
                f"— the gap is unlikely to be random."
            )
            detail = (
                f"We are {certainty_pct:.0f}% confident this difference is real, "
                f"not a fluke of sampling."
            )

        elif test_id in ("paired_t", "wilcoxon"):
            names = result.get("group_names", ["Before", "After"])
            means = result.get("group_means", [0, 0])
            direction = "increased" if means[1] > means[0] else "decreased"
            headline = (
                f"There was a real change: **{names[1]}** {direction} compared to **{names[0]}**."
            )
            detail = (
                f"We are {certainty_pct:.0f}% confident this change is genuine, not random variation."
            )

        elif test_id in ("anova", "kruskal"):
            headline = "At least one group is genuinely different from the others."
            detail = (
                f"We are {certainty_pct:.0f}% confident these groups don't all come from the same population."
            )

        elif test_id in ("pearson_r", "spearman_r"):
            direction = "positive" if effect_size > 0 else "negative"
            headline = f"There is a real **{direction} relationship** between these two variables."
            detail = (
                f"We are {certainty_pct:.0f}% confident this relationship exists in the real world."
            )

        elif test_id == "chi2_indep":
            headline = "The two categories are **not independent** — knowing one helps predict the other."
            detail = (
                f"We are {certainty_pct:.0f}% confident this association is real, not by chance."
            )

        else:
            headline = "A statistically significant result was found."
            detail = f"p = {p:.4f}, which is below your threshold of {alpha}."

    else:
        verdict = "❔ No clear difference was found"
        if test_id in ("independent_t", "mann_whitney"):
            names = result.get("group_names", ["Group 1", "Group 2"])
            headline = (
                f"Your data does **not** give enough evidence that "
                f"**{names[0]}** and **{names[1]}** are different."
            )
        elif test_id in ("paired_t", "wilcoxon"):
            headline = "Your data does **not** show a clear change between the two measurements."
        elif test_id in ("anova", "kruskal"):
            headline = "Your data does **not** show a clear difference between the groups."
        elif test_id in ("pearson_r", "spearman_r"):
            headline = "Your data does **not** show a clear relationship between these two variables."
        elif test_id == "chi2_indep":
            headline = "Your data does **not** show a clear association between the two categories."
        else:
            headline = "No statistically significant result was found."

        detail = (
            f"This doesn't mean the groups are identical — "
            f"it may mean your sample is too small to detect a real difference. "
            f"(p = {p:.4f}, threshold = {alpha})"
        )

    # ── Effect size sentence ──────────────────────────────────────────────────
    if test_id in ("pearson_r", "spearman_r"):
        r2_pct = round(effect_size**2 * 100, 1)
        effect_sentence = (
            f"The relationship is **{effect_word}** ({effect_label} = {effect_size:.3f}). "
            f"About {r2_pct}% of the variation in one variable can be explained by the other."
        )
    elif test_id == "chi2_indep":
        effect_sentence = (
            f"The strength of the association is **{effect_word}** ({effect_label} = {abs(effect_size):.3f}). "
            f"Values near 0 = no link; near 1 = very strong link."
        )
    else:
        effect_sentence = (
            f"The size of the difference is **{effect_word}** ({effect_label} = {abs(effect_size):.3f}). "
            f"Even a significant result can have a small practical effect — size matters too."
        )

    # ── CI sentence ───────────────────────────────────────────────────────────
    ci = result.get("ci")
    ci_sentence = ""
    if ci is not None and test_id not in ("pearson_r", "spearman_r"):
        ci_sentence = (
            f"We are 95% confident the true difference lies between "
            f"{ci[0]:.2f} and {ci[1]:.2f}."
        )
    elif ci is not None:
        ci_sentence = (
            f"95% confidence interval for r: [{ci[0]:.3f}, {ci[1]:.3f}]."
        )

    # ── Technical summary ─────────────────────────────────────────────────────
    stat_label = result.get("stat_label", "stat")
    stat_val   = result.get("stat", 0.0)
    df_val     = result.get("df", "?")
    technical = (
        f"{stat_label}({df_val}) = {stat_val:.4f}, "
        f"p = {p:.4f}, "
        f"{effect_label} = {abs(effect_size):.4f}, "
        f"α = {alpha}"
    )
    if result.get("welch"):
        technical += " (Welch's correction applied)"
    if result.get("note"):
        technical += f" | {result['note']}"

    return {
        "verdict": verdict,
        "headline": headline,
        "detail": detail,
        "effect_sentence": effect_sentence,
        "ci_sentence": ci_sentence,
        "technical": technical,
        "reject": reject,
        "p": p,
        "alpha": alpha,
    }
