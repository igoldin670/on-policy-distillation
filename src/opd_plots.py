"""Notebook 04 diagrams and plots. Performance plots require actual run records."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from src.baseline_plots import BLUE, NAVY, PURPLE, SLATE, STYLE, TEAL, _save

COLORS = {"starting_sft": SLATE, "sft_control": TEAL, "on_policy": BLUE}
LABELS = {
    "starting_sft": "Starting SFT",
    "sft_control": "Continued SFT",
    "on_policy": "On-policy KD",
}


def _panel(ax, x, y, w, h, title, detail, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=.012",
            facecolor="#F8FAFC",
            edgecolor=color,
            linewidth=1.6,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.7,
        title,
        ha="center",
        va="center",
        color=color,
        weight="bold",
        fontsize=12,
    )
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", color=NAVY, fontsize=10)


def render_diagrams(directory: Path):
    with plt.rc_context({**STYLE, "svg.fonttype": "path", "svg.hashsalt": "opd04"}):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.02,
            0.94,
            "The student writes. The teacher guides the next choices.",
            weight="bold",
            fontsize=19,
        )
        for x, title, detail, color in [
            (0.03, "1. Sample", "Current student writes\none fresh response", BLUE),
            (0.37, "2. Score", "Frozen teacher sees\nthe exact same prefixes", PURPLE),
            (0.71, "3. Learn", "Match next-token choices\nUpdate only the student", TEAL),
        ]:
            _panel(ax, x, 0.39, 0.26, 0.34, title, detail, color)
        for x in (0.30, 0.64):
            ax.annotate(
                "",
                xy=(x + 0.06, 0.56),
                xytext=(x, 0.56),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": SLATE},
            )
        ax.plot([0.84, 0.84, 0.16], [0.36, 0.25, 0.25], color=BLUE, lw=1.7)
        ax.annotate(
            "",
            xy=(0.16, 0.36),
            xytext=(0.16, 0.25),
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.7},
        )
        ax.text(
            0.5,
            0.16,
            "Repeat with the updated student. Old rollouts are not reused.",
            ha="center",
            color=BLUE,
            fontsize=12,
        )
        ax.text(
            0.5,
            0.04,
            "Teacher probabilities are targets, not correctness grades. No teacher-generated replacement answer.",
            ha="center",
            fontsize=10,
            color=SLATE,
        )
        _save(fig, directory, "on_policy_workflow")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.set(xlim=(-2.6, 4.8), ylim=(-0.8, 2.8))
        ax.axis("off")
        ax.text(
            -2.5,
            2.5,
            "Same tokens. Same prefixes. One prediction shift.",
            weight="bold",
            fontsize=20,
        )
        for y, name, tokens in [
            (
                1.8,
                "Complete sequence",
                ["Prompt", "Last prompt", "Response 1", "Response 2", "EOS"],
            ),
            (
                0.9,
                "Logits predict",
                ["Last prompt", "Response 1", "Response 2", "EOS", "unobserved"],
            ),
        ]:
            ax.text(-2.5, y, name, va="center", fontsize=10, weight="bold")
            for x, token in enumerate(tokens):
                active = x >= 2 if y > 1 else 1 <= x <= 3
                color = BLUE if active else SLATE
                ax.add_patch(
                    FancyBboxPatch(
                        (x - 0.43, y - 0.26),
                        0.86,
                        0.52,
                        boxstyle="round,pad=.012",
                        facecolor="#F8FAFC",
                        edgecolor=color,
                        linewidth=1.6,
                    )
                )
                ax.text(
                    x, y, token, ha="center", va="center", fontsize=10, weight="bold", color=color
                )
        ax.text(
            -2.5,
            0.13,
            "The response starts at input index P. Its predictions start at logits index P - 1.",
            color=BLUE,
            fontsize=11,
        )
        ax.text(
            -2.5,
            -0.32,
            "Keep response positions only. Include a sampled EOS; do not invent one at a token limit.",
            fontsize=11,
            color=SLATE,
        )
        _save(fig, directory, "token_alignment")

        fig, ax = plt.subplots(figsize=(12, 4.7))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.02,
            0.94,
            "One starting checkpoint. Three honest comparisons.",
            weight="bold",
            fontsize=20,
        )
        _panel(
            ax,
            0.03,
            0.38,
            0.25,
            0.29,
            "03 checkpoint",
            "Identical saved weights\nVerified SHA-256 fingerprints",
            SLATE,
        )
        for y, title, detail, color in [
            (0.65, "Starting SFT", "No further updates", SLATE),
            (0.37, "Continued SFT", "New reference solutions", TEAL),
            (0.08, "On-policy KD", "Fresh responses + teacher feedback", BLUE),
        ]:
            _panel(ax, 0.43, y, 0.48, 0.22, title, detail, color)
            ax.annotate(
                "",
                xy=(0.41, y + 0.11),
                xytext=(0.29, 0.52),
                arrowprops={"arrowstyle": "-|>", "color": color},
            )
        _save(fig, directory, "controlled_comparison")


def plot_run(histories, evaluations, directory: Path):
    """Draw only observed records; never generate an expected improvement curve."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        for ax, arm, title in zip(
            axes,
            ("sft_control", "on_policy"),
            (
                "Continued SFT: response cross-entropy",
                "On-policy KD: forward KL on fresh responses",
            ),
        ):
            history = histories[arm]
            ax.plot(
                [r["step"] for r in history],
                [r["loss"] for r in history],
                color=COLORS[arm],
                lw=1.6,
            )
            ax.set(
                title=title,
                xlabel="Optimizer update",
                ylabel="Nats per response token, then mean across responses",
            )
        fig.text(
            0.5,
            -0.01,
            "These are different objectives on different response prefixes. Their heights are not directly comparable.",
            ha="center",
            color=SLATE,
            fontsize=10,
        )
        _save(fig, directory, "training_objectives")

        fig, ax = plt.subplots(figsize=(10, 4.4))
        arms = list(COLORS)
        metrics = [
            ("strict", "accuracy", "Strict numeric match"),
            ("strict", "parse_rate", "Required final-answer format"),
            ("prose", "accuracy", "Exploratory prose match"),
        ]
        x = np.arange(len(metrics))
        for offset, arm in enumerate(arms):
            values = [
                100 * evaluations[arm]["scores"][checker][metric] for checker, metric, _ in metrics
            ]
            bars = ax.bar(
                x + (offset - 1) * 0.24, values, 0.22, label=LABELS[arm], color=COLORS[arm]
            )
            ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
        n = len(evaluations[arms[0]]["rows"])
        ax.set(
            xticks=x,
            xticklabels=[m[2] for m in metrics],
            ylim=(0, 110),
            ylabel="Percent of all questions",
            title=f"Primary held-out test: {n} paired questions, one training seed",
        )
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.2), ncol=3)
        fig.text(
            0.5,
            -0.01,
            "Numeric agreement checks a final number. It does not establish that the reasoning is valid.",
            ha="center",
            fontsize=10,
            color=SLATE,
        )
        _save(fig, directory, "primary_test_comparison")
