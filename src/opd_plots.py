"""Notebook 04 diagrams and plots. Performance plots require actual run records."""

import json
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
            (0.37, "2. Score", "Teacher chat context\nSame response prefixes", PURPLE),
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
            "Each update uses fresh responses from the current student.",
            ha="center",
            color=BLUE,
            fontsize=12,
        )
        ax.text(
            0.5,
            0.04,
            "Student prompt stays plain. Teacher prompt uses non-thinking chat. Response IDs stay unchanged.",
            ha="center",
            fontsize=10,
            color=SLATE,
        )
        _save(fig, directory, "on_policy_workflow")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.set(xlim=(-3, 4.8), ylim=(-0.8, 2.8))
        ax.axis("off")
        ax.text(-2.9, 2.5, "Different prompts. Identical response IDs.", weight="bold", fontsize=20)
        for y, name, context in [
            (1.8, "Student input", "Plain context"),
            (0.9, "Teacher input", "Chat context"),
        ]:
            ax.text(-2.9, y, name, va="center", fontsize=11, weight="bold")
            for x, token in enumerate([context, "Last prompt", "Response 1", "Response 2", "EOS"]):
                color = BLUE if x >= 2 else (SLATE if y > 1 else PURPLE)
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
                ax.text(x, y, token, ha="center", va="center", fontsize=9, color=color)
        ax.text(
            -2.9,
            0.12,
            r"First response uses student logits at $P_S-1$ and teacher logits at $P_T-1$.",
            color=BLUE,
            fontsize=11,
        )
        ax.text(
            -2.9,
            -0.34,
            r"Slice each model independently. Both response tensors have shape $[1,R,V]$.",
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


def plot_observed_failure(directory: Path):
    """Render only the archived GPU records and the saved one-response probe."""
    recorded = directory / "recorded_failure"
    history = json.loads((recorded / "on_policy_history.json").read_text())
    probe = json.loads((recorded / "teacher_probe.json").read_text())["rows"]
    boundary_position = max(row["response_position"] for row in probe)
    endings = [row for row in probe if row["response_position"] == boundary_position]
    with plt.rc_context({**STYLE, "svg.fonttype": "path", "svg.hashsalt": "opd-failure"}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        steps = [row["step"] for row in history]
        axes[0].plot(
            steps, [row["response_tokens"] / row["examples"] for row in history], color=BLUE
        )
        axes[0].axhline(256, color=SLATE, ls="--", lw=1, label="256-token budget")
        axes[0].set(
            title="Responses grow to the ceiling",
            xlabel="Optimizer update",
            ylabel="Mean response tokens",
            ylim=(0, 280),
        )
        axes[0].legend(frameon=False, fontsize=8, loc="lower right")
        axes[1].plot(
            steps,
            [100 * row["token_limit_stops"] / row["examples"] for row in history],
            color=PURPLE,
        )
        axes[1].set(
            title="Finishing behavior deteriorates",
            xlabel="Optimizer update",
            ylabel="Capped responses per group (%)",
            ylim=(-3, 105),
        )
        probabilities = [row["stop_probability_T1"] for row in endings]
        axes[2].bar(
            ["Raw prompt", "Non-thinking\nchat"], probabilities, color=[SLATE, TEAL], width=0.55
        )
        axes[2].set(
            yscale="log",
            ylim=(1e-9, 2),
            title="Teacher stopping mass changes",
            ylabel="Combined stop probability, T = 1",
        )
        for index, probability in enumerate(probabilities):
            axes[2].text(index, probability * 1.7, f"{probability:.3g}", ha="center", fontsize=10)
        fig.text(
            0.5,
            -0.015,
            "The first two panels show 52 logged updates. The last shows one response boundary. Accuracy requires separate checks.",
            ha="center",
            fontsize=9,
            color=SLATE,
        )
        _save(fig, directory, "observed_failure")


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
