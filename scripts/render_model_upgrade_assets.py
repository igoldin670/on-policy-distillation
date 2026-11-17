"""Render the model-upgrade teaching figures from pinned facts and labeled examples."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.baseline_plots import AMBER, BLUE, NAVY, PURPLE, SLATE, STYLE, TEAL, _arrow, _box, _save  # noqa: E402

ASSETS = ROOT / "notebooks/assets/05_qwen38_restart"


def canvas(title, subtitle, height=4.5):
    figure, axis = plt.subplots(figsize=(12, height))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    axis.text(0.02, 0.94, title, fontsize=20, weight="bold", va="top")
    axis.text(0.02, 0.81, subtitle, fontsize=11, color=SLATE, va="top")
    return figure, axis


def render():
    facts = json.loads((ASSETS / "model_facts.json").read_text())
    with plt.rc_context({**STYLE, "svg.hashsalt": "qwen38-teaching"}):
        figure, axis = canvas(
            "A new model pair, one check at a time",
            "The first milestone follows one sampled response through both models.",
        )
        for x, title, detail, color in [
            (0.025, "Pin", "Model revisions\nTokenizer files", BLUE),
            (0.225, "Verify", "Token meanings\nEnding IDs", TEAL),
            (0.425, "Generate", "Student response\nOriginal token IDs", BLUE),
            (0.625, "Score", "Frozen teacher\nAligned prefixes", PURPLE),
            (0.825, "Record", "Shapes and scores\nReadiness report", TEAL),
        ]:
            _box(axis, x, 0.33, 0.15, 0.34, title, detail, color)
            if x < 0.8:
                _arrow(axis, (x + 0.15, 0.50), (x + 0.20, 0.50))
        axis.text(
            0.5,
            0.12,
            "Next experiment milestones    fresh baseline → supervised checkpoint → guarded pilot",
            ha="center",
            color=SLATE,
        )
        _save(figure, ASSETS, "upgrade_workflow")

        figure, axis = canvas(
            "Two models from the same architecture family",
            "Text-only loading selects the language model from each published checkpoint.",
        )
        for x, role, title, color in [
            (0.04, "student", "Qwen3.5-0.8B-Base", BLUE),
            (0.55, "teacher", "Qwen3.8-27B", PURPLE),
        ]:
            f = facts[role]
            _box(
                axis,
                x,
                0.24,
                0.40,
                0.43,
                title,
                f"{role.capitalize()} · {f['layers']} decoder layers\n{f['linear_layers']} linear-attention + {f['full_attention_layers']} full-attention\nHidden dimension {f['hidden_size']:,}",
                color,
            )
        axis.text(
            0.5,
            0.11,
            "Both use pretrained text weights. Future training updates the student.",
            ha="center",
            color=SLATE,
            fontsize=11,
        )
        _save(figure, ASSETS, "model_family")

        figure, axis = canvas(
            "Read both tokenizer files before comparing IDs",
            "The standard loader combines vocabulary data and special-token definitions.",
        )
        _box(
            axis,
            0.035,
            0.32,
            0.25,
            0.34,
            "Raw student file",
            "248,066 entries\ntokenizer.json",
            BLUE,
        )
        _box(
            axis,
            0.37,
            0.32,
            0.25,
            0.34,
            "Official configuration",
            "11 additional definitions\ntokenizer_config.json",
            PURPLE,
        )
        _box(
            axis,
            0.705,
            0.32,
            0.25,
            0.34,
            "Resolved mapping",
            "248,077 identical IDs\nStudent and teacher",
            TEAL,
        )
        _arrow(axis, (0.285, 0.49), (0.37, 0.49))
        _arrow(axis, (0.62, 0.49), (0.705, 0.49))
        axis.text(
            0.5,
            0.12,
            "Both output heads contain 248,320 columns, including 243 padded rows.",
            ha="center",
            color=SLATE,
        )
        _save(figure, ASSETS, "tokenizer_resolution")

        figure, axis = canvas(
            "The same response has two different prompt offsets",
            "Illustrative positions. Each last prompt position predicts the first response token.",
            5.3,
        )
        for y, label, prompt_length, color in [
            (0.58, "Student", 3, BLUE),
            (0.30, "Teacher", 5, PURPLE),
        ]:
            axis.text(0.025, y + 0.03, label, weight="bold", color=color)
            _box(
                axis,
                0.15,
                y - 0.08,
                0.25,
                0.21,
                f"Prompt · {prompt_length} tokens",
                f"Last position {prompt_length - 1}",
                color,
            )
            for j, token in enumerate((42, 43, 248046)):
                _box(
                    axis,
                    0.47 + j * 0.16,
                    y - 0.08,
                    0.13,
                    0.21,
                    f"ID {token}",
                    f"Position {prompt_length + j}",
                    TEAL,
                )
            _arrow(axis, (0.4, y + 0.025), (0.47, y + 0.025), color)
        axis.text(
            0.5,
            0.09,
            "Student logits 2 → token 42 at position 3      Teacher logits 4 → token 42 at position 5",
            ha="center",
            color=NAVY,
            fontsize=11,
        )
        _save(figure, ASSETS, "causal_alignment")

        figure, axis = canvas(
            "Teacher sharding preserves separate GPU memories",
            "Illustrative placement. The loader calculates the actual layer map for your allocation.",
        )
        for x, title, color in [(0.06, "Logical GPU 0", BLUE), (0.56, "Logical GPU 1", PURPLE)]:
            axis.add_patch(
                FancyBboxPatch(
                    (x, 0.19),
                    0.38,
                    0.52,
                    boxstyle="round,pad=.015",
                    facecolor="#F8FAFC",
                    edgecolor=color,
                    lw=1.5,
                )
            )
            axis.text(x + 0.19, 0.64, title, ha="center", weight="bold", color=color, fontsize=13)
        _box(axis, 0.085, 0.38, 0.33, 0.17, "FP32 student", "Trainable in a future pilot", BLUE)
        _box(axis, 0.085, 0.23, 0.33, 0.12, "BF16 teacher layers", "Frozen weights", PURPLE)
        _box(
            axis, 0.585, 0.30, 0.33, 0.25, "Remaining teacher layers", "Frozen BF16 weights", PURPLE
        )
        _arrow(axis, (0.45, 0.43), (0.55, 0.43), SLATE)
        axis.text(
            0.5,
            0.09,
            "Memory reservations cover working tensors. Peak VRAM is measured during GPU qualification.",
            ha="center",
            color=SLATE,
            fontsize=10.5,
        )
        _save(figure, ASSETS, "gpu_layout")

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        x = np.arange(3)
        axes[0].bar(x - 0.18, [0.2, 0.79, 0.01], 0.36, color=BLUE, label="Student")
        axes[0].bar(x + 0.18, [0.2, 0.01, 0.79], 0.36, color=PURPLE, label="Teacher")
        axes[0].set(
            xticks=x,
            xticklabels=["Continue", "endoftext", "im_end"],
            ylim=(0, 1),
            ylabel="Probability",
            title="Two ending representations",
        )
        axes[0].legend(frameon=False)
        axes[1].bar(np.array([0, 1]) - 0.18, [0.2, 0.8], 0.36, color=BLUE, label="Student")
        axes[1].bar(np.array([0, 1]) + 0.18, [0.2, 0.8], 0.36, color=PURPLE, label="Teacher")
        axes[1].set(
            xticks=[0, 1],
            xticklabels=["Continue", "STOP"],
            ylim=(0, 1),
            title="Sum ending probabilities before KL",
        )
        figure.suptitle(
            "An illustrative three-action distribution", fontsize=18, weight="bold", y=1.03
        )
        figure.text(
            0.5,
            -0.015,
            "Toy probabilities for explanation. Real model distributions use all 248,320 output columns.",
            ha="center",
            color=SLATE,
        )
        figure.tight_layout()
        _save(figure, ASSETS, "eos_probabilities")

        figure, axis = canvas(
            "A future pilot checks quality before another update",
            "The readiness notebook introduces this policy. Training begins in a later milestone.",
            5.2,
        )
        for x, title, detail, color in [
            (0.03, "Update", "One complete\noptimizer boundary", BLUE),
            (0.35, "Save + validate", "NLL, strict accuracy\nRecent cap rate", PURPLE),
            (0.69, "Decide", "Continue or pause\nRecord the reason", TEAL),
        ]:
            _box(axis, x, 0.40, 0.27, 0.28, title, detail, color)
        _arrow(axis, (0.30, 0.54), (0.35, 0.54))
        _arrow(axis, (0.62, 0.54), (0.69, 0.54))
        _box(
            axis,
            0.15,
            0.08,
            0.31,
            0.19,
            "Latest recovery state",
            "Preserve complete optimizer state",
            BLUE,
        )
        _box(
            axis,
            0.55,
            0.08,
            0.31,
            0.19,
            "Last acceptable checkpoint",
            "Protected during cleanup",
            TEAL,
        )
        _arrow(axis, (0.48, 0.40), (0.31, 0.28))
        _arrow(axis, (0.83, 0.40), (0.70, 0.28), TEAL)
        _save(figure, ASSETS, "checkpoint_safety")

        record = json.loads(
            (
                ROOT
                / "notebooks/assets/04_on_policy_distillation/recorded_eos_comparison/user_reported_summary.json"
            ).read_text()
        )
        rows = [record[k] for k in ("starting", "raw_token", "semantic_stop")]
        labels = ["Starting\nupdate 0", "Raw-token\nupdate 18", "Grouped STOP\nupdate 32"]
        figure, axes = plt.subplots(2, 2, figsize=(12, 7.2))
        for axis, key, title in zip(
            axes.flat,
            ["strict_correct", "format_compliant", "token_limit_stops", "reference_nll"],
            [
                "Strict numeric matches / 64",
                "Required final-answer format / 64",
                "Validation token-limit stops / 64",
                "Reference-token NLL",
            ],
        ):
            values = [row[key] for row in rows]
            axis.bar(labels, values, color=[SLATE, AMBER, TEAL], width=0.58)
            axis.set_title(title, loc="left", fontsize=12, weight="bold")
            axis.set_ylim(0, 64 if key != "reference_nll" else 1.15)
            for i, value in enumerate(values):
                axis.text(
                    i,
                    value + (1 if key != "reference_nll" else 0.02),
                    f"{value:.3f}" if key == "reference_nll" else str(value),
                    ha="center",
                    fontsize=11,
                )
        figure.suptitle(
            "Audited validation endpoints from the original Qwen3 pilot", fontsize=18, weight="bold"
        )
        figure.text(
            0.5,
            0.01,
            "Final checkpoints have different update counts. Counts reproduce from the collected validation responses.",
            ha="center",
            color=SLATE,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.94))
        _save(figure, ROOT / "notebooks/assets/04_on_policy_distillation", "pilot_outcomes")


def render_audited_history():
    """Connect observed checkpoints and rolling response windows from the verified archive."""
    folder = ROOT / "notebooks/assets/04_on_policy_distillation/recorded_eos_comparison/audited"
    summary = json.loads((folder / "audit_summary.json").read_text())
    fixed = json.loads((folder / "starting_sft_fixed_diagnostics.json").read_text())
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(12, 7.8))
        for mode, label, color in (
            ("raw_token", "Raw-token", AMBER),
            ("semantic_stop", "Grouped STOP", TEAL),
        ):
            history = json.loads((folder / f"{mode}_history.json").read_text())
            checkpoints = [row for row in history if "generated_validation" in row]
            steps = [0] + [row["step"] for row in checkpoints]
            accuracy = [summary["baseline_accuracy"] * 100] + [
                row["generated_validation"]["strict_accuracy"] * 100 for row in checkpoints
            ]
            nll = [summary["baseline_nll"]] + [
                row["validation_reference_nll"] for row in checkpoints
            ]
            kl = [fixed["mean_semantic_stop_kl_nats"]] + [
                row["fixed_semantic_stop_kl"] for row in checkpoints
            ]
            axes[0, 0].plot(steps, accuracy, marker="o", color=color, label=label)
            axes[0, 1].plot(steps, nll, marker="o", color=color, label=label)
            flags = []
            cap_steps, cap_rates = [], []
            for row in history:
                flags.extend(row["response_limit_flags"])
                if len(flags) >= 32:
                    cap_steps.append(row["step"])
                    cap_rates.append(sum(flags[-32:]) / 32 * 100)
            axes[1, 0].plot(cap_steps, cap_rates, color=color, label=label)
            axes[1, 1].plot(steps, kl, marker="o", color=color, label=label)
        for axis in axes.flat:
            axis.set_xlabel("Completed optimizer updates")
            axis.set_xlim(0, 33)
            axis.axvline(8, color=SLATE, linestyle=":", linewidth=1.2)
            axis.legend(frameon=False, fontsize=9)
        axes[0, 0].set(
            ylabel="Strict matches (%)",
            ylim=(0, 65),
            title="Accuracy declines at the first validation",
        )
        axes[0, 1].set(ylabel="Reference-token NLL", title="Reference loss rises later")
        axes[1, 0].set(
            ylabel="Latest 32 responses capped (%)",
            ylim=(0, 100),
            title="Sampled training responses become longer",
        )
        axes[1, 0].axhline(25, color=SLATE, linestyle="--", linewidth=1)
        axes[1, 1].set(
            ylabel="Grouped forward KL (nats)",
            title="Fixed-prefix agreement can improve during decline",
        )
        figure.suptitle(
            "Verified trajectory and validation history from the Qwen3 comparison",
            fontsize=17,
            weight="bold",
        )
        figure.text(
            0.5,
            0.01,
            "Dotted line marks update 8. Validation markers show measured checkpoints. Training cap rates use sampled responses.",
            ha="center",
            fontsize=10,
            color=SLATE,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.95))
        _save(figure, ROOT / "notebooks/assets/04_on_policy_distillation", "pilot_history")


if __name__ == "__main__":
    render()
    render_audited_history()
