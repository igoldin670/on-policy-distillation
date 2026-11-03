"""Generate repo-native diagrams and a restrained GitHub banner (no model results)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def box(ax, x, y, w, h, title, detail, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            facecolor="#F8FAFC",
            edgecolor=color,
            linewidth=1.5,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.7,
        title,
        ha="center",
        va="center",
        weight="bold",
        color=color,
        fontsize=12,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        detail,
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )


def render_assets():
    from src.baseline_plots import STYLE, _save

    assets = ROOT / "notebooks/assets/03_sft_cold_start"
    with plt.rc_context({**STYLE, "svg.fonttype": "path", "svg.hashsalt": "opd03"}):
        fig, ax = plt.subplots(figsize=(12, 3.3))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.02,
            0.88,
            "AN EXPERIMENT YOU CAN FOLLOW",
            fontsize=11,
            color="#2563EB",
            weight="bold",
        )
        ax.text(
            0.02,
            0.64,
            "On-Policy Distillation",
            fontsize=34,
            color="#14253D",
            weight="bold",
        )
        ax.text(
            0.02,
            0.42,
            "Read the tensors. Check the math. Measure the result.",
            fontsize=16,
            color="#475569",
        )
        for x, label in zip(
            [0.02, 0.27, 0.52, 0.77],
            [
                "01 / Mechanics",
                "02 / Baseline",
                "03 / Supervised start",
                "04 / Distillation",
            ],
        ):
            ax.plot(
                [x, x + 0.2],
                [0.22, 0.22],
                color="#2563EB" if x < 0.7 else "#CBD5E1",
                lw=3,
            )
            ax.text(x, 0.09, label, fontsize=11, color="#14253D")
        _save(fig, ROOT / "assets", "project_banner")

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.01,
            0.94,
            "A supervised start before on-policy feedback",
            fontsize=20,
            weight="bold",
        )
        for x, title, detail, color in [
            (
                0.02,
                "Worked solutions",
                "GSM8K train split\nQuestion + reference answer",
                "#0F766E",
            ),
            (
                0.36,
                "Student learns",
                "Predict response tokens\nUpdate student weights",
                "#2563EB",
            ),
            (
                0.70,
                "Check and save",
                "Separate validation questions\nReusable final checkpoint",
                "#7C3AED",
            ),
        ]:
            box(ax, x, 0.29, 0.27, 0.42, title, detail, color)
        for x in [0.30, 0.64]:
            ax.annotate(
                "",
                xy=(x + 0.045, 0.5),
                xytext=(x, 0.5),
                arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#64748B"},
            )
        ax.text(
            0.5,
            0.08,
            "Notebook 03: reference-solution SFT. No teacher model is loaded.",
            ha="center",
            fontsize=12,
            color="#475569",
        )
        _save(fig, assets, "sft_workflow")

        fig, ax = plt.subplots(figsize=(12, 4.6))
        ax.set(xlim=(-1.7, 6.2), ylim=(-0.7, 3.1))
        ax.axis("off")
        ax.text(
            -1.6,
            2.8,
            "Train on the response, not the question or padding",
            fontsize=20,
            weight="bold",
        )
        rows = [
            (
                "Token role",
                ["Prompt 1", "Prompt 2", "Response 1", "Response 2", "EOS", "PAD"],
            ),
            ("Attention mask", ["1", "1", "1", "1", "1", "0"]),
            (
                "Training label",
                ["-100", "-100", "token ID", "token ID", "EOS ID", "-100"],
            ),
        ]
        for y, (label, values) in zip([2.0, 1.25, 0.5], rows):
            ax.text(-1.6, y, label, fontsize=11, weight="bold", va="center")
            for x, value in enumerate(values):
                response = 2 <= x <= 4
                ax.add_patch(
                    FancyBboxPatch(
                        (x - 0.42, y - 0.27),
                        0.84,
                        0.54,
                        boxstyle="round,pad=.02",
                        facecolor="#DBEAFE" if response else "#F1F5F9",
                        edgecolor="#93C5FD" if response else "#CBD5E1",
                    )
                )
                ax.text(x, y, value, ha="center", va="center", fontsize=10, color="#14253D")
        ax.text(
            -1.6,
            -0.22,
            "Prediction: logits after Prompt 2 score Response 1. The shift happens exactly once.",
            fontsize=11,
            color="#2563EB",
        )
        ax.text(
            -1.6,
            -0.55,
            "EOS is a real target. PAD is ignored by position, even when PAD and EOS share an ID.",
            fontsize=11,
            color="#475569",
        )
        _save(fig, assets, "response_mask")


if __name__ == "__main__":
    render_assets()
