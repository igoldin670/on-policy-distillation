"""Static, data-backed figures for Notebook 02; no remote assets or custom CSS."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY = "#14253D"
BLUE = "#2563EB"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
AMBER = "#B45309"
SLATE = "#64748B"
BACKGROUND = "#F8FAFC"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "axes.edgecolor": "#CBD5E1",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "none",
    "svg.hashsalt": "notebook02",
    "savefig.facecolor": "white",
}


def _save(figure, directory: Path, name: str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{name}.png"
    figure.savefig(png, dpi=170, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    figure.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    # Newlines already separate path coordinates, so strip Matplotlib's trailing
    # spaces to keep generated vector files clean without changing the artwork.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(figure)
    return png


def _box(ax, x, y, width, height, title, detail, color=BLUE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.1,
            edgecolor=color,
            facecolor=BACKGROUND,
        )
    )
    ax.text(
        x + width / 2,
        y + height * 0.78,
        title,
        ha="center",
        va="center",
        color=color,
        weight="bold",
        fontsize=11.5,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        detail,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=9.5,
        linespacing=1.4,
    )


def _arrow(ax, start, end, color=SLATE, dashed=False):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.4,
            "color": color,
            "linestyle": "--" if dashed else "-",
            "shrinkA": 5,
            "shrinkB": 5,
        },
    )


def plot_distillation_context(directory: Path) -> Path:
    """Conceptual training diagram, explicitly outside this baseline's execution."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.5, 4.5))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.02,
            0.95,
            "Research context: learn on the student's own prefixes",
            fontsize=17,
            weight="bold",
        )
        ax.text(
            0.02,
            0.87,
            "Conceptual training loop. Notebook 02 measures the original student; it does not run this loop.",
            color=SLATE,
        )
        _box(
            ax,
            0.025,
            0.38,
            0.19,
            0.23,
            "Student samples",
            "$y \\sim p_S(\\cdot\\mid x)$\nExact response IDs",
            BLUE,
        )
        _box(
            ax,
            0.28,
            0.38,
            0.18,
            0.23,
            "Shared prefixes",
            "$x, y_{<t}$\nSame token meanings",
            TEAL,
        )
        _box(
            ax,
            0.53,
            0.62,
            0.20,
            0.19,
            "Frozen teacher",
            "$p_T(\\cdot\\mid x,y_{<t})$\nNo teacher gradients",
            PURPLE,
        )
        _box(
            ax,
            0.53,
            0.20,
            0.20,
            0.19,
            "Student scores",
            "$p_S^\\theta(\\cdot\\mid x,y_{<t})$\nStudent gradients",
            BLUE,
        )
        _box(
            ax,
            0.80,
            0.38,
            0.17,
            0.23,
            "Divergence",
            "Full vocabulary\nResponse positions",
            NAVY,
        )
        _arrow(ax, (0.215, 0.495), (0.28, 0.495))
        _arrow(ax, (0.46, 0.52), (0.53, 0.71))
        _arrow(ax, (0.46, 0.47), (0.53, 0.29))
        _arrow(ax, (0.73, 0.71), (0.80, 0.54), PURPLE)
        _arrow(ax, (0.73, 0.29), (0.80, 0.44), BLUE)
        ax.plot(
            [0.885, 0.885, 0.12, 0.12],
            [0.38, 0.08, 0.08, 0.24],
            color=BLUE,
            ls="--",
            lw=1.3,
        )
        _arrow(ax, (0.12, 0.24), (0.12, 0.38), BLUE, dashed=True)
        ax.text(
            0.49,
            0.015,
            "Update student only, then sample fresh responses for the next iteration",
            color=BLUE,
            ha="center",
            fontsize=10,
        )
        return _save(fig, directory, "distillation_context")


def plot_evaluation_workflow(directory: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13.5, 3.6))
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.text(
            0.02, 0.93, "One protocol, two execution modes", fontsize=17, weight="bold"
        )
        _box(
            ax,
            0.025,
            0.52,
            0.20,
            0.22,
            "Replay (CPU)",
            "Versioned A100 responses\nCPU, no model download",
            TEAL,
        )
        _box(
            ax,
            0.025,
            0.15,
            0.20,
            0.22,
            "Generate (CUDA)",
            "Fresh pinned student\nCUDA runtime required",
            BLUE,
        )
        _box(
            ax,
            0.31,
            0.33,
            0.19,
            0.23,
            "Validate records",
            "Checksums, prompt IDs\nLengths and stop flags",
            NAVY,
        )
        _box(
            ax,
            0.57,
            0.33,
            0.18,
            0.23,
            "Two checkers",
            "Strict format\nProse, post-hoc",
            PURPLE,
        )
        _box(
            ax,
            0.82,
            0.33,
            0.15,
            0.23,
            "Evidence report",
            "All 32 questions\nTables, figures, ZIP",
            TEAL,
        )
        _arrow(ax, (0.225, 0.63), (0.31, 0.48))
        _arrow(ax, (0.225, 0.26), (0.31, 0.40))
        _arrow(ax, (0.50, 0.445), (0.57, 0.445))
        _arrow(ax, (0.75, 0.445), (0.82, 0.445))
        ax.text(
            0.5,
            0.04,
            "No teacher, optimizer, or weight update. The reference is used only after answer extraction.",
            ha="center",
            color=SLATE,
            fontsize=10,
        )
        return _save(fig, directory, "evaluation_workflow")


def plot_metric_comparison(metrics: list[dict], directory: Path) -> Path:
    """Compare two scoring policies on the same responses, not two model versions."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.8, 3.7))
        positions = [1, 0]
        offsets = [0, 0]
        for field, label, color in [
            ("correct", "Numeric match", TEAL),
            ("parsed_wrong", "Parsed mismatch", SLATE),
            ("unparsed", "No extracted answer", AMBER),
        ]:
            values = [row[field] for row in metrics]
            ax.barh(
                positions, values, left=offsets, height=0.48, color=color, label=label
            )
            for y, left, value in zip(positions, offsets, values, strict=True):
                if value:
                    ax.text(
                        left + value / 2,
                        y,
                        str(value),
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                    )
            offsets = [
                left + value for left, value in zip(offsets, values, strict=True)
            ]
        total = metrics[0]["questions"]
        ax.set(
            yticks=positions,
            yticklabels=[row["checker"] for row in metrics],
            xlabel=f"Questions (same {total} responses, all included)",
            xlim=(0, total),
        )
        ax.set_title(
            "Same responses, two answer-extraction policies",
            loc="left",
            fontsize=16,
            weight="bold",
            pad=35,
        )
        ax.text(
            0,
            1.10,
            "Comparison of extraction policies; this is not evidence of a student improvement.",
            transform=ax.transAxes,
            color=SLATE,
            fontsize=10,
        )
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.set_xticks(range(0, total + 1, max(1, total // 8)))
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.24),
            frameon=False,
            ncol=3,
            fontsize=10,
        )
        fig.subplots_adjust(left=0.24, bottom=0.27, top=0.78)
        return _save(fig, directory, "metric_comparison")


def plot_response_lengths(rows: list[dict], budget: int, directory: Path) -> Path:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        complete = [
            row["response_length"] for row in rows if not row["hit_token_limit"]
        ]
        limited = [row["response_length"] for row in rows if row["hit_token_limit"]]
        bins = [budget * index / 8 for index in range(9)]
        ax.hist(
            [complete, limited],
            bins=bins,
            stacked=True,
            color=[BLUE, AMBER],
            edgecolor="white",
            linewidth=1.2,
            label=[f"End token ({len(complete)})", f"Token budget ({len(limited)})"],
        )
        ax.axvline(budget, color=AMBER, lw=1.2, ls="--")
        ax.set(
            xlabel="Generated response tokens (including an end token when present)",
            ylabel="Questions",
            xlim=(0, budget * 1.04),
        )
        ax.set_title(
            "Response lengths and stopping behavior",
            loc="left",
            fontsize=16,
            weight="bold",
            pad=30,
        )
        ax.text(
            0,
            1.06,
            "Recorded greedy responses only. Length is not a measure of correctness.",
            transform=ax.transAxes,
            color=SLATE,
            fontsize=10,
        )
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        fig.subplots_adjust(top=0.80, bottom=0.18)
        return _save(fig, directory, "response_lengths")
