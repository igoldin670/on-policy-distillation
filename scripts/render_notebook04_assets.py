"""Regenerate only 04's explanatory PNG/SVG diagrams, never performance curves."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))

from src.opd_plots import plot_observed_failure, render_diagrams  # noqa: E402

if __name__ == "__main__":
    render_diagrams(ROOT / "notebooks/assets/04_on_policy_distillation")
    plot_observed_failure(ROOT / "notebooks/assets/04_on_policy_distillation")
