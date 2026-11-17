"""Inspect the new model pair without weights; --gpu adds one scoring check."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, help="Optional local pinned config/tokenizer folders."
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Load actual models; requires allocated CUDA."
    )
    parser.add_argument(
        "--qualify",
        action="store_true",
        help="Add synthetic long-shape, update and reload checks; requires --gpu.",
    )
    args = parser.parse_args()
    if args.qualify and not args.gpu:
        parser.error("--qualify requires --gpu")
    scratch = Path("/scratch") / os.environ.get("USER", "")
    if os.environ.get("SLURM_JOB_ID") and scratch.is_dir():
        os.environ.setdefault("HF_HOME", str(scratch / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from src.qwen38.model_upgrade import (
        inspect_upgrade,
        load_upgrade_models,
        load_upgrade_profile,
        score_upgrade_trajectory,
    )
    from src.utils import write_json

    profile = load_upgrade_profile(ROOT)
    tokenizers, configs, compatibility = inspect_upgrade(profile, sources=args.sources)
    report = {"profile": profile, "tokenizer_check": compatibility, "real_model_scoring": None}
    if args.gpu:
        student, teacher = load_upgrade_models(profile, configs)
        report["real_model_scoring"] = score_upgrade_trajectory(
            student,
            teacher,
            tokenizers,
            profile,
            "A store has 24 boxes. Each box contains 15 pencils. The store sells 87 pencils. "
            "How many pencils remain? Explain your reasoning step by step. "
            "End with one final line: #### <number>.",
        )
        if args.qualify:
            from src.qwen38.qualification import qualify_gpu

            report["gpu_qualification"] = qualify_gpu(student, teacher, tokenizers, profile, ROOT)
    directory = (
        ROOT
        / "results"
        / ("qwen38_readiness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "readiness.json", report)
    print(json.dumps(report, indent=2))
    print("Saved:", directory / "readiness.json")


if __name__ == "__main__":
    main()
