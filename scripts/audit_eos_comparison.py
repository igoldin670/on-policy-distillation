"""Verify a collected EOS comparison and derive compact, reproducible evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_opd_run import audit_run  # noqa: E402
from src.evaluation import rescore_response_rows  # noqa: E402
from src.opd import validate_trajectory_log  # noqa: E402
from src.training_safety import TrainingSafetyConfig, assess_training_safety  # noqa: E402
from src.utils import artifact_directory, sha256_file, verify_inventory, write_inventory, write_json  # noqa: E402


def audit_comparison(directory, destination, *, baseline_nll):
    directory, destination = Path(directory), Path(destination)
    source_hashes = verify_inventory(directory)
    if destination.exists():
        raise FileExistsError("Choose a fresh audit output directory.")
    destination.mkdir(parents=True)
    before = json.loads((directory / "starting_sft_validation.json").read_text())
    baseline_accuracy = before["scores"]["strict"]["accuracy"]
    policy = TrainingSafetyConfig()
    summary = {
        "source_files_verified": len(source_hashes),
        "baseline_nll": baseline_nll,
        "baseline_nll_source": "Saved Notebook 04 starting-validation stdout",
        "baseline_accuracy": baseline_accuracy,
        "safety_policy": policy.__dict__,
        "scope": "Artifact and frozen-checker replay. Saved model NLL/KL values are read from the GPU history.",
        "arms": {},
    }
    response_groups = {}
    for path in [directory / "starting_sft_validation.json", directory / "teacher_validation.json"]:
        evaluation = json.loads(path.read_text())
        annotations, scores = rescore_response_rows(evaluation["rows"])
        if scores != evaluation["scores"] or annotations != evaluation["annotations"]:
            raise ValueError(f"Saved evaluation metrics do not reproduce: {path.name}")
        response_groups[path.stem] = evaluation["rows"]
    initial_groups = []
    baseline_flags = {row["example_id"]: row["strict_correct"] for row in before["annotations"]}
    for mode in ("raw_token", "semantic_stop"):
        arm = directory / "eos_comparison" / mode
        history = json.loads((arm / "on_policy_history.json").read_text())
        specification = json.loads((arm / "experiment.json").read_text())
        splits = json.loads((arm / "data_splits.json").read_text())
        accumulation = specification["configuration"]["accumulation_steps"]
        count = validate_trajectory_log(
            arm / "training_trajectories.jsonl", splits["train"], len(history), accumulation
        )
        audit = audit_run(arm, response_limit=specification["configuration"]["max_response_length"])
        if any(
            audit[key]
            for key in (
                "trajectory_inconsistencies",
                "history_count_mismatches",
                "history_only_updates",
                "trajectory_only_updates",
            )
        ):
            raise ValueError(f"Trajectory/history consistency failed for {mode}")
        trajectories = [
            json.loads(line)
            for line in (arm / "training_trajectories.jsonl").read_text().splitlines()
        ]
        initial_groups.append(trajectories[:accumulation])
        boundaries = []
        for row in history:
            if "generated_validation" not in row:
                continue
            path = arm / f"on_policy_boundary_{row['step']:04d}_validation.json"
            evaluation = json.loads(path.read_text())
            annotations, scores = rescore_response_rows(evaluation["rows"])
            expected_metrics = {
                "questions": scores["strict"]["number_of_examples"],
                "strict_accuracy": scores["strict"]["accuracy"],
                "format_compliance": scores["strict"]["parse_rate"],
                "prose_numeric_accuracy": scores["prose"]["accuracy"],
                "prose_parse_rate": scores["prose"]["parse_rate"],
                "token_limit_stops": scores["prose"]["number_token_limit_stops"],
            }
            if (
                scores != evaluation["scores"]
                or annotations != evaluation["annotations"]
                or any(
                    row["generated_validation"][key] != value
                    for key, value in expected_metrics.items()
                )
            ):
                raise ValueError(f"Boundary metrics do not reproduce: {mode}/{row['step']}")
            gained = [
                r["example_id"]
                for r in annotations
                if r["strict_correct"] and not baseline_flags[r["example_id"]]
            ]
            lost = [
                r["example_id"]
                for r in annotations
                if not r["strict_correct"] and baseline_flags[r["example_id"]]
            ]
            boundaries.append(
                {
                    "step": row["step"],
                    "strict_correct": scores["strict"]["number_correct"],
                    "improved_question_ids": gained,
                    "regressed_question_ids": lost,
                    "validation_reference_nll": row["validation_reference_nll"],
                    "validation_token_limit_stops": row["generated_validation"][
                        "token_limit_stops"
                    ],
                }
            )
            response_groups[f"{mode}_step_{row['step']:04d}"] = evaluation["rows"]
        first_pause = None
        for index in range(len(history)):
            health = assess_training_safety(
                history[: index + 1], baseline_nll, baseline_accuracy, policy
            )
            if health["pause"]:
                first_pause = {"step": history[index]["step"], "reasons": health["reasons"]}
                break
        summary["arms"][mode] = {
            "updates": len(history),
            "trajectories": count,
            "trajectory_integrity": "passed",
            "frozen_checker_replay": "passed",
            "gradient_clipped_updates": sum(
                row["grad_norm_before_clipping"] > 1 for row in history
            ),
            "first_guarded_pause_replay": first_pause,
            "validation_boundaries": boundaries,
            "first_responses": audit["first_responses"],
            "latest_responses": audit["latest_responses"],
            "most_repetitive_responses": audit["most_repetitive_responses"],
        }
        write_json(destination / f"{mode}_history.json", history)
    summary["initial_rollouts_identical"] = initial_groups[0] == initial_groups[1]
    if not summary["initial_rollouts_identical"]:
        raise ValueError("Initial policies did not generate identical comparison trajectories.")
    # Full decoded validation answers allow public checker replay. Token arrays
    # remain available in the original collected archive and its source hashes.
    text_groups = {
        name: [
            {
                key: row[key]
                for key in (
                    "example_id",
                    "source_index",
                    "question",
                    "expected_answer",
                    "response_text",
                    "response_length",
                    "ended_with_eos",
                    "hit_token_limit",
                )
            }
            for row in rows
        ]
        for name, rows in response_groups.items()
    }
    write_json(destination / "validation_responses.json", text_groups)
    write_json(destination / "audit_summary.json", summary)
    write_json(destination / "source_SHA256SUMS.json", source_hashes)
    write_json(
        destination / "source_collection.json",
        json.loads((directory / "collection.json").read_text()),
    )
    write_inventory(destination)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--baseline-nll", type=float, required=True)
    args = parser.parse_args()
    source = artifact_directory(args.source, ROOT / ".cache/eos-evidence", max_bytes=128 * 2**20)
    result = audit_comparison(source, args.destination, baseline_nll=args.baseline_nll)
    print(json.dumps(result, indent=2))
    print(
        "Source archive SHA256:",
        sha256_file(args.source) if args.source.is_file() else "source folder",
    )


if __name__ == "__main__":
    main()
