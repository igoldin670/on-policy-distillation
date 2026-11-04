"""Read saved 04 rollouts without loading models or modifying an experiment.

Usage: python scripts/audit_opd_run.py /path/to/results/on_policy_distillation_...
This reports symptoms, not answer correctness or a proven cause of degeneration.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def audit_run(directory: Path, *, response_limit: int = 256, preview_count: int = 3):
    """Group saved trajectories by the student update that generated them."""
    if response_limit < 1 or preview_count < 1:
        raise ValueError("Response limit and preview count must be positive.")
    groups = defaultdict(list)
    previews = []
    inconsistencies = []
    incomplete_last_line = False
    path = directory / "training_trajectories.jsonl"
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A file copied during an append may end in an unfinished record.
                # Report that explicitly; malformed complete records are errors.
                if not line.endswith("\n"):
                    incomplete_last_line = True
                    break
                raise
            response = row["response_token_ids"]
            if not response:
                raise ValueError(f"Empty response at line {line_number}.")
            expected_eos = response[-1] in {151643, 151645}
            expected_cap = len(response) == response_limit and not expected_eos
            if (
                row["response_length"] != len(response)
                or row["ended_with_eos"] != expected_eos
                or row["hit_token_limit"] != expected_cap
                or len(response) > response_limit
                or row["complete_sequence_ids"] != row["prompt_token_ids"] + response
                or row["prompt_length"] != len(row["prompt_token_ids"])
            ):
                inconsistencies.append({"line": line_number, "example_id": row["example_id"]})
            groups[row["policy_step"]].append(row)
            # Repeated four-token windows are a clue, not proof of a bad answer.
            # Mathematical explanations can legitimately repeat phrases.
            windows = [tuple(response[i : i + 4]) for i in range(len(response) - 3)]
            repeated_fraction = 1 - len(set(windows)) / len(windows) if windows else 0.0
            previews.append(
                {
                    "update": row["policy_step"] + 1,
                    "example_id": row["example_id"],
                    "response_tokens": len(response),
                    "ended_with_eos": row["ended_with_eos"],
                    "hit_token_limit": row["hit_token_limit"],
                    "last_token_id": response[-1],
                    "repeated_four_token_fraction": round(repeated_fraction, 4),
                    "text_start": row["response_text"][:400],
                    "text_end": row["response_text"][-600:],
                }
            )
    if not groups:
        raise ValueError("No complete trajectory records found.")
    updates = []
    for policy_step, rows in sorted(groups.items()):
        lengths = [len(row["response_token_ids"]) for row in rows]
        updates.append(
            {
                "update": policy_step + 1,
                "responses": len(rows),
                "response_tokens": sum(lengths),
                "mean_response_length": sum(lengths) / len(lengths),
                "token_limit_stops": sum(row["hit_token_limit"] for row in rows),
                "eos_stops": sum(row["ended_with_eos"] for row in rows),
            }
        )
    history_path = directory / "on_policy_history.json"
    history_mismatches = []
    history_only_updates = []
    trajectory_only_updates = []
    validation = []
    if history_path.exists():
        history = json.loads(history_path.read_text())
        by_update = {row["update"]: row for row in updates}
        history_steps = {row["step"] for row in history}
        history_only_updates = sorted(history_steps - by_update.keys())
        trajectory_only_updates = sorted(by_update.keys() - history_steps)
        for record in history:
            observed = by_update.get(record["step"])
            if observed and any(
                record[key] != observed[key] for key in ("response_tokens", "token_limit_stops")
            ):
                history_mismatches.append(record["step"])
            if "validation_reference_nll" in record:
                validation.append(
                    {
                        key: record[key]
                        for key in ("step", "validation_reference_nll", "fixed_validation_kl")
                        if key in record
                    }
                )
    return {
        "scope": "Saved rollouts only. No model execution or answer-quality claim.",
        "response_limit": response_limit,
        "trajectory_count": len(previews),
        "incomplete_last_line": incomplete_last_line,
        "trajectory_inconsistencies": inconsistencies,
        "history_present": history_path.exists(),
        "history_count_mismatches": history_mismatches,
        "history_only_updates": history_only_updates,
        "trajectory_only_updates": trajectory_only_updates,
        "updates": updates,
        "validation_progress": validation,
        "first_responses": previews[:preview_count],
        "latest_responses": previews[-preview_count:],
        "most_repetitive_responses": sorted(
            previews, key=lambda row: row["repeated_four_token_fraction"], reverse=True
        )[:preview_count],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--response-limit", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.directory, response_limit=args.response_limit), indent=2))


if __name__ == "__main__":
    main()
