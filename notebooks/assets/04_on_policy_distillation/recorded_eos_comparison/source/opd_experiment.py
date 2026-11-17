"""Recovery and comparison helpers for 04. No model weights load on import."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from src.evaluation import rescore_response_rows
from src.opd import experiment_signature, read_resume
from src.utils import verify_inventory, write_json


class WalltimePause(RuntimeError):
    """The allocation is entering the time reserved for saving state."""


@dataclass(frozen=True)
class SlurmBudget:
    end_time: float | None = None
    reserve_seconds: float = 600

    @classmethod
    def from_environment(cls):
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            return cls()
        result = subprocess.run(
            ["scontrol", "show", "job", job, "--oneliner"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        match = re.search(r"\bEndTime=(\S+)", result.stdout)
        if not match or match[1] in {"Unknown", "N/A", "None"}:
            raise RuntimeError(
                "Could not determine the SLURM deadline. Check scontrol before training."
            )
        # scontrol and datetime use the same host timezone for an unzoned time.
        return cls(datetime.fromisoformat(match[1]).timestamp())

    def require_time(self):
        if self.end_time is not None and self.end_time - time.time() <= self.reserve_seconds:
            raise WalltimePause("SLURM checkpoint reserve reached (ten minutes).")


@contextmanager
def isolated_evaluation_rng():
    """Validation cannot consume the random stream used by future rollouts."""
    import torch

    python_state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)


def _evaluation_identity(examples, policy, identity):
    return hashlib.sha256(
        json.dumps(
            {"examples": examples, "policy": policy, "checkpoint_identity": identity},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _validate_rows(rows, examples, tokenizer, policy):
    from src.baseline import validate_generation_ids

    if len(rows) > len(examples):
        raise ValueError("Evaluation contains more responses than questions.")
    eos_ids = policy["eos_token_id"]
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
    for row, example in zip(rows, examples):
        if any(
            row[key] != example[key]
            for key in ("example_id", "source_index", "question", "prompt", "expected_answer")
        ):
            raise ValueError("Cached evaluation question identity differs.")
        prompt = tokenizer.encode(example["prompt"], add_special_tokens=False)
        response = row["response_token_ids"]
        validate_generation_ids(prompt, prompt + response, policy["max_new_tokens"])
        ended = response[-1] in eos_ids
        if (
            row["prompt_token_ids"] != prompt
            or row["prompt_length"] != len(prompt)
            or row["response_length"] != len(response)
            or row["ended_with_eos"] != ended
            or row["hit_token_limit"] != (len(response) == policy["max_new_tokens"] and not ended)
            or row["response_text"] != tokenizer.decode(response, skip_special_tokens=True)
        ):
            raise ValueError("Cached evaluation tokens or stopping flags differ.")


def evaluate_incrementally(
    model,
    tokenizer,
    examples,
    policy,
    device,
    dtype,
    *,
    path,
    identity,
    budget=None,
    generator=None,
):
    """Atomically save each answer, and resume only identical checkpoint/question policies."""
    from src.sft import generate_evaluation

    if not examples:
        raise ValueError("Evaluation needs at least one question.")
    budget = budget or SlurmBudget()
    generator = generator or generate_evaluation
    path = Path(path)
    partial = path.with_suffix(".partial.json")
    fingerprint = _evaluation_identity(examples, policy, identity)
    saved_path = path if path.exists() else partial
    rows = []
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        if saved.get("evaluation_identity") != fingerprint:
            raise ValueError("Evaluation checkpoint, questions, or generation policy changed.")
        rows = saved["rows"]
        _validate_rows(rows, examples, tokenizer, policy)
        if saved_path == path:
            annotations, scores = rescore_response_rows(rows)
            if (
                len(rows) != len(examples)
                or saved["scores"] != scores
                or saved["annotations"] != annotations
            ):
                raise ValueError("Completed evaluation metrics do not reproduce.")
            return saved
    with isolated_evaluation_rng():
        for index in range(len(rows), len(examples)):
            budget.require_time()
            if index == 0 or (index + 1) % 16 == 0 or index + 1 == len(examples):
                print(f"{path.stem}: evaluating {index + 1}/{len(examples)}", flush=True)
            # Suppress only the reused generator's redundant '1/1' progress line.
            captured = io.StringIO()
            try:
                with redirect_stdout(captured):
                    result = generator(
                        model,
                        tokenizer,
                        [examples[index]],
                        policy,
                        device,
                        dtype,
                        use_model_defaults=False,
                    )
            except BaseException:
                print(captured.getvalue(), end="")
                raise
            if len(result["rows"]) != 1:
                raise ValueError("A one-question evaluation must return one response.")
            rows.extend(result["rows"])
            _validate_rows(rows, examples, tokenizer, policy)
            write_json(partial, {"evaluation_identity": fingerprint, "rows": rows})
    annotations, scores = rescore_response_rows(rows)
    result = {
        "evaluation_identity": fingerprint,
        "rows": rows,
        "annotations": annotations,
        "scores": scores,
    }
    write_json(path, result)
    return result


def commit_boundary(directory, student, tokenizer, optimizer, scheduler, scaler, **metadata):
    """An existing immutable boundary can be reused after interrupted validation."""
    from src.opd import save_resume

    if len(metadata["history"]) != metadata["step"]:
        raise ValueError("Current history differs from the requested optimizer boundary.")
    snapshot = Path(directory) / "resume" / metadata["arm"] / f"step_{metadata['step']:04d}"
    if snapshot.exists():
        saved = read_resume(snapshot, arm=metadata["arm"], signature=metadata["signature"])
        if saved["completed_steps"] != metadata["step"]:
            raise ValueError("Snapshot boundary differs.")
        # Diagnostic fields may be completed after the immutable state was saved.
        for old, current in zip(saved["history"], metadata["history"]):
            if any(current.get(key) != value for key, value in old.items()):
                raise ValueError("Saved optimizer-boundary history changed.")
        return snapshot
    save_resume(directory, student, tokenizer, optimizer, scheduler, scaler, **metadata)
    return snapshot


def publish_pause(directory, *, arm, status, step, snapshot, signature, config, health, reason):
    result = {
        "arm": arm,
        "status": status,
        "completed_steps": step,
        "resume_snapshot": str(snapshot) if snapshot is not None else None,
        "signature": signature,
        "configuration": asdict(config),
        "health": health,
        "pause_reason": reason,
        "validation_status": "pending",
        "checkpoint_reload_verified": False,
    }
    write_json(Path(directory) / f"{arm}_pause.json", result)
    write_json(Path(directory) / "manifest.json", result)
    return result


def verify_historical_run(directory, checkpoint_hashes, protocol, splits, config=None):
    """Read-only checks before reusing recorded examples/control provenance."""
    from src.opd import validate_trajectory_log

    directory = Path(directory)
    specification = json.loads((directory / "experiment.json").read_text())
    historical_splits = json.loads((directory / "data_splits.json").read_text())
    if (
        specification["protocol"] != protocol
        or specification["starting_checkpoint_hashes"] != checkpoint_hashes
    ):
        raise ValueError("Historical source checkpoint/protocol differs.")
    for name in ("train", "validation", "primary_test", "historical_test"):
        if historical_splits[name] != splits[name]:
            raise ValueError(f"Historical split differs: {name}")
    if config is not None:
        current = asdict(config)
        if any(
            current.get(key) != value
            for key, value in specification["configuration"].items()
            if key not in {"eos_loss_mode", "eos_token_ids"}
        ):
            raise ValueError("Historical SFT control training recipe differs.")
    snapshots = {}
    for step in (16, 22):
        path = directory / "resume/on_policy" / f"step_{step:04d}"
        metadata = read_resume(path, arm="on_policy", signature=specification["signature"])
        if metadata["completed_steps"] != step:
            raise ValueError("Historical snapshot is stored under the wrong update number.")
        snapshots[str(step)] = str(path)
    history = json.loads((directory / "on_policy_history.json").read_text())
    if len(history) != 22 or any(
        any(current.get(key) != value for key, value in saved.items())
        for saved, current in zip(metadata["history"], history)
    ):
        raise ValueError("Historical history differs from its final saved optimizer boundary.")
    trajectory_count = validate_trajectory_log(
        directory / "training_trajectories.jsonl",
        historical_splits["train"],
        22,
        specification["configuration"]["accumulation_steps"],
    )
    control_path = directory / "checkpoints/sft_control"
    verify_inventory(control_path)
    control = json.loads((directory / "sft_control_training.json").read_text())
    if control["status"] != "completed" or not control["checkpoint_reload_verified"]:
        raise ValueError("Historical SFT control is not recorded complete and verified.")
    return {
        "directory": str(directory),
        "signature": specification["signature"],
        "snapshots": snapshots,
        "control_checkpoint": str(control_path),
        "control": control,
        "verified_trajectory_count": trajectory_count,
    }


def check_comparison_storage(directory, *, minimum_free_gib=40):
    """Filesystem capacity is a necessary check; personal HPC quotas need separate review."""
    free = shutil.disk_usage(directory).free / 2**30
    if free < minimum_free_gib:
        raise RuntimeError(
            f"EOS comparison needs at least {minimum_free_gib} GiB free; found {free:.1f}."
        )
    return {
        "filesystem_free_gib": free,
        "personal_quota_verified": False,
        "note": "Review HPC quota before additional large recovery snapshots.",
    }


def run_eos_comparison(
    train_arm,
    teacher,
    *,
    directory,
    config,
    protocol,
    splits,
    checkpoint_hashes,
    evaluation_policy,
    code_hashes,
    starting_validation_nll,
):
    """Two fresh student policies, one loaded teacher, one controlled configuration change."""
    directory = Path(directory)
    results, arm_signatures = {}, {}
    for mode in ("raw_token", "semantic_stop"):
        recipe = replace(config, eos_loss_mode=mode)
        child = directory / "eos_comparison" / mode
        child.mkdir(parents=True, exist_ok=True)
        signature = experiment_signature(
            recipe, protocol, splits, checkpoint_hashes, evaluation_policy, code_hashes
        )
        arm_signatures[mode] = signature
        specification = json.loads(
            json.dumps(
                {
                    "signature": signature,
                    "configuration": asdict(recipe),
                    "eos_loss_mode": mode,
                    "protocol": protocol,
                    "starting_checkpoint_hashes": checkpoint_hashes,
                    "evaluation_policy": evaluation_policy,
                    "code_hashes": code_hashes,
                },
                allow_nan=False,
            )
        )
        path = child / "experiment.json"
        if path.exists() and json.loads(path.read_text()) != specification:
            raise ValueError("EOS comparison recipe changed. Start a fresh comparison.")
        write_json(path, specification)
        split_path = child / "data_splits.json"
        if not split_path.exists():
            write_json(split_path, splits)
        result = train_arm(
            "on_policy",
            teacher=teacher,
            stop_after_steps=min(recipe.pilot_updates, recipe.optimizer_steps),
            run_config=recipe,
            output_directory=child,
            run_signature=signature,
            recover_pause_validation=True,
        )
        results[mode] = result
        write_json(directory / "eos_comparison_results.json", results)
        if result.get("validation_status") != "completed" or result["status"] == "walltime_paused":
            break
    gates = comparison_gates(results, config, starting_validation_nll)
    gates["arm_signatures"] = arm_signatures
    if set(results) == {"raw_token", "semantic_stop"} and all(
        result["completed_steps"] > 0 for result in results.values()
    ):
        # The initial policies, prompts and RNG are identical. Verify their first
        # accumulation group before interpreting later policy divergence.
        initial_groups = []
        for mode in ("raw_token", "semantic_stop"):
            path = directory / "eos_comparison" / mode / "training_trajectories.jsonl"
            initial_groups.append(
                [json.loads(line) for line in path.read_text().splitlines()][
                    : config.accumulation_steps
                ]
            )
        matched = initial_groups[0] == initial_groups[1]
        gates["initial_rollout_groups_identical"] = matched
        gates["ready_for_full_run"] = gates["ready_for_full_run"] and matched
    write_json(directory / "eos_comparison_gates.json", gates)
    return {"results": results, "gates": gates}


def comparison_gates(results, config, starting_validation_nll):
    """Explicit engineering acceptance thresholds, conditional on one training seed."""
    semantic = results.get("semantic_stop")
    raw = results.get("raw_token")
    if (
        not semantic
        or not raw
        or any(r.get("validation_status") != "completed" for r in (raw, semantic))
    ):
        return {"ready_for_full_run": False, "status": "incomplete"}

    def flags(result, step):
        return [
            flag
            for row in result["history"]
            if row["step"] <= step
            for flag in row["response_limit_flags"]
        ]

    common_step = min(raw["completed_steps"], semantic["completed_steps"])
    raw_recent = flags(raw, common_step)[-config.cap_window_responses :]
    semantic_common = flags(semantic, common_step)[-config.cap_window_responses :]
    final_recent = flags(semantic, semantic["completed_steps"])[-config.cap_window_responses :]
    if not raw_recent or not semantic_common:
        return {"ready_for_full_run": False, "status": "incomplete"}
    contrast = sum(raw_recent) / len(raw_recent) - sum(semantic_common) / len(semantic_common)
    nll = semantic["final_validation_reference_nll"]
    gates = {
        "completed_pilot": semantic["status"] == "pilot_complete"
        and semantic["completed_steps"] == min(config.pilot_updates, config.optimizer_steps),
        "finite_updates": all(
            math.isfinite(row[key])
            for row in semantic["history"]
            for key in ("loss", "grad_norm_before_clipping")
        ),
        "final_cap_rate_at_most_25_percent": len(final_recent) == config.cap_window_responses
        and sum(final_recent) / len(final_recent) <= 0.25,
        "nll_within_10_percent": math.isfinite(nll) and nll <= 1.1 * starting_validation_nll,
        "strict_validation_at_least_50_percent": semantic["final_validation"]["strict_accuracy"]
        >= 0.5,
        "meaningful_matched_cap_reduction": len(raw_recent) == config.cap_window_responses
        and contrast >= 0.25,
    }
    return {
        "ready_for_full_run": all(gates.values()),
        "status": "review_required",
        "gates": gates,
        "common_update": common_step,
        "cap_rate_reduction": contrast,
        "scope": "Engineering gates on one seed; held-out quality evaluation remains pending.",
    }
