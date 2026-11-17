"""Validation-based training pauses and checkpoint-bound measurement recovery."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from src.utils import sha256_file, write_json


@dataclass(frozen=True)
class TrainingSafetyConfig:
    version: int = 2
    max_nll_ratio: float = 1.10
    max_accuracy_drop: float = 0.05
    max_cap_fraction: float = 0.25
    response_window: int = 32

    def __post_init__(self):
        if self.version != 2:
            raise ValueError("Unsupported training safety policy version.")
        if not math.isfinite(self.max_nll_ratio) or self.max_nll_ratio <= 1:
            raise ValueError("NLL ratio must be finite and greater than one.")
        if any(
            not math.isfinite(v) or not 0 <= v <= 1
            for v in (
                self.max_accuracy_drop,
                self.max_cap_fraction,
            )
        ):
            raise ValueError("Accuracy/cap limits must be finite fractions in [0, 1].")
        if type(self.response_window) is not int or self.response_window < 1:
            raise ValueError("The response window must be a positive integer.")


def assess_training_safety(history, baseline_nll, baseline_accuracy, policy):
    """Engineering review thresholds relative to this run's starting checkpoint."""
    if not math.isfinite(baseline_nll) or baseline_nll <= 0:
        raise ValueError("A finite positive starting NLL is required.")
    if not math.isfinite(baseline_accuracy) or not 0 <= baseline_accuracy <= 1:
        raise ValueError("Starting strict accuracy must be a fraction in [0, 1].")
    flags = [flag for row in history for flag in row.get("response_limit_flags", [])]
    recent = flags[-policy.response_window :]
    reasons = []
    if len(recent) == policy.response_window:
        fraction = sum(recent) / len(recent)
        if fraction > policy.max_cap_fraction:
            reasons.append(
                {
                    "metric": "recent_cap_fraction",
                    "observed": fraction,
                    "limit": policy.max_cap_fraction,
                }
            )
    measured = [row for row in history if "validation_reference_nll" in row]
    if measured:
        record = measured[-1]
        nll = record["validation_reference_nll"]
        if not math.isfinite(nll):
            raise FloatingPointError("Nonfinite validation NLL cannot qualify a checkpoint.")
        if nll > baseline_nll * policy.max_nll_ratio:
            reasons.append(
                {
                    "metric": "validation_reference_nll",
                    "observed": nll,
                    "limit": baseline_nll * policy.max_nll_ratio,
                }
            )
        validation = record.get("generated_validation")
        if validation is not None:
            accuracy = validation["strict_accuracy"]
            if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("Invalid generated validation accuracy.")
            if accuracy < baseline_accuracy - policy.max_accuracy_drop - 1e-12:
                reasons.append(
                    {
                        "metric": "strict_accuracy",
                        "observed": accuracy,
                        "limit": max(0.0, baseline_accuracy - policy.max_accuracy_drop),
                    }
                )
    return {
        "pause": bool(reasons),
        "reasons": reasons,
        "warnings": [f"Training pause requested by {r['metric']}." for r in reasons],
        "recent_responses": len(recent),
        "recent_capped": sum(recent),
        "baseline_nll": baseline_nll,
        "baseline_accuracy": baseline_accuracy,
        "safety_policy": asdict(policy),
    }


def boundary_identity(snapshot, signature, policy):
    return {
        "signature": signature,
        "safety_policy": asdict(policy),
        "snapshot_inventory_sha256": sha256_file(Path(snapshot) / "SHA256SUMS.json"),
    }


def save_boundary_measurements(directory, record, snapshot, signature, policy, *, complete):
    """Publish derived metrics separately from the immutable optimizer snapshot."""
    fields = (
        "validation_reference_nll",
        "generated_validation",
        "fixed_validation_kl",
        "fixed_raw_token_kl",
        "fixed_semantic_stop_kl",
    )
    payload = {
        **boundary_identity(snapshot, signature, policy),
        "step": record["step"],
        "validation_complete": complete,
        "measurements": {key: record[key] for key in fields if key in record},
    }
    write_json(Path(directory) / f"boundary_{record['step']:04d}_metrics.json", payload)


def restore_boundary_measurements(directory, history, snapshot, signature, policy):
    """Validate measurement identity before health checks or resumed updates."""
    if not history:
        return
    path = Path(directory) / f"boundary_{history[-1]['step']:04d}_metrics.json"
    if not path.exists():
        return
    saved = json.loads(path.read_text())
    expected = boundary_identity(snapshot, signature, policy)
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved validation belongs to different weights or safety settings.")
    if saved["step"] != history[-1]["step"]:
        raise ValueError("Saved validation step differs from the optimizer boundary.")
    for key, value in saved["measurements"].items():
        if key in history[-1] and history[-1][key] != value:
            raise ValueError("Immutable history and saved validation disagree.")
        history[-1][key] = value


def accept_checkpoint(directory, snapshot, record, signature, policy):
    """Pin only a fully measured acceptable optimizer boundary."""
    if "generated_validation" not in record:
        raise ValueError("Complete generated validation before accepting a checkpoint.")
    relative = Path(snapshot).resolve().relative_to(Path(directory).resolve())
    write_json(
        Path(directory) / "last_acceptable.json",
        {
            **boundary_identity(snapshot, signature, policy),
            "step": record["step"],
            "snapshot": str(relative),
            "validation": record["generated_validation"],
            "validation_reference_nll": record["validation_reference_nll"],
        },
    )


def protected_checkpoint(directory, signature):
    path = Path(directory) / "last_acceptable.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved["signature"] != signature:
        raise ValueError("Accepted checkpoint belongs to a different experiment.")
    snapshot = Path(saved["snapshot"])
    if not snapshot.is_absolute():
        snapshot = Path(directory) / snapshot
    if sha256_file(snapshot / "SHA256SUMS.json") != saved["snapshot_inventory_sha256"]:
        raise ValueError("Accepted checkpoint inventory changed.")
    return snapshot.resolve()


def pin_starting_checkpoint(directory, checkpoint, signature, policy):
    path = Path(directory) / "last_acceptable.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("step") == 0:
            if saved["signature"] != signature or saved["snapshot_inventory_sha256"] != sha256_file(
                Path(checkpoint) / "SHA256SUMS.json"
            ):
                raise ValueError("Starting checkpoint identity changed while resuming.")
            # Verified baseline weights may live at a new path after transfer.
            saved["snapshot"] = str(Path(checkpoint).resolve())
            write_json(path, saved)
        protected_checkpoint(directory, signature)
        return
    write_json(
        path,
        {
            **boundary_identity(checkpoint, signature, policy),
            "step": 0,
            "snapshot": str(Path(checkpoint).resolve()),
            "source": "verified_starting_checkpoint",
        },
    )


def record_training_failure(directory, *, error, step, snapshot, signature):
    directory = Path(directory)
    status = (
        "interrupted"
        if isinstance(error, KeyboardInterrupt)
        else ("numerical_failed" if isinstance(error, FloatingPointError) else "runtime_failed")
    )
    payload = {
        "status": status,
        "completed_steps": step,
        "signature": signature,
        "error_type": type(error).__name__,
        "error": str(error),
        "resume_snapshot": str(snapshot) if snapshot is not None else None,
        "last_acceptable_snapshot": str(protected_checkpoint(directory, signature))
        if (directory / "last_acceptable.json").exists()
        else None,
    }
    progress = directory / "progress.jsonl"
    if progress.exists():
        lines = progress.read_text().splitlines()
        payload["last_recorded_stage"] = json.loads(lines[-1]) if lines else None
    write_json(directory / "failure.json", payload)
    write_json(directory / "manifest.json", payload)
    return payload


def assert_finite_parameters(model):
    import torch

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Nonfinite student parameter after update: {name}")


def capture_update_rng():
    """Keep the clean optimizer boundary's RNG before generating a new group."""
    import torch

    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_update_rng(state):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
