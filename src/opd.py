"""Notebook 04 orchestration helpers. Importing this module loads no model weights.

The notebook keeps the optimizer loop visible. These helpers isolate artifact
checks, data-independent tensor operations, restart state, and report packaging.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.logprobs import response_prediction_logits
from src.opd_loss import forward_kl_loss
from src.sft import collate_examples, precision_context, response_loss_sum
from src.utils import artifact_directory, verify_inventory, write_inventory, write_json

TEACHER_ID = "Qwen/Qwen3-32B"
TEACHER_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
TEACHER_WEIGHT_BYTES = 65_535_751_214


@dataclass(frozen=True)
class OPDConfig:
    seed: int = 42
    train_examples: int = 1024
    test_examples: int = 256
    accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1
    validation_interval: int = 32
    checkpoint_interval: int = 32
    max_prompt_length: int = 512
    max_response_length: int = 256
    max_sequence_length: int = 1024
    teacher_temperature: float = 0.1
    kl_chunk_size: int = 16
    diagnostic_examples: int = 8

    def __post_init__(self):
        for name in (
            "train_examples",
            "test_examples",
            "accumulation_steps",
            "validation_interval",
            "checkpoint_interval",
            "max_prompt_length",
            "max_response_length",
            "max_sequence_length",
            "kl_chunk_size",
            "diagnostic_examples",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_prompt_length + self.max_response_length > self.max_sequence_length:
            raise ValueError("Prompt + rollout limits exceed the complete-sequence budget.")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("Warmup fraction must be in [0, 1).")
        for name in ("learning_rate", "max_grad_norm", "teacher_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")

    @property
    def optimizer_steps(self):
        return math.ceil(self.train_examples / self.accumulation_steps)


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # This reduces a source of numerical variation; cross-hardware bitwise
    # reproducibility is still not promised by PyTorch/CUDA.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False


def load_sft_report(path: Path, cache: Path) -> tuple[Path, dict]:
    from src.evaluation import rescore_response_rows

    directory = artifact_directory(path, cache, max_bytes=64 * 2**20)
    verify_inventory(directory)

    def read(name):
        return json.loads((directory / name).read_text())

    manifest, protocol, splits = (
        read("manifest.json"),
        read("protocol.json"),
        read("data_splits.json"),
    )
    if manifest.get("status") != "completed" or manifest.get("method") != "reference_solution_sft":
        raise ValueError("04 requires a completed Notebook 03 SFT report.")
    if (
        manifest["student_model_id"] != protocol["student_model_id"]
        or manifest["student_revision"] != protocol["student_revision"]
    ):
        raise ValueError("SFT model identity differs between manifest and protocol.")
    if not manifest.get("checkpoint_reload_verified"):
        raise ValueError("The SFT report does not record a verified saved checkpoint.")
    for name in ("before_validation", "after_validation", "final_test"):
        evaluation = read(name + ".json")
        _, scores = rescore_response_rows(evaluation["rows"])
        if scores != evaluation["scores"]:
            raise ValueError(f"SFT saved metrics do not reproduce: {name}")
    return directory, {
        "manifest": manifest,
        "protocol": protocol,
        "splits": splits,
        "checkpoint_hashes": read("checkpoint_SHA256SUMS.json"),
    }


def resolve_cold_start(root: Path, *, report_path=None, checkpoint_path=None):
    """Use explicit folder/ZIP paths, or the newest complete 03 runtime directory."""
    root = Path(root)
    if report_path is None and checkpoint_path is None:
        candidates = sorted((root / "results").glob("sft_cold_start_*"), reverse=True)
        match = next(
            (
                p
                for p in candidates
                if (p / "report/manifest.json").is_file()
                and (p / "checkpoint/model.safetensors").is_file()
            ),
            None,
        )
        if match is not None:
            report_path, checkpoint_path = match / "report", match / "checkpoint"
    if report_path is None or checkpoint_path is None:
        raise FileNotFoundError(
            "Set SFT_REPORT_PATH and SFT_CHECKPOINT_PATH to server-side folders or ZIPs. "
            "report.zip contains evidence, not weights. Upload checkpoint.zip if the old Colab runtime is gone."
        )
    report_directory, evidence = load_sft_report(Path(report_path), root / ".cache/opd-inputs")
    checkpoint_directory = artifact_directory(Path(checkpoint_path), root / ".cache/opd-inputs")
    verify_inventory(checkpoint_directory, evidence["checkpoint_hashes"])
    model_config = json.loads((checkpoint_directory / "config.json").read_text())
    if model_config.get("model_type") != "qwen3":
        raise ValueError("Expected the saved Qwen3 student, not a substituted checkpoint.")
    return report_directory, checkpoint_directory, evidence


def cuda_preflight(root: Path) -> dict:
    """Fail early for the chosen unquantized 32B teacher; never change models silently."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training needs the Colab CUDA kernel. Local inspection uses no weights."
        )
    free, total = torch.cuda.mem_get_info()
    if total < 75 * 2**30 or free < 73 * 2**30:
        raise RuntimeError(
            f"This unquantized 32B-teacher experiment needs an almost-empty A100 80GB. "
            f"Available: {free / 2**30:.1f}/{total / 2**30:.1f} GiB. "
            "Restart the kernel to release earlier models. A100 40GB/L4 are insufficient for this recipe."
        )
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    blobs = cache / "hub/models--Qwen--Qwen3-32B/blobs"
    cached_bytes = sum(p.stat().st_size for p in blobs.glob("*") if p.is_file())
    required_disk = max(0, TEACHER_WEIGHT_BYTES - cached_bytes) + 40 * 2**30
    disk_free = shutil.disk_usage(root).free
    if disk_free < required_disk:
        raise RuntimeError(
            f"Need about {required_disk / 2**30:.0f} GiB free disk for teacher/cache/exports; have {disk_free / 2**30:.0f}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The fixed 32B-teacher recipe requires BF16 support.")
    return {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": total / 2**30,
        "gpu_free_gib_at_start": free / 2**30,
        "bf16_supported": True,
        "disk_free_gib_at_start": disk_free / 2**30,
        "memory_estimate_is_not_an_execution_guarantee": True,
    }


def load_student(checkpoint_directory: Path, device):
    import torch
    from transformers import AutoModelForCausalLM

    student = AutoModelForCausalLM.from_pretrained(
        checkpoint_directory,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(student.config, "attention_dropout", 0.0) != 0.0:
        raise ValueError("This recipe assumes dropout-free sampling/scoring consistency.")
    assert all(p.dtype == torch.float32 and p.requires_grad for p in student.parameters())
    return student.train()


def load_teacher(device):
    import torch
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID,
        revision=TEACHER_REVISION,
        token=False,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    teacher.requires_grad_(False).eval()
    teacher.config.use_cache = False
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    return teacher


def trajectory_tensors(trajectory, device):
    import torch

    prompt, response = trajectory["prompt_token_ids"], trajectory["response_token_ids"]
    if not prompt or not response or trajectory["complete_sequence_ids"] != prompt + response:
        raise ValueError("Trajectory must preserve its exact prompt and sampled response IDs.")
    if trajectory["prompt_length"] != len(prompt) or trajectory["response_length"] != len(response):
        raise ValueError("Trajectory length metadata differs from its token IDs.")
    ids = torch.tensor([prompt + response], dtype=torch.long, device=device)
    return ids, torch.ones_like(ids)


def trajectory_kl(student, teacher, trajectory, device, dtype, config):
    """Teacher and student see identical IDs, attention masks, and causal prefixes."""
    import torch

    complete_sequence_ids, attention_mask = trajectory_tensors(trajectory, device)
    prompt_length = trajectory["prompt_length"]
    teacher.eval()
    with torch.no_grad(), precision_context(device, dtype):
        teacher_logits = teacher(
            input_ids=complete_sequence_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        teacher_response_logits = response_prediction_logits(
            teacher_logits, complete_sequence_ids, prompt_length
        ).contiguous()
    del teacher_logits
    with precision_context(device, dtype):
        student_logits = student(
            input_ids=complete_sequence_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        student_response_logits = response_prediction_logits(
            student_logits, complete_sequence_ids, prompt_length
        )
        loss = forward_kl_loss(
            student_response_logits,
            teacher_response_logits,
            teacher_temperature=config.teacher_temperature,
            chunk_size=config.kl_chunk_size,
        )
    return loss


def backward_opd_group(student, teacher, trajectories, device, dtype, config, scaler, policy_step):
    """Mean of response means: long responses do not outweigh short responses."""
    if not trajectories or any(t["policy_step"] != policy_step for t in trajectories):
        raise ValueError("Rollouts must come from the current, unchanged student policy.")
    total_loss = 0.0
    for trajectory in trajectories:
        loss = trajectory_kl(student, teacher, trajectory, device, dtype, config)
        scaler.scale(loss / len(trajectories)).backward()
        total_loss += float(loss.detach())
    if any(p.grad is not None for p in teacher.parameters()):
        raise RuntimeError("The frozen teacher accumulated a gradient.")
    return {
        "loss": total_loss / len(trajectories),
        "response_tokens": sum(t["response_length"] for t in trajectories),
        "examples": len(trajectories),
        "token_limit_stops": sum(t["hit_token_limit"] for t in trajectories),
    }


def backward_sft_group(student, examples, pad_token_id, device, dtype, scaler):
    """04's control averages response means, unlike 03's pooled-token mean."""
    if not examples:
        raise ValueError("Cannot update on an empty group.")
    total_loss, total_tokens = 0.0, 0
    for example in examples:
        batch = {
            key: value.to(device)
            for key, value in collate_examples([example], pad_token_id).items()
        }
        with precision_context(device, dtype):
            loss_sum, count = response_loss_sum(student, batch)
            response_mean = loss_sum / count
        scaler.scale(response_mean / len(examples)).backward()
        total_loss += float(response_mean.detach())
        total_tokens += count
    return {
        "loss": total_loss / len(examples),
        "response_tokens": total_tokens,
        "examples": len(examples),
        "token_limit_stops": 0,
    }


def optimizer_update(student, optimizer, scheduler, scaler, max_grad_norm):
    import torch

    scaler.unscale_(optimizer)
    # Non-finite gradients stop the run before either optimizer or scheduler moves.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        student.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    learning_rate = optimizer.param_groups[0]["lr"]
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {"grad_norm_before_clipping": float(grad_norm), "learning_rate": learning_rate}


def fixed_trajectory_diagnostics(student, teacher, trajectories, device, dtype, config):
    import torch

    if not trajectories:
        raise ValueError("Fixed-prefix diagnostics require at least one trajectory.")
    was_training = student.training
    student.eval()
    rows = []
    try:
        with torch.no_grad():
            for trajectory in trajectories:
                loss = trajectory_kl(student, teacher, trajectory, device, dtype, config)
                rows.append(
                    {
                        "example_id": trajectory["example_id"],
                        "forward_kl_nats": float(loss),
                        "response_tokens": trajectory["response_length"],
                    }
                )
    finally:
        student.train(was_training)
    return {
        "mean_forward_kl_nats": sum(r["forward_kl_nats"] for r in rows) / len(rows),
        "rows": rows,
    }


def experiment_signature(
    config, protocol, splits, checkpoint_hashes, evaluation_policy, code_hashes=None
):
    import hashlib

    specification = {
        "configuration": asdict(config),
        "protocol": protocol,
        "splits": splits,
        "starting_checkpoint_hashes": checkpoint_hashes,
        "evaluation_policy": evaluation_policy,
        "code_hashes": code_hashes or {},
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
    }
    return hashlib.sha256(
        json.dumps(specification, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_resume(
    directory, student, tokenizer, optimizer, scheduler, scaler, *, arm, step, signature, history
):
    """Save only at an optimizer boundary. Keep the two most recent OWNED snapshots."""
    import torch

    from src.sft import save_checkpoint

    parent = Path(directory) / "resume" / arm
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"step_{step:04d}"
    if destination.exists():
        raise FileExistsError(f"Resume snapshot already exists: {destination}")
    # An interrupted write never becomes an eligible resume boundary. Unique
    # staging names also avoid blocking recovery after a hard kernel disconnect.
    with tempfile.TemporaryDirectory(
        prefix=f".step_{step:04d}-", suffix=".incomplete", dir=parent
    ) as temporary:
        staging = Path(temporary)
        save_checkpoint(student, tokenizer, staging / "student")
        state = {
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        torch.save(state, staging / "training_state.pt")
        write_json(
            staging / "resume.json",
            {
                "arm": arm,
                "completed_steps": step,
                "signature": signature,
                "history": history,
                "optimizer_boundary": True,
            },
        )
        write_inventory(staging)
        staging.rename(destination)
    # Only directories with this run's signature and arm are eligible for pruning.
    owned = []
    for candidate in sorted(parent.glob("step_*")):
        record = candidate / "resume.json"
        if record.is_file():
            metadata = json.loads(record.read_text())
            if metadata.get("signature") == signature and metadata.get("arm") == arm:
                owned.append(candidate)
    for candidate in owned[:-2]:
        shutil.rmtree(candidate)
    return destination


def read_resume(path, *, arm, signature):
    path = Path(path)
    verify_inventory(path)
    metadata = json.loads((path / "resume.json").read_text())
    if (
        metadata.get("arm") != arm
        or metadata.get("signature") != signature
        or not metadata.get("optimizer_boundary")
    ):
        raise ValueError("Resume arm/protocol/starting checkpoint differs from this experiment.")
    step = metadata.get("completed_steps")
    history = metadata.get("history", [])
    if (
        type(step) is not int
        or step < 1
        or [row.get("step") for row in history] != list(range(1, step + 1))
    ):
        raise ValueError("Resume history is inconsistent with its completed optimizer steps.")
    return metadata


def restore_training_state(path, optimizer, scheduler, scaler):
    import torch

    # These are our own checksummed state dictionaries. weights_only avoids
    # loading executable objects from a pickle checkpoint.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs from the saved runtime.")
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def append_trajectories(path, trajectories):
    with Path(path).open("a", encoding="utf-8") as stream:
        for trajectory in trajectories:
            stream.write(json.dumps(trajectory, ensure_ascii=False, allow_nan=False) + "\n")


def truncate_uncommitted_trajectories(path, completed_steps):
    """On resume, discard rollouts from updates newer than the saved boundary."""
    path = Path(path)
    if not path.exists():
        return
    temporary = path.with_suffix(".recovering")
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # An interrupted final write is not a completed trajectory.
            if row["policy_step"] < completed_steps:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_trajectory_log(path, examples, completed_steps, accumulation_steps):
    """Prove the audit log contains exactly one current-policy rollout per prompt."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    expected_examples = examples[: completed_steps * accumulation_steps]
    if len(rows) != len(expected_examples):
        raise ValueError("Rollout audit log is incomplete at the saved optimizer boundary.")
    for index, (row, example) in enumerate(zip(rows, expected_examples)):
        prompt = example["input_ids"][: example["prompt_length"]]
        if (
            row["example_id"] != example["example_id"]
            or row["policy_step"] != index // accumulation_steps
        ):
            raise ValueError("Rollout audit log has a stale policy or wrong question order.")
        if (
            row["prompt_token_ids"] != prompt
            or row["complete_sequence_ids"] != prompt + row["response_token_ids"]
        ):
            raise ValueError("Rollout audit log changed a prompt or sampled response.")
    return len(rows)


def paired_accuracy_difference(first, second, *, seed=2026, repetitions=10000):
    """Paired bootstrap over QUESTIONS, not a multi-training-seed confidence claim."""
    import numpy as np

    left = {r["example_id"]: r["strict_correct"] for r in first["annotations"]}
    right = {r["example_id"]: r["strict_correct"] for r in second["annotations"]}
    if len(left) != len(first["annotations"]) or len(right) != len(second["annotations"]):
        raise ValueError("Paired comparisons cannot contain duplicate question IDs.")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Bootstrap repetitions must be a positive integer.")
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons require exactly the same question IDs.")
    differences = np.array([int(right[key]) - int(left[key]) for key in sorted(left)], dtype=float)
    generator = np.random.default_rng(seed)
    draws = differences[
        generator.integers(0, len(differences), size=(repetitions, len(differences)))
    ].mean(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "difference_second_minus_first": float(differences.mean()),
        "paired_bootstrap_95_interval": [float(low), float(high)],
        "questions": len(differences),
        "bootstrap_seed": seed,
        "bootstrap_repetitions": repetitions,
        "scope": "Question-sampling uncertainty conditional on one fixed training seed.",
    }


def release_cuda_memory():
    """Call after deleting model/optimizer references in the notebook's namespace."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def new_run_directory(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(root) / "results" / f"on_policy_distillation_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def export_report(root, directory, manifest):
    """Atomically archive evidence; safely reuse a completed report on re-export."""
    directory, root = Path(directory), Path(root)
    report = directory / "report"
    if report.exists():
        verify_inventory(report)
        saved = json.loads((report / "manifest.json").read_text())
        if (
            saved.get("signature") != manifest.get("signature")
            or saved["status"] != manifest["status"]
        ):
            raise FileExistsError("A different experiment's report already exists here.")
        # Rerunning the export cell to request weight ZIPs must not overwrite the
        # completed evidence or fail simply because report.zip already exists.
        return Path(shutil.make_archive(str(directory / "report"), "zip", root_dir=report))
    with tempfile.TemporaryDirectory(
        prefix=".report-", suffix=".incomplete", dir=directory
    ) as temporary:
        staging = Path(temporary)
        for pattern in ("*.json", "*.jsonl", "*.png", "*.svg"):
            for source in sorted(directory.glob(pattern)):
                shutil.copyfile(source, staging / source.name)
        sources = staging / "source"
        sources.mkdir()
        for source in sorted((root / "src").glob("*.py")):
            shutil.copyfile(source, sources / source.name)
        for name in (
            "requirements-generation.txt",
            "requirements-sft.txt",
            "requirements-opd.txt",
            "requirements-notebook.txt",
        ):
            shutil.copyfile(root / name, sources / name)
        notebook = json.loads((root / "notebooks/04_on_policy_distillation.ipynb").read_text())
        write_json(
            sources / "notebook04_source.json",
            [
                {"cell_type": cell["cell_type"], "source": cell["source"]}
                for cell in notebook["cells"]
            ],
        )
        write_json(
            staging / "manifest.json",
            {
                **manifest,
                "checkpoint_weights_in_report": False,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        write_inventory(staging)
        verify_inventory(staging)
        staging.rename(report)
    return Path(shutil.make_archive(str(directory / "report"), "zip", root_dir=report))
