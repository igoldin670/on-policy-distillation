"""One reproducible SFT and GKD workflow for notebooks and command-line QA.

Inspection loads pinned tokenizers and data. Pretrained models load only when a
training/evaluation function is called explicitly inside an allocated GPU kernel.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from src import opd, sft
from src.data import prepare_opd_splits
from src.generation import sample_trajectory
from src.logprobs import teacher_prompt_token_ids
from src.opd_experiment import SlurmBudget, evaluate_incrementally, isolated_evaluation_rng
from src.opd_runner import run_training_arm
from src.profiles import (
    load_profile,
    load_profile_student,
    load_profile_tokenizer,
    profile_protocol,
)
from src.qwen38.model_upgrade import inspect_upgrade, load_upgrade_models
from src.training_safety import TrainingSafetyConfig
from src.utils import sha256_file, verify_inventory, write_json


def source_hashes(root):
    root = Path(root)
    paths = list((root / "src").rglob("*.py")) + list((root / "configs").glob("*.yaml"))
    paths += list(root.glob("requirements*.txt")) + list(root.glob("constraints*.txt"))
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}
    for name in ("03_sft_cold_start", "04_on_policy_distillation"):
        path = root / "notebooks" / f"{name}.ipynb"
        # Profile, mode, run paths, and pilot/full selection are execution
        # settings. Their resolved scientific inputs are signed separately.
        source = [
            c["source"]
            for c in json.loads(path.read_text())["cells"]
            if c["cell_type"] == "code" and c["id"] not in {"sft-setup", "opd-setup-runtime"}
        ]
        hashes[str(path.relative_to(root))] = hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest()
    return hashes


def new_directory(root, prefix):
    path = (
        Path(root)
        / "results"
        / (prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def record_runtime(directory, device):
    import importlib.metadata
    import platform

    import torch

    software = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "tokenizers", "accelerate", "datasets")
    }
    autocast_dtype = (
        ("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
        if device.type == "cuda"
        else None
    )
    path = Path(directory) / "runtime.json"
    if path.exists() and json.loads(path.read_text())["software"] != software:
        raise ValueError(
            "Resume software differs. Activate the original environment or start a fresh experiment."
        )
    if path.exists() and json.loads(path.read_text()).get("autocast_dtype") != autocast_dtype:
        raise ValueError("Resume precision differs from the saved runtime.")
    record = {
        "software": software,
        "autocast_dtype": autocast_dtype,
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "kind": "allocated_cuda_experiment"
        if device.type == "cuda"
        else "tiny_random_cpu_test_only",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if device.type == "cuda"
        else [],
    }
    write_json(path, record)
    with (Path(directory) / "runtime_sessions.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}) + "\n"
        )


def prepare_cold_start(root, profile_name="qwen38", config=None):
    """Freeze real training/validation data without loading student weights."""
    from datasets import disable_progress_bars, load_dataset

    disable_progress_bars()
    root = Path(root)
    profile = load_profile(root, profile_name)
    protocol = profile_protocol(root, profile)
    if profile_name == "qwen38":
        local = os.environ.get("OPD_QWEN38_LOCAL_SOURCES")
        tokenizers, configs, compatibility = inspect_upgrade(
            profile, sources=Path(local) if local else None
        )
        tokenizer = tokenizers["student"]
    else:
        from transformers import AutoTokenizer

        from src.logprobs import verify_shared_tokenizer

        tokenizer = load_profile_tokenizer(profile)
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            profile["teacher_model_id"], revision=profile["teacher_revision"], token=False
        )
        compatibility = verify_shared_tokenizer(
            tokenizer,
            teacher_tokenizer,
            ["  3 + 2 = 5.\n#### 5", "Café 中文 🙂"],
            profile["output_vocab_size"],
        )
        tokenizers, configs = {"student": tokenizer, "teacher": teacher_tokenizer}, {}
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = config or sft.SFTConfig()
    dataset = load_dataset(
        protocol["dataset_id"],
        protocol["dataset_config"],
        split="train",
        revision=protocol["dataset_revision"],
        token=False,
    )
    splits = sft.prepare_splits(dataset, tokenizer, protocol, config)
    return dict(
        root=root,
        profile=profile,
        protocol=protocol,
        tokenizer=tokenizer,
        tokenizers=tokenizers,
        model_configs=configs,
        compatibility=compatibility,
        sft_config=config,
        cold_splits=splits,
        source_train=dataset,
    )


def _runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select the allocated CUDA Jupyter kernel. Inspect mode needs no GPU.")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return device, dtype


def evaluate_model(model, state, examples, directory, name, identity, device, dtype, budget=None):
    return evaluate_incrementally(
        model,
        state["tokenizer"],
        examples,
        state["protocol"]["generation_config"],
        device,
        dtype,
        path=Path(directory) / f"{name}.json",
        identity=identity,
        budget=budget,
    )


def profiled_sample(state, student, example, *, directory=None, **settings):
    trajectory = sample_trajectory(student, state["tokenizer"], example, **settings)
    if any(
        token < 0 or token >= state["profile"]["tokenizer_vocab_size"]
        for token in trajectory["response_token_ids"]
    ):
        if directory is not None:
            write_json(
                Path(directory)
                / (
                    "invalid_trajectory_"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    + ".json"
                ),
                trajectory,
            )
        raise ValueError(
            "Student sampled an unmapped output-head row. Preserve the failure for review."
        )
    trajectory["sampling_source"] = "student"
    return trajectory


def make_context(
    state,
    directory,
    config,
    signature,
    starting_checkpoint,
    baseline,
    baseline_nll,
    device,
    dtype,
    diagnostics=(),
    safety=None,
):
    """Explicit dependencies for the shared training entry point."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def evaluate(model, examples, name, *, directory=directory, identity=signature, budget=None):
        return evaluate_model(
            model, state, examples, directory, name, identity, device, dtype, budget
        )

    names = (
        "append_trajectories",
        "backward_opd_group",
        "backward_sft_group",
        "export_report",
        "fixed_trajectory_diagnostics",
        "optimizer_update",
        "read_resume",
        "release_cuda_memory",
        "restore_training_state",
        "seed_everything",
        "training_health",
        "truncate_uncommitted_trajectories",
        "validate_trajectory_log",
    )
    context = {name: getattr(opd, name) for name in names}
    context.update(
        {
            name: getattr(sft, name)
            for name in (
                "evaluation_summary",
                "save_checkpoint",
                "validation_loss",
                "verify_checkpoint",
            )
        }
    )
    context.update(
        PROJECT_ROOT=state["root"],
        config=config,
        run_directory=directory,
        signature=signature,
        starting_checkpoint_directory=Path(starting_checkpoint),
        starting_validation_loss=baseline_nll,
        starting_validation=baseline,
        tokenizer=state["tokenizer"],
        device=device,
        compute_dtype=dtype,
        train_examples=state["splits"]["train"],
        validation_examples=state["splits"]["validation"],
        diagnostic_trajectories=list(diagnostics),
        stop_token_ids=state["profile"]["eos_token_ids"],
        evaluation_policy=state["protocol"]["generation_config"],
        sample_trajectory=lambda student, tokenizer, example, **settings: profiled_sample(
            state, student, example, directory=directory, **settings
        ),
        evaluate_and_save=evaluate,
        verify_inventory=verify_inventory,
        write_json=write_json,
        safety_policy=safety,
        load_student=lambda path, target: load_profile_student(state["profile"], target, path),
    )
    return context


def _specification(state, directory, config, checkpoint_hashes, *, kind):
    code = source_hashes(state["root"])
    signature = opd.experiment_signature(
        config,
        state["protocol"],
        state["splits"],
        checkpoint_hashes,
        state["protocol"]["generation_config"],
        code,
        model_profile=state["profile"],
    )
    payload = dict(
        signature=signature,
        kind=kind,
        model_profile=state["profile"],
        configuration=asdict(config),
        protocol=state["protocol"],
        code_hashes=code,
        starting_checkpoint_hashes=checkpoint_hashes,
        safety_policy=asdict(TrainingSafetyConfig()) if kind != "reference_solution_sft" else None,
    )
    path = Path(directory) / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload)):
        raise ValueError("Experiment inputs or code changed. Use a fresh run directory.")
    write_json(path, payload)
    write_json(Path(directory) / "data_splits.json", state["splits"])
    return signature


def run_cold_start(state, run_directory=None):
    """A resumable one-epoch SFT experiment with deferred held-out evaluation."""
    import torch

    device, dtype = _runtime()
    directory = (
        Path(run_directory)
        if run_directory
        else new_directory(state["root"], "sft_cold_start_" + state["profile"]["profile"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(state, splits=state["cold_splits"])
    record_runtime(directory, device)
    sft_config = state["sft_config"]
    if sft_config.epochs != 1 or sft_config.microbatch_size != 1:
        raise ValueError("This fixed study uses one epoch and microbatch size one.")
    config = opd.OPDConfig(
        train_examples=len(state["splits"]["train"]),
        accumulation_steps=sft_config.accumulation_steps,
        learning_rate=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        warmup_fraction=sft_config.warmup_fraction,
        max_grad_norm=sft_config.max_grad_norm,
        checkpoint_interval=8,
        validation_interval=8,
        eos_token_ids=tuple(state["profile"]["eos_token_ids"]),
    )
    signature = _specification(state, directory, config, {}, kind="reference_solution_sft")
    budget = SlurmBudget.from_environment()
    budget.require_time()
    initial = directory / "initial"
    # Preserve initial weights once so a timeout can recover the same cold start.
    if not initial.exists():
        student = load_profile_student(state["profile"], device)
        sft.save_checkpoint(student, state["tokenizer"], initial)
        del student
        opd.release_cuda_memory()
    student = load_profile_student(state["profile"], device, initial)
    with isolated_evaluation_rng():
        before = evaluate_model(
            student,
            state,
            state["splits"]["validation"],
            directory,
            "before_validation",
            signature,
            device,
            dtype,
            budget,
        )
        before_nll = sft.validation_loss(
            student, state["splits"]["validation"], state["tokenizer"].pad_token_id, device, dtype
        )
    del student
    opd.release_cuda_memory()
    context = make_context(
        state, directory, config, signature, initial, before, before_nll, device, dtype
    )
    result = run_training_arm(context, "sft_control", recover_pause_validation=True)
    if result["status"] != "completed":
        return dict(directory=str(directory), result=result, report=None)
    checkpoint = directory / "checkpoints/sft_control"
    after = json.loads(
        (
            directory / f"sft_control_boundary_{config.optimizer_steps:04d}_validation.json"
        ).read_text()
    )
    if not (directory / "report").exists():
        runtime = dict(
            kind="real_student_cuda_training"
            if device.type == "cuda"
            else "tiny_random_cpu_test_only",
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            optimizer_steps=result["completed_steps"],
            before_validation_loss=before_nll,
            after_validation_loss=result["final_validation_reference_nll"],
        )
        sft.save_report(
            state["root"],
            directory,
            sft_config,
            state["protocol"],
            state["splits"],
            result["history"],
            {"before_validation": before, "after_validation": after},
            runtime,
            result["checkpoint_reload_verified"],
            checkpoint_directory=checkpoint,
        )
    sft.verify_report(directory / "report")
    return dict(
        directory=str(directory),
        checkpoint=str(checkpoint),
        result=result,
        report=str(directory / "report"),
    )


def resolve_profile_cold_start(root, profile, run_directory=None):
    candidates = (
        [Path(run_directory)]
        if run_directory
        else sorted((Path(root) / "results").glob("sft_cold_start_*"), reverse=True)
    )
    for directory in candidates:
        path = directory / "report/manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        if (
            manifest.get("student_model_id") != profile["student_model_id"]
            or manifest.get("student_revision") != profile["student_revision"]
        ):
            if run_directory:
                raise ValueError("Selected SFT run belongs to a different model profile.")
            continue
        report, evidence = opd.load_sft_report(
            directory / "report", Path(root) / ".cache/opd-inputs"
        )
        checkpoint = (report / evidence["manifest"]["checkpoint_directory"]).resolve()
        verify_inventory(checkpoint, evidence["checkpoint_hashes"])
        config = json.loads((checkpoint / "config.json").read_text())
        if (
            config.get("model_type") != profile["model_type"]
            or config.get("vocab_size") != profile["output_vocab_size"]
        ):
            raise ValueError("SFT checkpoint does not match the selected model profile.")
        return report, checkpoint, evidence
    raise FileNotFoundError(
        "Run Notebook 03 for this profile first, then set SFT_RUN_DIRECTORY to its completed run folder."
    )


def prepare_distillation(state):
    from datasets import load_dataset

    protocol = state["protocol"]
    test = load_dataset(
        protocol["dataset_id"],
        protocol["dataset_config"],
        split="test",
        revision=protocol["dataset_revision"],
        token=False,
    )
    splits = prepare_opd_splits(
        state["source_train"],
        test,
        state["tokenizer"],
        protocol,
        copy.deepcopy(state["cold_splits"]),
        max_response_length=256,
    )
    for example in splits["train"] + splits["validation"] + splits["primary_test"]:
        example["teacher_prompt_token_ids"] = teacher_prompt_token_ids(
            state["tokenizers"]["teacher"], example["prompt"]
        )
        if (
            example in splits["train"]
            and len(example["teacher_prompt_token_ids"]) + example["target_length"] > 1024
        ):
            raise ValueError("Teacher reference context exceeds the fixed sequence budget.")
    # Validate teacher generation formatting during CPU inspection as well.
    teacher_evaluation_examples(state, splits["validation"] + splits["primary_test"])
    return dict(state, splits=splits)


def _load_teacher(state):
    if state["profile"]["profile"] == "legacy":
        return opd.load_teacher("cuda:0")
    _, teacher = load_upgrade_models(state["profile"], state["model_configs"], load_student=False)
    return teacher


def teacher_evaluation_examples(state, examples):
    """Prove that generation reads the exact pinned teacher prefix IDs."""
    tokenizer = state.get("tokenizers", {}).get("teacher", state["tokenizer"])
    prepared = []
    for example in examples:
        ids = example["teacher_prompt_token_ids"]
        prompt = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if state["tokenizer"].encode(prompt, add_special_tokens=False) != ids:
            raise ValueError("Teacher evaluation prefix does not round-trip to its exact IDs.")
        prepared.append(dict(example, prompt=prompt))
    return prepared


def _pilot_passes(result, baseline_nll, baseline_accuracy):
    history = result.get("history", [])
    if result.get("status") != "pilot_complete" or not history:
        return False
    health = opd.training_health(
        history,
        opd.OPDConfig(),
        baseline_nll,
        safety_policy=TrainingSafetyConfig(),
        baseline_accuracy=baseline_accuracy,
    )
    return result.get("validation_status") == "completed" and not health["pause"]


def _run_distillation(
    state, *, sft_run_directory=None, run_directory=None, stage="pilot", config=None
):
    """Two predeclared pilots, then matched supervised/on-policy KD and final testing."""
    if stage not in {"pilot", "full"}:
        raise ValueError("Choose pilot or full.")
    if stage == "full" and run_directory is None:
        raise ValueError("Set RESUME_RUN_DIRECTORY to the reviewed pilot folder.")
    if stage == "full":
        selection_path = Path(run_directory) / "selection.json"
        if (
            not selection_path.exists()
            or json.loads(selection_path.read_text()).get("status") != "qualified_pilot"
        ):
            raise ValueError("Full training requires a qualified pilot selection.")
    device, dtype = _runtime()
    if device.type == "cuda" and state["profile"]["profile"] == "qwen38":
        from src.qwen38.qualification import require_qualification

        print("GPU qualification", require_qualification(state["root"], state["profile"]))
    _, checkpoint, evidence = resolve_profile_cold_start(
        state["root"], state["profile"], sft_run_directory
    )
    if evidence["splits"] != state["cold_splits"]:
        raise ValueError("Prepared cold-start splits differ from the verified SFT report.")
    directory = (
        Path(run_directory)
        if run_directory
        else new_directory(state["root"], "gkd_" + state["profile"]["profile"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    record_runtime(directory, device)
    config = config or opd.OPDConfig(
        eos_loss_mode="semantic_stop",
        eos_token_ids=tuple(state["profile"]["eos_token_ids"]),
        teacher_context="qwen3_5_nonthinking_v1"
        if state["profile"]["profile"] == "qwen38"
        else "qwen3_nonthinking_v1",
        validation_interval=8,
        checkpoint_interval=8,
    )
    parent_signature = _specification(
        state, directory, config, evidence["checkpoint_hashes"], kind="controlled_gkd"
    )
    # Reserve enough for two retained snapshots plus an accepted one per arm,
    # an atomic write, pilot alternatives and final student checkpoints.
    parameter_bytes = sum(p.stat().st_size for p in checkpoint.glob("*.safetensors"))
    storage_required = 32 * parameter_bytes + 4 * 2**30
    if shutil.disk_usage(directory).free < storage_required:
        raise RuntimeError(
            f"Recovery storage needs approximately {storage_required / 2**30:.1f} GiB free. Check personal quota too."
        )
    budget = SlurmBudget.from_environment()
    budget.require_time()
    student = load_profile_student(state["profile"], device, checkpoint)
    with isolated_evaluation_rng():
        baseline = evaluate_model(
            student,
            state,
            state["splits"]["validation"],
            directory,
            "starting_sft_validation",
            parent_signature,
            device,
            dtype,
            budget,
        )
        baseline_nll = sft.validation_loss(
            student, state["splits"]["validation"], state["tokenizer"].pad_token_id, device, dtype
        )
        opd.seed_everything(42)
        diagnostics = []
        for ex in state["splits"]["validation"][: config.diagnostic_examples]:
            budget.require_time()
            diagnostics.append(
                profiled_sample(
                    state,
                    student,
                    ex,
                    directory=directory,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=config.max_response_length,
                    stop_token_ids=state["profile"]["eos_token_ids"],
                    policy_step=0,
                )
            )
    del student
    opd.release_cuda_memory()
    write_json(
        directory / "starting_sft_metrics.json",
        {"validation": baseline["scores"], "reference_nll": baseline_nll},
    )
    write_json(directory / "fixed_validation_trajectories.json", diagnostics)
    teacher = _load_teacher(state)
    try:
        # The teacher is scored under its non-thinking chat prefix, consistently
        # with KD conditioning. Reference answers never enter generation inputs.
        teacher_examples = teacher_evaluation_examples(state, state["splits"]["validation"])
        teacher_validation = evaluate_model(
            teacher,
            state,
            teacher_examples,
            directory,
            "teacher_validation",
            parent_signature,
            device,
            dtype,
            budget,
        )
        write_json(
            directory / "teacher_readiness.json",
            {
                "student_strict_accuracy": baseline["scores"]["strict"]["accuracy"],
                "teacher_strict_accuracy": teacher_validation["scores"]["strict"]["accuracy"],
                "teacher_exceeds_student": teacher_validation["scores"]["strict"]["accuracy"]
                > baseline["scores"]["strict"]["accuracy"],
            },
        )
        pilots = (
            json.loads((directory / "pilots.json").read_text())
            if (directory / "pilots.json").exists()
            else {}
        )
        rates_to_run = (
            () if stage == "full" and (directory / "selection.json").exists() else (1e-6, 5e-6)
        )
        for rate in rates_to_run:
            budget.require_time()
            recipe = replace(config, learning_rate=rate)
            child = directory / f"pilot_lr_{rate:g}"
            child.mkdir(exist_ok=True)
            signature = _specification(
                state, child, recipe, evidence["checkpoint_hashes"], kind="on_policy_gkd"
            )
            context = make_context(
                state,
                child,
                recipe,
                signature,
                checkpoint,
                baseline,
                baseline_nll,
                device,
                dtype,
                diagnostics,
                TrainingSafetyConfig(),
            )
            result = run_training_arm(
                context,
                "on_policy",
                teacher,
                stop_after_steps=min(config.pilot_updates, config.optimizer_steps),
                recover_pause_validation=True,
            )
            pilots[str(rate)] = result
            write_json(directory / "pilots.json", pilots)
            if result["status"] == "walltime_paused":
                return dict(directory=str(directory), status="walltime_paused", pilots=pilots)
        accepted = [
            (float(rate), result)
            for rate, result in pilots.items()
            if _pilot_passes(result, baseline_nll, baseline["scores"]["strict"]["accuracy"])
        ]
        if stage == "full" and (directory / "selection.json").exists():
            selection = json.loads((directory / "selection.json").read_text())
            if selection["selected_learning_rate"] is None:
                return dict(directory=str(directory), **selection)
            rate = selection["selected_learning_rate"]
        elif not accepted:
            selection = {"status": "review_required", "selected_learning_rate": None}
        else:
            rate, _ = min(
                accepted,
                key=lambda item: (
                    -item[1]["final_validation"]["strict_accuracy"],
                    item[1]["final_validation_reference_nll"],
                    item[0],
                ),
            )
            selection = {"status": "qualified_pilot", "selected_learning_rate": rate}
        write_json(directory / "selection.json", selection)
        if stage == "pilot" or selection["selected_learning_rate"] is None:
            opd.export_report(
                state["root"],
                directory,
                {**selection, "signature": parent_signature},
                report_name=f"report_pilots_{len(pilots)}",
            )
            return dict(directory=str(directory), pilots=pilots, **selection)
        recipe = replace(config, learning_rate=rate)
        results = {}
        for arm, child in (
            ("on_policy", directory / f"pilot_lr_{rate:g}"),
            ("supervised_kd", directory / "supervised_kd"),
        ):
            child.mkdir(exist_ok=True)
            signature = _specification(
                state,
                child,
                recipe,
                evidence["checkpoint_hashes"],
                kind="on_policy_gkd" if arm == "on_policy" else "supervised_kd_control",
            )
            context = make_context(
                state,
                child,
                recipe,
                signature,
                checkpoint,
                baseline,
                baseline_nll,
                device,
                dtype,
                diagnostics,
                TrainingSafetyConfig(),
            )
            results[arm] = run_training_arm(context, arm, teacher, recover_pause_validation=True)
            write_json(directory / "training_results.json", results)
            if results[arm]["status"] != "completed":
                return dict(
                    directory=str(directory), status=results[arm]["status"], results=results
                )
        evaluations = {}
        cases = [("original_student", None), ("starting_sft", checkpoint)] + [
            (
                arm,
                Path(result["resume_snapshot"]) / "student"
                if result.get("resume_snapshot")
                else (
                    directory
                    / (f"pilot_lr_{rate:g}" if arm == "on_policy" else arm)
                    / "checkpoints"
                    / arm
                ),
            )
            for arm, result in results.items()
        ]
        for name, location in cases:
            budget.require_time()
            student = load_profile_student(state["profile"], device, location)
            evaluations[name] = evaluate_model(
                student,
                state,
                state["splits"]["primary_test"],
                directory,
                "primary_" + name,
                parent_signature + name,
                device,
                dtype,
                budget,
            )
            del student
            opd.release_cuda_memory()
        teacher_test = teacher_evaluation_examples(state, state["splits"]["primary_test"])
        evaluations["teacher"] = evaluate_model(
            teacher,
            state,
            teacher_test,
            directory,
            "primary_teacher",
            parent_signature,
            device,
            dtype,
            budget,
        )
        comparisons = {
            name: opd.paired_accuracy_difference(evaluations["starting_sft"], evaluations[name])
            for name in ("supervised_kd", "on_policy")
        }
        write_json(directory / "paired_comparisons.json", comparisons)
        final = dict(
            signature=parent_signature,
            status="completed",
            selected_learning_rate=rate,
            metrics={name: evaluation["scores"] for name, evaluation in evaluations.items()},
            paired_comparisons=comparisons,
        )
        write_json(directory / "final_results.json", final)
        opd.export_report(state["root"], directory, final)
        return dict(directory=str(directory), **final)
    finally:
        del teacher
        opd.release_cuda_memory()


def run_distillation(
    state, *, sft_run_directory=None, run_directory=None, stage="pilot", config=None
):
    """Publish parent status and small evidence even when an allocation ends."""
    from src.opd_experiment import WalltimePause

    if stage == "full" and run_directory is None:
        raise ValueError("Set RESUME_RUN_DIRECTORY to the reviewed pilot folder.")
    directory = (
        Path(run_directory)
        if run_directory
        else new_directory(state["root"], "gkd_" + state["profile"]["profile"])
    )
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = _run_distillation(
            state,
            sft_run_directory=sft_run_directory,
            run_directory=directory,
            stage=stage,
            config=config,
        )
    except WalltimePause:
        result = {"directory": str(directory), "status": "walltime_paused"}
    except (KeyboardInterrupt, RuntimeError, ValueError, OSError) as error:
        write_json(
            directory / "stage_failure.json",
            {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    write_json(directory / "stage_status.json", result)
    from src.opd_plots import plot_pipeline_run

    plot_pipeline_run(directory)
    specification = (
        json.loads((directory / "experiment.json").read_text())
        if (directory / "experiment.json").exists()
        else {}
    )
    try:
        opd.export_report(
            state["root"],
            directory,
            {"signature": specification.get("signature"), **result},
            report_name="report_stage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        )
    except OSError as error:
        print(
            "Report packaging failed. Saved run files remain at", directory, str(error), flush=True
        )
    return result
