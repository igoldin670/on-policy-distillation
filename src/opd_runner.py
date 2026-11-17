"""Shared notebook/CLI training loop with explicit dependencies and durable boundaries.

The context mapping supplies datasets, model loading, evaluation and tensor helpers.
No kernel globals are read here. Callers can provide tiny CPU models for the same
entry point used by the real experiment.
"""

import json
import math
import time
from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup


def run_training_arm(
    context,
    arm,
    teacher=None,
    *,
    stop_after_steps=None,
    run_config=None,
    output_directory=None,
    run_signature=None,
    recover_pause_validation=False,
):
    from dataclasses import asdict
    from datetime import datetime, timezone

    from transformers import GenerationConfig

    from src.opd_experiment import (
        SlurmBudget,
        commit_boundary,
        isolated_evaluation_rng,
        publish_pause,
    )
    from src.opd_experiment import (
        WalltimePause as TrainingWalltimePause,
    )

    # Explicit overrides keep comparison arms independent. Existing pilot/full
    # calls use the notebook's selected recipe and run directory.
    config = run_config if run_config is not None else context["config"]
    run_directory = (
        Path(output_directory) if output_directory is not None else context["run_directory"]
    )
    signature = run_signature if run_signature is not None else context["signature"]
    from src.training_safety import (
        accept_checkpoint,
        capture_update_rng,
        pin_starting_checkpoint,
        record_training_failure,
        restore_boundary_measurements,
        restore_update_rng,
        save_boundary_measurements,
    )

    # The new policy qualifies KD pilots. The fixed supervised control keeps
    # its existing recipe and cannot overwrite a KD accepted-checkpoint pointer.
    safety_policy = context.get("safety_policy") if arm in {"on_policy", "supervised_kd"} else None
    baseline_accuracy = (
        context.get("starting_validation", {}).get("scores", {}).get("strict", {}).get("accuracy")
    )

    def assess_health():
        if safety_policy is not None or arm == "on_policy":
            return context["training_health"](
                history,
                config,
                context["starting_validation_loss"],
                safety_policy=safety_policy,
                baseline_accuracy=baseline_accuracy,
            )
        return {"pause": False, "warnings": []}

    def pause_kind(assessment):
        if any(r["metric"] != "recent_cap_fraction" for r in assessment.get("reasons", [])):
            return "quality_paused"
        return "health_paused"

    budget = SlurmBudget.from_environment()
    assert arm in {"sft_control", "on_policy", "supervised_kd"}
    if arm in {"on_policy", "supervised_kd"} and teacher is None:
        raise ValueError("On-policy KD needs the frozen teacher.")
    target_steps = config.optimizer_steps if stop_after_steps is None else stop_after_steps
    if type(target_steps) is not int or not 1 <= target_steps <= config.optimizer_steps:
        raise ValueError("The requested stage must end within the fixed experiment.")
    final_directory = run_directory / "checkpoints" / arm
    result_path = run_directory / f"{arm}_training.json"
    progress_path = run_directory / "progress.jsonl"

    def record_progress(update, stage, event="started", **details):
        # Close each entry before starting the next operation so it can be
        # read while the run is active or after the kernel is interrupted.
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "update": update,
            "stage": stage,
            "event": event,
            **details,
        }
        with progress_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")

    if final_directory.exists() and result_path.exists():
        context["verify_inventory"](final_directory)
        completed = json.loads(result_path.read_text())
        assert completed["completed_steps"] == config.optimizer_steps
        assert completed.get("signature", signature) == signature
        print(f"{arm}: reusing its verified completed checkpoint.")
        return completed

    snapshots = sorted((run_directory / "resume" / arm).glob("step_*"))
    resume_path = snapshots[-1] if snapshots else None
    resume_metadata = (
        context["read_resume"](resume_path, arm=arm, signature=signature) if resume_path else None
    )
    history = resume_metadata["history"] if resume_metadata else []
    completed_steps = len(history)
    if safety_policy is not None:
        if resume_metadata and resume_metadata.get("training_safety_policy") != asdict(
            safety_policy
        ):
            raise ValueError("Resume safety settings differ. Start a new experiment.")
        pin_starting_checkpoint(
            run_directory, context["starting_checkpoint_directory"], signature, safety_policy
        )
        if resume_path is not None:
            restore_boundary_measurements(
                run_directory, history, resume_path, signature, safety_policy
            )
    if completed_steps > target_steps:
        raise ValueError("This snapshot is beyond the pilot boundary. Choose the full stage.")
    health = assess_health()
    if health["pause"] and not recover_pause_validation:
        raise RuntimeError(
            "This snapshot was health-paused. Review the evidence and start a revised experiment."
        )
    model_source = (
        resume_path / "student" if resume_path else context["starting_checkpoint_directory"]
    )
    student = context["load_student"](model_source, context["device"])
    student.generation_config = GenerationConfig.from_dict(context["evaluation_policy"])
    student.generation_config.use_cache = True
    # These fields travel with every newly saved Hugging Face checkpoint.
    student.config.opd_eos_loss_mode = config.eos_loss_mode
    student.config.opd_eos_token_ids = list(config.eos_token_ids)

    def measure_boundary(student, record, name, *, generate_answers=True):
        # A checksummed optimizer snapshot precedes this measurement. Metrics
        # live in an atomic sidecar because the snapshot itself stays immutable.
        boundary_snapshot = run_directory / "resume" / arm / f"step_{record['step']:04d}"
        if safety_policy is not None:
            restore_boundary_measurements(
                run_directory, history, boundary_snapshot, signature, safety_policy
            )
            if "generated_validation" in record:
                return context["evaluate_and_save"](
                    student,
                    context["validation_examples"],
                    name,
                    directory=run_directory,
                    identity=signature,
                    budget=budget,
                )
        # Greedy validation and diagnostic forwards preserve rollout RNG states.
        with isolated_evaluation_rng():
            budget.require_time()
            record_progress(record["step"], "validation")
            record["validation_reference_nll"] = context["validation_loss"](
                student,
                context["validation_examples"],
                context["tokenizer"].pad_token_id,
                context["device"],
                context["compute_dtype"],
            )
            if safety_policy is not None:
                save_boundary_measurements(
                    run_directory,
                    record,
                    boundary_snapshot,
                    signature,
                    safety_policy,
                    complete=False,
                )
                assessment = assess_health()
                if assessment["pause"]:
                    publish_pause(
                        run_directory,
                        arm=arm,
                        status=pause_kind(assessment),
                        step=len(history),
                        snapshot=boundary_snapshot,
                        signature=signature,
                        config=config,
                        health=assessment,
                        reason="validation quality threshold",
                    )
            if teacher is not None:
                budget.require_time()
                diagnostics = context["fixed_trajectory_diagnostics"](
                    student,
                    teacher,
                    context["diagnostic_trajectories"],
                    context["device"],
                    context["compute_dtype"],
                    config,
                )
                record["fixed_validation_kl"] = diagnostics["mean_forward_kl_nats"]
                record["fixed_raw_token_kl"] = diagnostics["mean_raw_token_kl_nats"]
                record["fixed_semantic_stop_kl"] = diagnostics["mean_semantic_stop_kl_nats"]
            if not generate_answers:
                record_progress(record["step"], "validation", "completed")
                return None
            measured = context["evaluate_and_save"](
                student,
                context["validation_examples"],
                name,
                directory=run_directory,
                identity=signature,
                budget=budget,
            )
        record["generated_validation"] = context["evaluation_summary"](measured, arm, "validation")
        if safety_policy is not None:
            save_boundary_measurements(
                run_directory, record, boundary_snapshot, signature, safety_policy, complete=True
            )
            if not assess_health()["pause"]:
                accept_checkpoint(
                    run_directory, boundary_snapshot, record, signature, safety_policy
                )
        record_progress(record["step"], "validation", "completed")
        return measured

    def finish_pause(student, status, reason, snapshot):
        # Publish status BEFORE validation. A timeout can leave validation pending
        # while the optimizer boundary and reason remain inspectable on disk.
        health = assess_health()
        result = publish_pause(
            run_directory,
            arm=arm,
            status=status,
            step=len(history),
            snapshot=snapshot,
            signature=signature,
            config=config,
            health=health,
            reason=reason,
        )
        result.update(history=history, resumed_from_step=completed_steps)
        if history:
            try:
                measured = measure_boundary(
                    student, history[-1], f"{arm}_boundary_{len(history):04d}_validation"
                )
                result.update(
                    validation_status="completed",
                    final_validation=context["evaluation_summary"](measured, status, "validation"),
                    final_validation_reference_nll=history[-1]["validation_reference_nll"],
                )
            except TrainingWalltimePause:
                print(
                    "Checkpoint saved. Validation will resume in the next allocation.", flush=True
                )
        result["health"] = assess_health()
        if safety_policy is not None and result["health"]["pause"]:
            status = pause_kind(result["health"])
            result["status"] = status
            result["pause_reason"] = "training or validation safety threshold"
        context["write_json"](run_directory / f"{arm}_history.json", history)
        context["write_json"](run_directory / f"{arm}_pause.json", result)
        context["write_json"](run_directory / f"{arm}_{status}.json", result)
        context["write_json"](run_directory / "manifest.json", result)
        # Publish the small report here so a disconnected notebook client does
        # not have to execute the later export cell to retain this boundary.
        if safety_policy is not None:
            try:
                context["export_report"](
                    context["PROJECT_ROOT"],
                    run_directory,
                    result,
                    report_name=f"report_{status}_{len(history):04d}_{time.time_ns()}",
                )
            except OSError as error:
                # The checkpoint and pause manifest already exist. A full disk
                # or transfer error must remain visible without hiding that state.
                print(
                    f"Report packaging failed. Checkpoint retained at {snapshot}. {error}",
                    flush=True,
                )
        return result

    trajectory_path = run_directory / "training_trajectories.jsonl"
    if arm == "on_policy":
        context["truncate_uncommitted_trajectories"](trajectory_path, completed_steps)
        context["validate_trajectory_log"](
            trajectory_path, context["train_examples"], completed_steps, config.accumulation_steps
        )
    elif arm == "supervised_kd":
        context["truncate_uncommitted_trajectories"](
            run_directory / "reference_trajectories.jsonl", completed_steps
        )

    # A comparison may finish interrupted evaluation of an already stopped arm.
    # It never restarts updates from a health-paused policy.
    if recover_pause_validation and (
        health["pause"]
        or (completed_steps == target_steps and target_steps < config.optimizer_steps)
    ):
        try:
            return finish_pause(
                student,
                pause_kind(health) if health["pause"] else "pilot_complete",
                "health guard" if health["pause"] else "pilot boundary",
                resume_path,
            )
        finally:
            del student
            context["release_cuda_memory"]()

    context["seed_everything"](config.seed)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(config.optimizer_steps * config.warmup_fraction),
        num_training_steps=config.optimizer_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.device(context["device"]).type == "cuda"
        and context["compute_dtype"] == torch.float16,
    )
    if resume_path:
        context["restore_training_state"](resume_path, optimizer, scheduler, scaler)
    if teacher is not None:
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert not optimizer_ids.intersection(id(p) for p in teacher.parameters())
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    optimizer.zero_grad(set_to_none=True)
    if torch.device(context["device"]).type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
        torch.cuda.synchronize()
    started = time.perf_counter()
    pause_status, pause_reason = None, None
    latest_snapshot = resume_path
    update_pending = False
    update_rng = None

    def save_boundary(student, optimizer, scheduler, scaler, *, status=None, reason=None):
        # At step zero, the verified 03 checkpoint already supplies the start state.
        snapshot = None
        if history:
            record_progress(len(history), "checkpoint")
            snapshot = commit_boundary(
                run_directory,
                student,
                context["tokenizer"],
                optimizer,
                scheduler,
                scaler,
                arm=arm,
                step=len(history),
                signature=signature,
                history=history,
                config=config,
                training_safety_policy=safety_policy,
            )
        if status is not None:
            # Publish immediately after the immutable checkpoint becomes visible.
            # Cleanup or later validation can then be interrupted safely.
            publish_pause(
                run_directory,
                arm=arm,
                status=status,
                step=len(history),
                snapshot=snapshot,
                signature=signature,
                config=config,
                health=health,
                reason=reason,
            )
        if history:
            record_progress(len(history), "checkpoint", "completed")
        return snapshot

    try:
        # A boundary snapshot can precede its validation. Finish that work before
        # generating the next group, using cached answers from interrupted evaluation.
        if history and "generated_validation" not in history[-1]:
            measure_boundary(student, history[-1], f"{arm}_boundary_{len(history):04d}_validation")
        for update_index in range(completed_steps + 1, target_steps + 1):
            # Re-evaluate recovered validation BEFORE allowing another update.
            health = assess_health()
            if health["pause"]:
                pause_status, pause_reason = (
                    pause_kind(health),
                    "saved validation or stopping threshold",
                )
                latest_snapshot = save_boundary(
                    student, optimizer, scheduler, scaler, status=pause_status, reason=pause_reason
                )
                break
            budget.require_time()
            # If the deadline arrives mid-group, restore the RNG from this
            # clean boundary before saving the previous completed update.
            update_rng = capture_update_rng() if safety_policy is not None else None
            update_pending = True
            group_start = (update_index - 1) * config.accumulation_steps
            examples = context["train_examples"][
                group_start : group_start + config.accumulation_steps
            ]
            step_started = time.perf_counter()
            student.train()
            if arm == "on_policy":
                # All eight fresh responses come from the same unchanged policy.
                record_progress(update_index, "student_generation")
                trajectories = []
                for example_index, example in enumerate(examples):
                    if safety_policy is not None:
                        budget.require_time()
                        record_progress(
                            update_index,
                            "student_generation",
                            example_index=example_index,
                            example_id=example["example_id"],
                        )
                    trajectories.append(
                        context["sample_trajectory"](
                            student,
                            context["tokenizer"],
                            example,
                            device=context["device"],
                            dtype=context["compute_dtype"],
                            max_new_tokens=config.max_response_length,
                            stop_token_ids=context["stop_token_ids"],
                            policy_step=update_index - 1,
                        )
                    )
                record_progress(update_index, "teacher_scoring_and_backward")
                metrics = context["backward_opd_group"](
                    student,
                    teacher,
                    trajectories,
                    context["device"],
                    context["compute_dtype"],
                    config,
                    scaler,
                    policy_step=update_index - 1,
                    before_trajectory=budget.require_time if safety_policy is not None else None,
                )
            elif arm == "supervised_kd":
                # Reference IDs supply off-policy prefixes. The same teacher,
                # full-head divergence and response averaging isolate data source.
                trajectories = []
                for example in examples:
                    budget.require_time()
                    prompt = example["input_ids"][: example["prompt_length"]]
                    response = example["input_ids"][example["prompt_length"] :]
                    trajectories.append(
                        {
                            "example_id": example["example_id"],
                            "policy_step": update_index - 1,
                            "sampling_source": "reference",
                            "response_text": context["tokenizer"].decode(
                                response, skip_special_tokens=True
                            ),
                            "prompt_token_ids": prompt,
                            "response_token_ids": response,
                            "complete_sequence_ids": prompt + response,
                            "prompt_length": len(prompt),
                            "response_length": len(response),
                            "teacher_prompt_token_ids": example["teacher_prompt_token_ids"],
                            "teacher_prompt_length": len(example["teacher_prompt_token_ids"]),
                            "ended_with_eos": response[-1] in context["stop_token_ids"],
                            "hit_token_limit": False,
                        }
                    )
                context["append_trajectories"](
                    run_directory / "reference_trajectories.jsonl", trajectories
                )
                record_progress(update_index, "reference_teacher_scoring_and_backward")
                metrics = context["backward_opd_group"](
                    student,
                    teacher,
                    trajectories,
                    context["device"],
                    context["compute_dtype"],
                    config,
                    scaler,
                    policy_step=update_index - 1,
                    before_trajectory=budget.require_time,
                )
            else:
                # Reference response labels receive cross-entropy. Prompt/padding
                # labels are -100, preserving real EOS tokens as training targets.
                record_progress(update_index, "supervised_forward_and_backward")
                metrics = context["backward_sft_group"](
                    student,
                    examples,
                    context["tokenizer"].pad_token_id,
                    context["device"],
                    context["compute_dtype"],
                    scaler,
                )
            record_progress(update_index, "optimizer_step")
            update_metrics = context["optimizer_update"](
                student, optimizer, scheduler, scaler, config.max_grad_norm
            )
            if torch.device(context["device"]).type == "cuda":
                torch.cuda.synchronize()
            record = {
                "step": update_index,
                **metrics,
                **update_metrics,
                "update_seconds": time.perf_counter() - step_started,
                "peak_allocated_gpu_gib_so_far": torch.cuda.max_memory_allocated() / 2**30
                if torch.device(context["device"]).type == "cuda"
                else None,
            }
            if torch.device(context["device"]).type == "cuda":
                record["gpu_memory"] = [
                    {
                        "index": index,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                    }
                    for index in range(torch.cuda.device_count())
                ]
            if arm == "on_policy":
                context["append_trajectories"](trajectory_path, trajectories)
                del trajectories
            history.append(record)
            update_pending = False
            health = assess_health()
            for warning in health["warnings"]:
                print("REVIEW:", warning, flush=True)
            context["write_json"](run_directory / f"{arm}_history.json", history)
            record_progress(update_index, "training_update", "completed")
            print(
                f"{arm} [{config.eos_loss_mode}] | update {update_index:3d}/{config.optimizer_steps} | "
                f"loss {metrics['loss']:.4f} | mean response tokens {metrics['response_tokens'] / metrics['examples']:.1f} | "
                f"cap stops {metrics['token_limit_stops']}/{metrics['examples']} | EOS stops {metrics.get('eos_stops', 'reference')}",
                flush=True,
            )
            interval = (
                config.pilot_check_interval
                if arm == "on_policy" and update_index <= config.pilot_updates
                else config.checkpoint_interval
            )
            validation_interval = (
                config.pilot_check_interval
                if update_index <= config.pilot_updates
                else config.validation_interval
            )
            boundary = (
                update_index % interval == 0 or update_index == target_steps or health["pause"]
            )
            if health["pause"]:
                pause_status, pause_reason = pause_kind(health), "training safety threshold"
            elif update_index == target_steps and target_steps < config.optimizer_steps:
                pause_status, pause_reason = "pilot_complete", "pilot boundary"
            if boundary:
                latest_snapshot = save_boundary(
                    student, optimizer, scheduler, scaler, status=pause_status, reason=pause_reason
                )
            if pause_status is not None:
                # finish_pause saves each follow-up answer incrementally.
                break
            if update_index % validation_interval == 0 or update_index == target_steps:
                # Save BEFORE the potentially slow generation of validation answers.
                if not boundary:
                    latest_snapshot = save_boundary(student, optimizer, scheduler, scaler)
                measure_boundary(
                    student,
                    record,
                    f"{arm}_boundary_{update_index:04d}_validation",
                    generate_answers=arm != "sft_control" or update_index == target_steps,
                )
                context["write_json"](run_directory / f"{arm}_history.json", history)
                # Fresh validation can pause immediately, before the next update.
                health = assess_health()
                if health["pause"]:
                    pause_status, pause_reason = pause_kind(health), "validation quality threshold"
                    latest_snapshot = save_boundary(
                        student,
                        optimizer,
                        scheduler,
                        scaler,
                        status=pause_status,
                        reason=pause_reason,
                    )
                    break
    except TrainingWalltimePause:
        if update_pending and update_rng is not None:
            optimizer.zero_grad(set_to_none=True)
            restore_update_rng(update_rng)
        pause_status, pause_reason = "walltime_paused", "SLURM checkpoint reserve"
        latest_snapshot = save_boundary(
            student, optimizer, scheduler, scaler, status=pause_status, reason=pause_reason
        )
    except (KeyboardInterrupt, RuntimeError, FloatingPointError, OSError) as error:
        # An incomplete update is never published as a valid recovery snapshot.
        # Existing complete boundaries remain available even if storage is full.
        try:
            failure = record_training_failure(
                run_directory,
                error=error,
                step=len(history),
                snapshot=latest_snapshot,
                signature=signature,
            )
            if safety_policy is not None:
                context["export_report"](
                    context["PROJECT_ROOT"],
                    run_directory,
                    failure,
                    report_name=f"report_failure_{len(history):04d}_{time.time_ns()}",
                )
        except (OSError, ValueError) as export_error:
            print("Failure-report export could not finish:", export_error, flush=True)
        del optimizer, scheduler, scaler, student
        context["release_cuda_memory"]()
        raise
    elapsed = time.perf_counter() - started
    peak_gib = (
        torch.cuda.max_memory_allocated() / 2**30
        if torch.device(context["device"]).type == "cuda"
        else None
    )
    del optimizer, scheduler, scaler
    context["release_cuda_memory"]()
    if arm == "on_policy":
        context["validate_trajectory_log"](
            trajectory_path, context["train_examples"], len(history), config.accumulation_steps
        )
    if pause_status is not None or len(history) < config.optimizer_steps:
        status = pause_status or "pilot_complete"
        try:
            return finish_pause(student, status, pause_reason or "pilot boundary", latest_snapshot)
        finally:
            del student
            context["release_cuda_memory"]()

    record_progress(len(history), "final_checkpoint")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    if final_directory.exists():
        context["verify_inventory"](final_directory)
    else:
        context["save_checkpoint"](student, context["tokenizer"], final_directory)
    checkpoint_verified = context["verify_checkpoint"](
        student,
        context["tokenizer"],
        final_directory,
        context["validation_examples"][0]["input_ids"],
        context["device"],
        context["compute_dtype"],
        probe_texts=["  spaces\n", "3 + 2 = 5.\n#### 5", "Café 中文 🙂"],
    )
    record_progress(len(history), "final_checkpoint", "completed")
    record_progress(len(history), "validation")
    final_validation = context["evaluate_and_save"](
        student,
        context["validation_examples"],
        f"{arm}_boundary_{len(history):04d}_validation",
        directory=run_directory,
        identity=signature,
        budget=budget,
    )
    result = {
        "arm": arm,
        "status": "completed",
        "signature": signature,
        "eos_loss_mode": config.eos_loss_mode,
        "validation_status": "completed",
        "health": assess_health(),
        "completed_steps": len(history),
        "history": history,
        "current_session_loop_seconds": elapsed,
        "peak_allocated_gpu_gib": peak_gib,
        "summed_recorded_update_seconds": sum(row["update_seconds"] for row in history),
        "maximum_recorded_peak_gpu_gib": max(
            (
                row["peak_allocated_gpu_gib_so_far"]
                for row in history
                if row["peak_allocated_gpu_gib_so_far"] is not None
            ),
            default=None,
        ),
        "checkpoint_reload_verified": checkpoint_verified,
        "final_validation": context["evaluation_summary"](final_validation, arm, "validation"),
        "final_validation_reference_nll": history[-1]["validation_reference_nll"],
        "resumed_from_step": completed_steps,
    }
    context["write_json"](result_path, result)
    record_progress(len(history), "validation", "completed")
    del student
    context["release_cuda_memory"]()
    return result
