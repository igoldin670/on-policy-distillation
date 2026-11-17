"""Frozen-model stopping probes on original student token IDs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.opd import load_student, release_cuda_memory, trajectory_tensors
from src.opd_experiment import SlurmBudget, isolated_evaluation_rng
from src.opd_loss import forward_kl_per_token
from src.sft import precision_context


def select_probe_trajectories(fixed, training):
    """Eight fixed validation, four early EOS, four latest capped responses."""
    early = sorted(
        (row for row in training if row["policy_step"] < 4 and row["ended_with_eos"]),
        key=lambda row: (row["policy_step"], row["example_id"]),
    )
    late = sorted(
        (row for row in training if row["policy_step"] >= 13 and row["hit_token_limit"]),
        key=lambda row: (-row["policy_step"], row["example_id"]),
    )
    if len(fixed) < 8 or len(early) < 4 or len(late) < 4:
        raise ValueError("Need eight fixed, four early EOS, and four late capped trajectories.")
    return [
        dict(row, probe_group=group)
        for group, rows in (
            ("fixed_validation", fixed[:8]),
            ("early_completed", early[:4]),
            ("late_capped", late[:4]),
        )
        for row in rows
    ]


def probe_boundaries(trajectory, tokenizer):
    """k response IDs in the prefix -> logits at P+k-1 predict the next token.

    An observed EOS is scored BEFORE that token. The post-cap prefix includes
    all sampled tokens; it has no observed next target. Never append a fake EOS.
    """
    response = trajectory["response_token_ids"]
    size = len(response)
    points = {0: ["first_response_prediction"]}
    points.setdefault(size - 1, []).append(
        "before_observed_eos" if trajectory["ended_with_eos"] else "before_last_observed_token"
    )
    if trajectory["hit_token_limit"]:
        points.setdefault(size, []).append("after_truncation")
    decoded = tokenizer.decode(response, skip_special_tokens=True)
    match = re.search(
        r"(?:^|\n)####[ \t]+[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)[ \t]*(?=\r?\n|$)", decoded
    )
    if match:
        answer_prefix = decoded[: match.end()]
        # Decode cumulative ORIGINAL IDs only to locate a display boundary.
        # No text is re-tokenized, and no generation/training target is rewritten.
        limit = size - int(trajectory["ended_with_eos"])
        for count in range(1, limit + 1):
            if tokenizer.decode(response[:count], skip_special_tokens=True).startswith(
                answer_prefix
            ):
                points.setdefault(count, []).append("after_first_final_answer_line")
                break
    return [
        {
            "response_prefix_length": count,
            "boundaries": labels,
            "observed_next_token_id": response[count] if count < size else None,
        }
        for count, labels in sorted(points.items())
    ]


def stopping_statistics(student_logits, teacher_logits, *, eos_token_ids, temperature):
    """Probe CPU float32 logit gradients; neither model's weights are updated."""
    import torch

    eos = list(eos_token_ids)
    # These detached copies are small [number_of_boundaries, V] tensors.
    # Only the copy receives gradients to explain the two mathematical objectives.
    with torch.enable_grad():
        student = student_logits.detach().float().cpu().requires_grad_()
        teacher = teacher_logits.detach().float().cpu()
        raw = forward_kl_per_token(student, teacher, temperature)
        grouped = forward_kl_per_token(student, teacher, temperature, eos_token_ids=eos)
        raw_gradient = torch.autograd.grad(raw.sum(), student, retain_graph=True)[0]
        grouped_gradient = torch.autograd.grad(grouped.sum(), student)[0]
    student_lp = student.detach().log_softmax(-1)
    teacher_lp = teacher.log_softmax(-1)
    teacher_target_lp = (teacher / temperature).log_softmax(-1)
    rows = []
    for position in range(student.shape[0]):
        row = {
            "raw_token_kl_nats": float(raw[position].detach()),
            "semantic_stop_kl_nats": float(grouped[position].detach()),
        }
        for name, log_probs in (
            ("student", student_lp),
            ("teacher_T1", teacher_lp),
            ("teacher_target", teacher_target_lp),
        ):
            row[name + "_eos_probabilities"] = {
                str(token): log_probs[position, token].exp().item() for token in eos
            }
            stop_log_probability = log_probs[position, eos].logsumexp(-1).item()
            row[name + "_stop_log_probability"] = stop_log_probability
            row[name + "_stop_probability"] = log_probs[position, eos].logsumexp(-1).exp().item()
        row["raw_student_eos_logit_gradients"] = {
            str(token): raw_gradient[position, token].item() for token in eos
        }
        row["grouped_student_eos_logit_gradients"] = {
            str(token): grouped_gradient[position, token].item() for token in eos
        }
        rows.append(row)
    return rows


def probe_checkpoint_set(
    teacher, tokenizer, trajectories, checkpoints, device, dtype, config, *, budget=None
):
    """One frozen teacher, sequential student checkpoints, identical recorded prefixes."""
    import torch

    budget = budget or SlurmBudget()
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("The teacher must be frozen for diagnosis.")
    teacher.eval()
    prepared = []
    with torch.no_grad(), isolated_evaluation_rng():
        for trajectory in trajectories:
            budget.require_time()
            points = probe_boundaries(trajectory, tokenizer)
            ids, mask = trajectory_tensors(trajectory, device, context="teacher")
            with precision_context(device, dtype):
                logits = teacher(input_ids=ids, attention_mask=mask, use_cache=False).logits
            indices = [
                trajectory["teacher_prompt_length"] + point["response_prefix_length"] - 1
                for point in points
            ]
            scores = logits[0, indices].float().cpu()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Teacher stopping-probe logits are not finite.")
            prepared.append((trajectory, points, scores))
            del logits, ids, mask
    results = []
    for name, checkpoint in checkpoints.items():
        budget.require_time()
        student = load_student(Path(checkpoint), device)
        student.eval()
        try:
            with torch.no_grad(), isolated_evaluation_rng():
                for trajectory, points, teacher_scores in prepared:
                    budget.require_time()
                    ids, mask = trajectory_tensors(trajectory, device)
                    with precision_context(device, dtype):
                        logits = student(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    indices = [
                        trajectory["prompt_length"] + point["response_prefix_length"] - 1
                        for point in points
                    ]
                    student_scores = logits[0, indices].float().cpu()
                    if not torch.isfinite(student_scores).all():
                        raise FloatingPointError("Student stopping-probe logits are not finite.")
                    del logits, ids, mask
                    statistics = stopping_statistics(
                        student_scores,
                        teacher_scores,
                        eos_token_ids=config.eos_token_ids,
                        temperature=config.teacher_temperature,
                    )
                    for point, scores in zip(points, statistics):
                        count = point["response_prefix_length"]
                        results.append(
                            {
                                "checkpoint": name,
                                "example_id": trajectory["example_id"],
                                "probe_group": trajectory["probe_group"],
                                **point,
                                "observed_next_token": repr(
                                    tokenizer.decode([point["observed_next_token_id"]])
                                )
                                if point["observed_next_token_id"] is not None
                                else None,
                                "response_prefix_tail": repr(
                                    tokenizer.decode(
                                        trajectory["response_token_ids"][:count],
                                        skip_special_tokens=False,
                                    )[-160:]
                                ),
                                **scores,
                                "student_predicting_position": trajectory["prompt_length"]
                                + count
                                - 1,
                                "teacher_predicting_position": trajectory["teacher_prompt_length"]
                                + count
                                - 1,
                            }
                        )
        finally:
            del student
            release_cuda_memory()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    return {
        "scope": "Frozen forward passes and detached logit gradients; no weight updates.",
        "rows": results,
    }


def load_historical_probes(directory):
    directory = Path(directory)
    fixed = json.loads((directory / "fixed_validation_trajectories.json").read_text())
    training = [
        json.loads(line)
        for line in (directory / "training_trajectories.jsonl").read_text().splitlines()
    ]
    return select_probe_trajectories(fixed, training)
