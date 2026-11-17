"""Exact forward KL over the full vocabulary, optionally grouping EOS aliases."""

from __future__ import annotations

import math


def forward_kl_per_token(
    student_logits, teacher_logits, teacher_temperature=0.1, *, eos_token_ids=None
):
    """q = softmax(teacher / T), p = softmax(student); return [..., response].

    The student stays at temperature 1. There is no additional T-squared scaling.
    Multiple EOS IDs optionally form a single stopping action after normalization.
    Detach teacher scores even when a caller accidentally supplies a live graph.
    """
    import torch

    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Teacher/student logits must have identical [..., R, V] shapes.")
    if not math.isfinite(teacher_temperature) or teacher_temperature <= 0:
        raise ValueError("Teacher temperature must be finite and positive.")
    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(
        teacher_logits.detach().float() / teacher_temperature, dim=-1
    )
    # Each column contributes, including tokens the student did not sample.
    contributions = teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ending_ids = tuple(eos_token_ids) if eos_token_ids is not None else ()
    if len(set(ending_ids)) != len(ending_ids) or any(
        type(token_id) is not int or not 0 <= token_id < student_logits.shape[-1]
        for token_id in ending_ids
    ):
        raise ValueError("EOS IDs must be unique integer vocabulary indices.")
    if len(ending_ids) <= 1:
        return contributions.sum(-1)

    # Temperature has ALREADY been applied and each distribution normalized.
    # logsumexp sums ending probability mass stably in log space. Both ending
    # spellings become one STOP column at EVERY response position, with shape [B,R].
    student_stop = student_log_probs[..., list(ending_ids)].logsumexp(-1)
    teacher_stop = teacher_log_probs[..., list(ending_ids)].logsumexp(-1)
    ending_mask = torch.zeros(
        student_logits.shape[-1], dtype=torch.bool, device=student_logits.device
    )
    ending_mask[list(ending_ids)] = True
    ordinary_kl = contributions.masked_fill(ending_mask, 0).sum(-1)
    return ordinary_kl + teacher_stop.exp() * (teacher_stop - student_stop)


def forward_kl_loss(
    student_logits,
    teacher_logits,
    *,
    teacher_temperature=0.1,
    chunk_size=16,
    recompute=True,
    eos_token_ids=None,
):
    """Mean KL for one unpadded response, with bounded FP32 softmax workspace.

    Response positions are chunked. Every ordinary vocabulary column contributes,
    and an optional EOS group contributes its combined stopping probability.
    Activation checkpointing recomputes chunk probability arithmetic on backward,
    avoiding retention of all FP32 probability tables. The model's own gradient
    checkpointing separately controls transformer activation memory.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("Process one unpadded response at a time: [1, R, V].")
    if student_logits.shape != teacher_logits.shape or student_logits.shape[1] == 0:
        raise ValueError("Need identical nonempty response-logit shapes.")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    # Freeze an iterable once so checkpoint recomputation sees identical EOS IDs.
    eos_token_ids = tuple(eos_token_ids) if eos_token_ids is not None else None

    def chunk_sum(student_chunk, teacher_chunk):
        return forward_kl_per_token(
            student_chunk, teacher_chunk, teacher_temperature, eos_token_ids=eos_token_ids
        ).sum()

    loss_sum = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, student_logits.shape[1], chunk_size):
        student_chunk = student_logits[:, start : start + chunk_size, :]
        teacher_chunk = teacher_logits[:, start : start + chunk_size, :].detach()
        if recompute and torch.is_grad_enabled() and student_chunk.requires_grad:
            loss_sum = loss_sum + checkpoint(
                chunk_sum, student_chunk, teacher_chunk, use_reentrant=False
            )
        else:
            loss_sum = loss_sum + chunk_sum(student_chunk, teacher_chunk)
    loss = loss_sum / student_logits.shape[1]
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite forward KL; the update must stop.")
    if loss.detach().item() < -1e-5:
        raise FloatingPointError("Negative KL exceeds FP32 rounding tolerance.")
    return loss
