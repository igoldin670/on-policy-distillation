"""Exact full-vocabulary forward KL; no sampling gradient or top-k approximation."""

from __future__ import annotations

import math


def forward_kl_per_token(student_logits, teacher_logits, teacher_temperature=0.1):
    """q = softmax(teacher / T), p = softmax(student); return [..., response].

    The student stays at temperature 1. There is no additional T-squared scaling.
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
    # Sum across the ENTIRE vocabulary, not just the token the student sampled.
    return (teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)).sum(-1)


def forward_kl_loss(
    student_logits, teacher_logits, *, teacher_temperature=0.1, chunk_size=16, recompute=True
):
    """Mean KL for one unpadded response, with bounded FP32 softmax workspace.

    Response positions are chunked, but every vocabulary column is retained.
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

    def chunk_sum(student_chunk, teacher_chunk):
        return forward_kl_per_token(student_chunk, teacher_chunk, teacher_temperature).sum()

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
