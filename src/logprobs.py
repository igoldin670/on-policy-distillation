"""Explicit causal alignment: response token j is predicted at position j - 1."""

from __future__ import annotations


def response_prediction_logits(logits, complete_sequence_ids, prompt_length: int):
    """[1, L, V] -> [1, R, V], keeping response predictions only (no padding)."""
    import torch

    if complete_sequence_ids.ndim != 2 or complete_sequence_ids.shape[0] != 1:
        raise ValueError("Expected one unpadded complete sequence: [1, L].")
    length = complete_sequence_ids.shape[1]
    if not 1 <= prompt_length < length:
        raise ValueError("Need a nonempty prompt and a nonempty response.")
    if logits.ndim != 3 or logits.shape[:2] != complete_sequence_ids.shape:
        raise ValueError("Logits must have shape [1, complete_length, vocabulary].")
    # Logits after the LAST prompt token predict the FIRST response token.
    # The final sequence position has no observed next token, so it is omitted.
    response_logits = logits[:, prompt_length - 1 : -1, :]
    targets = complete_sequence_ids[:, prompt_length:]
    shifted_targets = complete_sequence_ids[:, 1:][:, prompt_length - 1 :]
    if response_logits.shape[1] != targets.shape[1] or not torch.equal(shifted_targets, targets):
        raise ValueError("Response predictions are not aligned to response IDs.")
    if targets.min() < 0 or targets.max() >= logits.shape[-1]:
        raise ValueError("A response ID is outside the output vocabulary.")
    return response_logits


def selected_log_probs(response_logits, response_token_ids):
    """Return [1, R] scores for observed tokens, useful for inspection only."""
    import torch

    if response_logits.shape[:2] != response_token_ids.shape:
        raise ValueError("One observed token is required for each prediction.")
    # Float32 softmax is more stable than BF16/FP16 probability arithmetic.
    return (
        torch.log_softmax(response_logits.float(), dim=-1)
        .gather(-1, response_token_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def verify_shared_tokenizer(student_tokenizer, teacher_tokenizer, texts, vocab_size):
    """Verify meanings AND text encoding; matching vocabulary sizes is insufficient."""
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher and student token IDs have different meanings.")
    if max(student_tokenizer.get_vocab().values()) >= vocab_size:
        raise ValueError("Tokenizer IDs exceed the model's output head.")
    for text in texts:
        left = student_tokenizer.encode(text, add_special_tokens=False)
        right = teacher_tokenizer.encode(text, add_special_tokens=False)
        if left != right or student_tokenizer.decode(left) != teacher_tokenizer.decode(right):
            raise ValueError("Teacher and student tokenize/decode the same text differently.")
    # EOS *roles* may differ even when all token meanings are identical.
    return {
        "identical_token_mapping": True,
        "text_probes_checked": len(texts),
        "output_vocabulary_size": vocab_size,
        "student_eos_token_id": student_tokenizer.eos_token_id,
        "teacher_eos_token_id": teacher_tokenizer.eos_token_id,
    }
