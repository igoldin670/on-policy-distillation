"""One fresh student trajectory, sampled without a gradient or hidden filtering."""

from __future__ import annotations

from src.sft import precision_context


def sample_trajectory(
    student, tokenizer, example, *, device, dtype, max_new_tokens, stop_token_ids, policy_step
):
    import torch
    from transformers import GenerationConfig

    prompt_token_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
    if not prompt_token_ids:
        raise ValueError("Cannot sample an empty prompt.")
    prompt_inputs = {
        "input_ids": torch.tensor([prompt_token_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_token_ids)), dtype=torch.long, device=device),
    }
    # Construct a fresh policy so a model's stored chat-generation defaults cannot
    # silently add top-k filtering, repetition penalties, or different end IDs.
    generation_policy = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        repetition_penalty=1.0,
        eos_token_id=list(stop_token_ids),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,
        use_cache=True,
    )
    was_training = student.training
    student.eval()
    try:
        with torch.no_grad(), precision_context(device, dtype):
            complete_sequence_ids = student.generate(
                **prompt_inputs,
                generation_config=generation_policy,
                # Otherwise Transformers can replace temperature=1/top_p=1 with
                # non-default values stored in this model's generation config.
                use_model_defaults=False,
            )
    finally:
        student.train(was_training)
    if complete_sequence_ids.ndim != 2 or complete_sequence_ids.shape[0] != 1:
        raise ValueError("Expected a single generated sequence: [1, L].")
    complete = complete_sequence_ids[0].tolist()
    prompt_length = len(prompt_token_ids)
    response = complete[prompt_length:]
    if complete[:prompt_length] != prompt_token_ids or not 1 <= len(response) <= max_new_tokens:
        raise ValueError("Generated response or original prompt boundary is invalid.")
    ended = response[-1] in stop_token_ids
    return {
        "example_id": example["example_id"],
        "policy_step": policy_step,
        "prompt_token_ids": prompt_token_ids,
        "response_token_ids": response,
        "prompt_length": prompt_length,
        "response_length": len(response),
        "complete_sequence_ids": complete,
        "response_text": tokenizer.decode(response, skip_special_tokens=True),
        "ended_with_eos": ended,
        "hit_token_limit": len(response) == max_new_tokens and not ended,
    }
