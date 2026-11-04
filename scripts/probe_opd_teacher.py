"""Read-only context probe for an ALREADY LOADED frozen Qwen3 teacher.

Run in the same Colab notebook kernel after interrupting the training cell:
    from scripts.probe_opd_teacher import probe_teacher_context
    row = json.loads((run_directory / "training_trajectories.jsonl").read_text().splitlines()[0])
    display(pd.DataFrame(probe_teacher_context(teacher, tokenizer, row, compute_dtype)))

The chat context is a diagnostic comparison. It is not a training change and does
not establish that changing the prompt would improve the student.
"""

from __future__ import annotations

import math

import torch

from src.logprobs import teacher_prompt_token_ids
from src.sft import precision_context


def probe_teacher_context(teacher, tokenizer, trajectory, dtype, temperature=0.1):
    """Compare raw and non-thinking chat contexts on identical saved response IDs."""
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("Use the frozen teacher already loaded by Notebook 04.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive.")
    prompt = trajectory["prompt_token_ids"]
    response = trajectory["response_token_ids"]
    if not prompt or not response:
        raise ValueError("A complete saved prompt and response are required.")
    prompt_text = tokenizer.decode(prompt, skip_special_tokens=False)
    chat_prompt = teacher_prompt_token_ids(tokenizer, prompt_text)
    device = next(teacher.parameters()).device
    results = []
    was_training = teacher.training
    teacher.eval()
    try:
        # No generation, sampling, optimizer, or gradient updates occur here.
        with torch.no_grad():
            for context, prefix in (
                ("current_raw_prompt", prompt),
                ("nonthinking_chat", chat_prompt),
            ):
                ids = torch.tensor([prefix + response], dtype=torch.long, device=device)
                with precision_context(device, dtype):
                    logits = teacher(
                        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                    ).logits
                # [1, L, V] -> [2, V]. Each selected position predicts one saved
                # response token, including the last token (often EOS in early rows).
                positions = sorted({0, len(response) - 1})
                selected = logits[0, [len(prefix) - 1 + position for position in positions]].float()
                del logits
                log_probs = selected.log_softmax(-1)
                targets = (selected / temperature).log_softmax(-1)
                for index, position in enumerate(positions):
                    top_values, top_ids = log_probs[index].topk(5)
                    results.append(
                        {
                            "context": context,
                            "response_position": position,
                            "observed_token": repr(tokenizer.decode([response[position]])),
                            "observed_log_probability_T1": log_probs[
                                index, response[position]
                            ].item(),
                            "observed_target_log_probability": targets[
                                index, response[position]
                            ].item(),
                            "stop_probability_T1": log_probs[index, [151643, 151645]]
                            .exp()
                            .sum()
                            .item(),
                            "stop_target_probability": targets[index, [151643, 151645]]
                            .exp()
                            .sum()
                            .item(),
                            # Log-space mass remains readable when a sharpened
                            # probability underflows to zero in FP32.
                            "stop_log_probability_T1": log_probs[index, [151643, 151645]]
                            .logsumexp(0)
                            .item(),
                            "stop_target_log_probability": targets[index, [151643, 151645]]
                            .logsumexp(0)
                            .item(),
                            "top_next_tokens_T1": [
                                {
                                    "token": repr(tokenizer.decode([token_id])),
                                    "probability": value.exp().item(),
                                }
                                for token_id, value in zip(top_ids.tolist(), top_values)
                            ],
                        }
                    )
                del selected, log_probs, targets, ids
    finally:
        teacher.train(was_training)
    return results
