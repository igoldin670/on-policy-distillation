"""One disposable GPU update to check 04's real models and peak memory.

Run inside a SLURM GPU allocation, from the project root. This is a hardware
qualification test. It does not save trained weights or establish task quality.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("Choose between 1 and 256 new tokens.")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/source-verification"))

    import torch
    from transformers import AutoTokenizer

    from src.generation import sample_trajectory
    from src.logprobs import teacher_prompt_token_ids, verify_shared_tokenizer
    from src.opd import (
        TEACHER_ID,
        TEACHER_REVISION,
        OPDConfig,
        backward_opd_group,
        cuda_preflight,
        load_student,
        load_teacher,
        optimizer_update,
        resolve_cold_start,
        seed_everything,
    )
    from src.sft import load_checkpoint_tokenizer

    config = OPDConfig()
    seed_everything(config.seed)
    runtime = cuda_preflight(ROOT)
    _, checkpoint_directory, evidence = resolve_cold_start(
        ROOT, report_path=args.report, checkpoint_path=args.checkpoint
    )
    print(json.dumps(runtime, indent=2), flush=True)
    tokenizer = load_checkpoint_tokenizer(checkpoint_directory)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_ID, revision=TEACHER_REVISION, token=False
    )
    # Use a recorded validation question without supplying its answer to generation.
    question = evidence["splits"]["validation"][0]["prompt"]
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    device = torch.device("cuda:0")
    teacher = load_teacher(device)
    verify_shared_tokenizer(tokenizer, teacher_tokenizer, [question], teacher.config.vocab_size)
    student = load_student(checkpoint_directory, device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = sorted({student.config.eos_token_id, teacher.config.eos_token_id})
    example = {
        "example_id": "hardware_smoke/validation_question",
        "prompt": question,
        "teacher_prompt_token_ids": teacher_prompt_token_ids(teacher_tokenizer, question),
    }
    # sample_trajectory returns a single student's own response, without gradients.
    # Both models then score the same response IDs under their own prompt contexts.
    trajectory = sample_trajectory(
        student,
        tokenizer,
        example,
        device=device,
        dtype=torch.bfloat16,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids,
        policy_step=0,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # A constant nonzero LR exercises a real update and allocates Adam's state.
    # This disposable test omits the pilot's warmup and eight-response accumulation.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = backward_opd_group(
        student, teacher, [trajectory], device, torch.bfloat16, config, scaler, 0
    )
    metrics.update(optimizer_update(student, optimizer, scheduler, scaler, config.max_grad_norm))
    assert optimizer.state, "Adam states must exist after the smoke update."
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    memory = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        free, total = torch.cuda.mem_get_info(index)
        memory.append(
            {
                "index": index,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
                "free_gib_after_step": free / 2**30,
                "total_gib": total / 2**30,
            }
        )
    print(
        json.dumps(
            {
                "status": "one_disposable_update_passed",
                "scope": "One response; longer prompts/responses and full pilot remain unverified.",
                "metrics": metrics,
                "memory": memory,
                "teacher_device_map": teacher.hf_device_map,
                "trajectory": trajectory,
                "weights_saved": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
