"""Disposable real-GPU shape/update/reload qualification, separate from research data."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from src.opd import OPDConfig, optimizer_update, trajectory_kl
from src.opd_experiment import SlurmBudget, isolated_evaluation_rng
from src.sft import save_checkpoint, verify_checkpoint


def qualify_gpu(student, teacher, tokenizers, profile, root):
    """Stress configured shapes using synthetic IDs, then discard updated weights."""
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    from src.compat import generation_options

    device = next(student.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("The real-model qualification requires allocated CUDA.")
    budget = SlurmBudget.from_environment()
    started = time.perf_counter()

    class CheckDeadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            budget.require_time()
            return False

    tokenizer = tokenizers["student"]
    # These deliberately repeated IDs measure memory at long shapes. They are
    # synthetic inputs and supply no evidence about arithmetic or model quality.
    token = tokenizer.encode(" pencil", add_special_tokens=False)[-1]
    prompt = [token] * 512
    response = [token] * 255 + [profile["eos_token_ids"][0]]
    teacher_prefix = [token] * 544
    trajectory = {
        "prompt_token_ids": prompt,
        "response_token_ids": response,
        "complete_sequence_ids": prompt + response,
        "prompt_length": 512,
        "response_length": 256,
        "teacher_prompt_token_ids": teacher_prefix,
        "teacher_prompt_length": 544,
    }
    config = OPDConfig(
        eos_loss_mode="semantic_stop",
        eos_token_ids=tuple(profile["eos_token_ids"]),
        teacher_context="qwen3_5_nonthinking_v1",
        max_sequence_length=1056,
    )
    teacher_parameter = next(teacher.parameters())
    teacher_before = teacher_parameter.detach().reshape(-1)[:32].clone()
    with isolated_evaluation_rng():
        budget.require_time()
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        student.train()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        optimizer.zero_grad(set_to_none=True)
        loss = trajectory_kl(student, teacher, trajectory, device, torch.bfloat16, config)
        scaler.scale(loss).backward()
        update = optimizer_update(student, optimizer, scheduler, scaler, 1.0)
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert torch.equal(teacher_parameter.detach().reshape(-1)[:32], teacher_before)
        measured_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        del loss, optimizer, scheduler, scaler
        student.eval()
        budget.require_time()
        # min_new_tokens deliberately forces a long memory probe. This policy
        # is exclusive to qualification and never supplies training trajectories.
        policy = GenerationConfig(
            max_new_tokens=512,
            min_new_tokens=512,
            do_sample=False,
            num_beams=1,
            eos_token_id=profile["eos_token_ids"],
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=None,
            use_cache=True,
        )
        inputs = torch.tensor([prompt], device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = student.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=policy,
                stopping_criteria=StoppingCriteriaList([CheckDeadline()]),
                **generation_options(student),
            )
            assert generated.shape == (1, 1024)
            # Teacher validation uses a chat prefix plus up to 512 new IDs.
            budget.require_time()
            teacher_ids = torch.tensor([teacher_prefix + [token] * 512], device=device)
            scores = teacher(
                input_ids=teacher_ids, attention_mask=torch.ones_like(teacher_ids), use_cache=False
            ).logits
            assert scores.shape == (1, 1056, profile["output_vocab_size"])
            assert torch.isfinite(scores).all()
            del scores, teacher_ids, generated, inputs
        # Check actual HF serialization and reload after freeing optimizer state.
        with tempfile.TemporaryDirectory(
            prefix="qualification-", dir=Path(root) / "results"
        ) as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            save_checkpoint(student, tokenizer, checkpoint)
            verified = verify_checkpoint(
                student, tokenizer, checkpoint, prompt[:8], device, torch.bfloat16
            )
    memory = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        memory.append(
            {
                "index": index,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
            }
        )
    return {
        "status": "disposable_gpu_qualification_passed",
        "qualification_fingerprint": qualification_fingerprint(profile, root),
        "total_seconds": time.perf_counter() - started,
        "scope": "Synthetic shape/update test. No task-quality claim.",
        "training_response_shape": [1, 256, profile["output_vocab_size"]],
        "student_cached_complete_shape": [1, 1024],
        "teacher_forward_shape": [1, 1056, profile["output_vocab_size"]],
        "loss": measured_loss,
        "optimizer_update": update,
        "checkpoint_reload_verified": verified,
        "weights_persisted": False,
        "gpu_memory": memory,
    }


def qualification_fingerprint(profile, root):
    """Identify the loader/loss implementation and the allocated hardware class."""
    import hashlib
    import importlib.metadata
    import json

    import torch

    from src.utils import sha256_file

    names = (
        "src/qwen38/model_upgrade.py",
        "src/qwen38/qualification.py",
        "src/opd_loss.py",
        "src/profiles.py",
        "src/compat.py",
    )
    identity = {
        "profile": profile,
        "source": {name: sha256_file(Path(root) / name) for name in names},
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "tokenizers")
        },
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def require_qualification(root, profile):
    """Reuse a completed qualification for this profile/software/GPU class."""
    import json

    expected = qualification_fingerprint(profile, root)
    paths = list((Path(root) / "results").glob("qwen38_qualification_*/qualification.json"))
    paths += list((Path(root) / "results").glob("qwen38_readiness_*/readiness.json"))
    for path in sorted(paths, reverse=True):
        saved = json.loads(path.read_text())
        result = saved.get("gpu_qualification", saved)
        if (
            result
            and result.get("status") == "disposable_gpu_qualification_passed"
            and result.get("qualification_fingerprint") == expected
        ):
            return path
    raise RuntimeError(
        "Run Notebook 05 in gpu_check mode for this model profile, software and GPU class before the KD pilots."
    )
