"""Pinned model identities shared by inspection, SFT, and distillation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from src.baseline import verify_recorded_run
from src.qwen38.model_upgrade import load_text_weights, load_upgrade_profile


def load_profile(root, name="qwen38"):
    root = Path(root)
    if name == "qwen38":
        return load_upgrade_profile(root)
    if name != "legacy":
        raise ValueError("Select qwen38 or legacy.")
    protocol = verify_recorded_run(root / "notebooks/assets/02_baseline_evaluation/recorded_run")[
        "manifest"
    ]["protocol"]
    from src.opd import TEACHER_ID, TEACHER_REVISION, TEACHER_WEIGHT_BYTES

    return {
        "profile": "legacy",
        "student_model_id": protocol["student_model_id"],
        "student_revision": protocol["student_revision"],
        "teacher_model_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "model_type": "qwen3",
        "output_vocab_size": 151936,
        "tokenizer_vocab_size": 151669,
        "eos_token_ids": [151643, 151645],
        "vocabulary_mode": "full_head",
        "teacher_weight_bytes": TEACHER_WEIGHT_BYTES,
        "student_gpu_reserve_gib": 12,
        "other_gpu_reserve_gib": 3,
    }


def profile_protocol(root, profile):
    """Retain the pinned data, prompt, and historically reserved question IDs."""
    original = verify_recorded_run(
        Path(root) / "notebooks/assets/02_baseline_evaluation/recorded_run"
    )
    protocol = copy.deepcopy(original["manifest"]["protocol"])
    protocol.update(
        student_model_id=profile["student_model_id"],
        student_revision=profile["student_revision"],
        model_profile=profile["profile"],
        max_prompt_tokens=512,
    )
    policy = protocol["generation_config"]
    policy.update(
        eos_token_id=profile["eos_token_ids"], max_new_tokens=512, do_sample=False, num_beams=1
    )
    return protocol


def load_profile_student(profile, device, checkpoint=None):
    """Load published text weights or a checksummed SFT/resume checkpoint."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.utils import verify_inventory

    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        verify_inventory(checkpoint)
        config = json.loads((checkpoint / "config.json").read_text())
        if (
            config.get("model_type") != profile["model_type"]
            or config.get("vocab_size") != profile["output_vocab_size"]
        ):
            raise ValueError(
                "Checkpoint architecture or output head differs from the selected profile."
            )
        location, options = str(checkpoint), {"local_files_only": True}
    else:
        location = profile["student_model_id"]
        options = {"revision": profile["student_revision"], "token": False}
        config = AutoConfig.from_pretrained(location, **options)
    if profile["model_type"] == "qwen3_5_text":
        model = load_text_weights(
            location,
            config=config,
            dtype=torch.float32,
            device_map={"": str(device)},
            revision=None if checkpoint else profile["student_revision"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            location, dtype=torch.float32, attn_implementation="sdpa", **options
        ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model.config, "attention_dropout", 0.0) != 0:
        raise ValueError("This experiment requires dropout-free sampling/scoring consistency.")
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())
    return model.train()


def load_profile_tokenizer(profile):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        profile["student_model_id"], revision=profile["student_revision"], token=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
