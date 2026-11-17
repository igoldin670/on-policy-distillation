"""Pinned Qwen3.8 migration, isolated from historical experiment hash inputs.

Importing this module downloads nothing and loads no weights. A new model needs
new baseline/SFT evidence; old Qwen3 checkpoints are deliberately unsupported.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.logprobs import response_prediction_logits
from src.opd_loss import forward_kl_loss


def load_upgrade_profile(root: Path) -> dict:
    import yaml

    profile = yaml.safe_load((Path(root) / "configs/small_experiment.yaml").read_text())
    if profile["profile"] != "qwen38" or profile["vocabulary_mode"] != "full_head":
        raise ValueError("Use the explicit Qwen3.8 full-head profile.")
    for role in ("student", "teacher"):
        revision = profile[f"{role}_revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Pin both models to full hexadecimal commit revisions.")
    if not 0 < profile["tokenizer_vocab_size"] <= profile["output_vocab_size"]:
        raise ValueError("Shared IDs must fit both output heads.")
    if not set(profile["eos_token_ids"]).issubset(range(profile["tokenizer_vocab_size"])):
        raise ValueError("Both stopping IDs must belong to the shared vocabulary.")
    return profile


def text_config_values(config) -> dict:
    """Published repositories include vision; this experiment uses only text."""
    values = config if isinstance(config, dict) else config.to_dict()
    text = values.get("text_config", values)
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError("A Qwen3 checkpoint cannot initialize the new text architecture.")
    return text


def verify_upgrade_tokenizers(student, teacher, student_config, teacher_config, profile, texts):
    """Check fully resolved tokenizer mappings, BPE rules, and text round trips.

    The student's raw tokenizer.json omits 11 tokens which its official
    tokenizer_config.json supplies. AutoTokenizer resolves both files normally.
    Comparing just tokenizer.json or just output-head sizes gives a false result.
    """
    left, right = student.get_vocab(), teacher.get_vocab()
    width = profile["tokenizer_vocab_size"]
    if left != right:
        raise ValueError("Teacher and student token IDs have different meanings.")
    if set(left.values()) != set(range(width)):
        raise ValueError("Resolved tokenizer IDs differ from the pinned vocabulary.")
    student_backend = json.loads(student.backend_tokenizer.to_str())
    teacher_backend = json.loads(teacher.backend_tokenizer.to_str())
    for component in ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder"):
        if student_backend[component] != teacher_backend[component]:
            raise ValueError(f"Tokenizer {component} differs; a size check is insufficient.")
    for config in (student_config, teacher_config):
        if text_config_values(config)["vocab_size"] != profile["output_vocab_size"]:
            raise ValueError("An output head differs from the pinned experiment.")
    for text in texts:
        a = student.encode(text, add_special_tokens=False)
        b = teacher.encode(text, add_special_tokens=False)
        if a != b or student.decode(a) != teacher.decode(b):
            raise ValueError(f"Text encoding or decoding differs: {text!r}")
    stops = [student.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|im_end|>")]
    if stops != profile["eos_token_ids"]:
        raise ValueError("Do not reuse Qwen3's old stopping IDs.")
    return {
        "status": "identical_tokenizers_verified",
        "full_tokenizers_identical": True,
        "tokenizer_vocab_size": width,
        "output_head_size": profile["output_vocab_size"],
        "text_probes_checked": len(texts),
        "eos_token_ids": stops,
        "vocabulary_mode": "full_head",
        "scope": "Identical resolved IDs and BPE; not a quality or training result.",
    }


def representative_texts():
    return [
        "Question: What is 24 * 15 - 87?\n\nAnswer:\n",
        "24 * 15 = 360. 360 - 87 = 273.\n#### 273",
        "\t  spaces\n\n",
        "£12.50 − 3 = 9.50",
        "Café 中文 🙂 １２３",
        "<think>Reasoning.</think>",
        "<tool_response>42</tool_response>",
        "<|audio_start|><|audio_end|>",
    ] + [f"{n} + {n + 1} = {2 * n + 1}.\n#### {2 * n + 1}" for n in range(1000)]


def upgrade_teacher_prompt_ids(tokenizer, prompt):
    """Request a flat ID list explicitly; Transformers 5 defaults to a mapping."""
    if not prompt or not tokenizer.chat_template:
        raise ValueError("A prompt and the pinned teacher template are required.")
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
        raise ValueError("Teacher context must be a flat list of token IDs.")
    return ids


def inspect_upgrade(profile, *, sources: Path | None = None):
    """CPU-only: fetch configurations/tokenizers or inspect already downloaded ones."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizers, configs = {}, {}
    for role in ("student", "teacher"):
        if sources:
            from src.utils import sha256_file

            manifest = json.loads((sources / role / "download_manifest.json").read_text())
            if (manifest["model_id"], manifest["revision"]) != (
                profile[f"{role}_model_id"],
                profile[f"{role}_revision"],
            ):
                raise ValueError("Local inspection files do not match the pinned model revision.")
            for name, expected_hash in manifest["sha256"].items():
                if sha256_file(sources / role / name) != expected_hash:
                    raise ValueError(f"Inspection source checksum differs: {role}/{name}")
        location = str(sources / role) if sources else profile[f"{role}_model_id"]
        options = (
            {"local_files_only": True}
            if sources
            else {
                "revision": profile[f"{role}_revision"],
                "token": False,
            }
        )
        tokenizers[role] = AutoTokenizer.from_pretrained(location, **options)
        configs[role] = AutoConfig.from_pretrained(location, **options)
    report = verify_upgrade_tokenizers(
        tokenizers["student"],
        tokenizers["teacher"],
        configs["student"],
        configs["teacher"],
        profile,
        representative_texts(),
    )
    # Exercise the actual new chat-template API during CPU inspection as well.
    prefix = upgrade_teacher_prompt_ids(tokenizers["teacher"], "What is 2 + 2?")
    report["nonthinking_teacher_prompt_tokens"] = len(prefix)
    return tokenizers, configs, report


def generation_policy(profile, pad_token_id, *, max_new_tokens=64):
    from transformers import GenerationConfig

    # Preserve the student's unfiltered distribution. If it samples an unused
    # output-head row, the trajectory check fails rather than silently masking it.
    return GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens,
        eos_token_id=profile["eos_token_ids"],
        pad_token_id=pad_token_id,
        bos_token_id=None,
        use_cache=True,
    )


def teacher_memory_budgets(profile):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select an allocated CUDA kernel for the real-model check.")
    count = torch.cuda.device_count()
    if count not in (1, 2):
        raise RuntimeError("Qualify one 80GB GPU or two 40GB-or-larger GPUs first.")
    budgets = {}
    for index in range(count):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This teacher check requires BF16 on every GPU.")
        free, total = torch.cuda.mem_get_info(index)
        minimum = 75 if count == 1 else 38
        if total < minimum * 2**30 or free < (minimum - 1) * 2**30:
            raise RuntimeError("Free model memory first; do not load a second teacher kernel.")
        reserve = (
            profile["student_gpu_reserve_gib"] if index == 0 else profile["other_gpu_reserve_gib"]
        )
        budgets[index] = int(free - reserve * 2**30)
    if sum(budgets.values()) < profile["teacher_weight_bytes"]:
        raise RuntimeError("BF16 teacher cannot fit the reserved weight budgets.")
    return budgets


def upgrade_cache_preflight(profile):
    """Inspect the cache filesystem, including already downloaded teacher blobs."""
    from huggingface_hub.constants import HF_HUB_CACHE

    cache = Path(os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE))
    cache.mkdir(parents=True, exist_ok=True)
    # Count only completed weight files linked by the requested revision.
    # Other cached revisions cannot reduce this download's space requirement.
    snapshot = (
        cache
        / ("models--" + profile["teacher_model_id"].replace("/", "--"))
        / "snapshots"
        / profile["teacher_revision"]
    )
    cached_bytes = sum(
        p.stat().st_size
        for p in snapshot.glob("*.safetensors")
        if p.is_file() and not p.resolve().name.endswith(".incomplete")
    )
    required = max(0, profile["teacher_weight_bytes"] - cached_bytes) + 8 * 2**30
    free = shutil.disk_usage(cache).free
    if free < required:
        raise RuntimeError(
            f"Model cache needs about {required / 2**30:.1f} GiB free; found {free / 2**30:.1f}."
        )
    return {
        "cache": str(cache),
        "filesystem_free_gib": free / 2**30,
        "estimated_additional_gib": required / 2**30,
        "personal_quota_verified": False,
    }


def load_text_weights(location, *, config, dtype, device_map, revision=None):
    """Load the text component, checking for missing or unexpectedly initialized weights."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    options = {"revision": revision, "token": False} if revision else {"local_files_only": True}
    model, information = Qwen3_5ForCausalLM.from_pretrained(
        location,
        config=Qwen3_5TextConfig(**text_config_values(config)),
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
        output_loading_info=True,
        **options,
    )
    for key in ("missing_keys", "mismatched_keys", "error_msgs"):
        if information.get(key):
            raise RuntimeError(f"Text weights did not reload exactly: {key}: {information[key]}")
    unexpected = [
        key
        for key in information.get("unexpected_keys", [])
        if not key.startswith(("model.visual.", "mtp."))
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected non-vision weights: {unexpected[:5]}")
    return model


def load_upgrade_models(profile, configs, *, load_student=True):
    """Frozen BF16 teacher is sharded; FP32 student stays on logical CUDA GPU 0."""
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    print("Model-cache preflight:", upgrade_cache_preflight(profile), flush=True)
    budgets = teacher_memory_budgets(profile)
    for index in budgets:
        torch.cuda.reset_peak_memory_stats(index)
    with init_empty_weights():
        skeleton = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text_config_values(configs["teacher"])))
        skeleton.tie_weights()
    placement = infer_auto_device_map(
        skeleton,
        max_memory=budgets,
        no_split_module_classes=skeleton._no_split_modules,
        dtype=torch.bfloat16,
    )
    del skeleton
    if not set(placement.values()).issubset(budgets):
        raise RuntimeError("Teacher planning requires CPU/disk offload; this setup rejects it.")
    print("Loading the pinned text-only BF16 teacher across:", placement, flush=True)
    teacher = load_text_weights(
        profile["teacher_model_id"],
        config=configs["teacher"],
        dtype=torch.bfloat16,
        device_map=placement,
        revision=profile["teacher_revision"],
    )
    teacher.requires_grad_(False).eval()
    teacher.config.use_cache = False
    if not load_student:
        return None, teacher
    print("Loading the new FP32 student on CUDA GPU 0...", flush=True)
    try:
        student = load_text_weights(
            profile["student_model_id"],
            config=configs["student"],
            dtype=torch.float32,
            device_map={"": 0},
            revision=profile["student_revision"],
        )
    except BaseException:
        # Release this attempt's frozen teacher if student initialization fails.
        del teacher
        from src.opd import release_cuda_memory

        release_cuda_memory()
        raise
    student.eval()
    student.config.use_cache = False
    return student, teacher


def score_upgrade_trajectory(student, teacher, tokenizers, profile, prompt, *, max_new_tokens=64):
    """Generate once, preserve IDs, and compare aligned, full-head distributions.

    No backward call, optimizer, or weight update occurs in this qualification.
    """
    import torch

    from src.logprobs import selected_log_probs

    device = next(student.parameters()).device
    tokenizer = tokenizers["student"]
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids or any(token >= profile["tokenizer_vocab_size"] for token in prompt_ids):
        raise ValueError("Prompt must be nonempty and encoded in the verified tokenizer IDs.")
    inputs = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    policy = generation_policy(profile, tokenizer.pad_token_id, max_new_tokens=max_new_tokens)
    torch.manual_seed(42)
    student.eval()
    teacher.eval()
    if any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise ValueError("Teacher must be frozen before scoring.")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        complete = student.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            generation_config=policy,
        )
        assert torch.equal(complete[:, : len(prompt_ids)], inputs)
        response = complete[:, len(prompt_ids) :]
        assert response.shape[1] > 0 and (response < profile["tokenizer_vocab_size"]).all()
        teacher_prefix = upgrade_teacher_prompt_ids(tokenizers["teacher"], prompt)
        teacher_ids = torch.tensor([teacher_prefix + response[0].tolist()], device=device)
        student_logits = student(
            input_ids=complete,
            attention_mask=torch.ones_like(complete),
            use_cache=False,
        ).logits
        teacher_logits = teacher(
            input_ids=teacher_ids,
            attention_mask=torch.ones_like(teacher_ids),
            use_cache=False,
        ).logits
        student_response = response_prediction_logits(student_logits, complete, len(prompt_ids))
        teacher_response = response_prediction_logits(
            teacher_logits, teacher_ids, len(teacher_prefix)
        ).to(device)
        assert torch.equal(teacher_ids[:, len(teacher_prefix) :], response)
        assert torch.isfinite(student_response).all() and torch.isfinite(teacher_response).all()
        # Report mass on padded output rows. These rows remain in the KL calculation.
        teacher_full_probs = teacher_response.float().softmax(-1)
        excluded_mass = teacher_full_probs[..., profile["tokenizer_vocab_size"] :].sum(-1)
        # T=1 inspects ordinary teacher probabilities; this is not a training recipe.
        loss = forward_kl_loss(
            student_response,
            teacher_response,
            teacher_temperature=1.0,
            chunk_size=16,
            recompute=False,
            eos_token_ids=profile["eos_token_ids"],
        )
        student_log_probs = selected_log_probs(student_response, response)
        teacher_log_probs = selected_log_probs(teacher_response, response)
        assert student_log_probs.shape == teacher_log_probs.shape == response.shape
        assert torch.isfinite(student_log_probs).all() and torch.isfinite(teacher_log_probs).all()
    ended = response[0, -1].item() in profile["eos_token_ids"]
    memory = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
        free, total = torch.cuda.mem_get_info(index)
        memory.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_gib": total / 2**30,
                "free_gib_after_scoring": free / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
            }
        )
    return {
        "status": "new_model_scoring_passed",
        "weights_updated": False,
        "checkpoint_03_loaded": False,
        "scope": "One response; no SFT, training, quality improvement, or full-run qualification.",
        "prompt": prompt,
        "student_prompt_length": len(prompt_ids),
        "teacher_prompt_length": len(teacher_prefix),
        "response_length": response.shape[1],
        "response_token_ids": response[0].tolist(),
        "student_prompt_token_ids": prompt_ids,
        "teacher_prompt_token_ids": teacher_prefix,
        "student_predicting_first_response_position": len(prompt_ids) - 1,
        "teacher_predicting_first_response_position": len(teacher_prefix) - 1,
        "response_text": tokenizer.decode(response[0].tolist(), skip_special_tokens=True),
        "ended_with_eos": ended,
        "hit_token_limit": response.shape[1] == max_new_tokens and not ended,
        "aligned_logits_shape": list(student_response.shape),
        "full_head_semantic_stop_kl_T1": float(loss),
        "teacher_padded_row_probability_mass_mean_T1": excluded_mass.mean().item(),
        "teacher_padded_row_probability_mass_max_T1": excluded_mass.max().item(),
        "student_log_probs_T1": student_log_probs[0].tolist(),
        "teacher_log_probs_T1": teacher_log_probs[0].tolist(),
        "sampling_policy": policy.to_dict(),
        "gpu_memory": memory,
    }
