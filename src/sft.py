"""Small supervised cold start; no model weights are loaded when this is imported.

Pure data helpers and explicit tensor operations support Notebook 03. This is
reference-solution SFT, not teacher scoring or on-policy distillation.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import unicodedata
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.baseline import analysis_environment, sha256_file, validate_generation_ids
from src.evaluation import extract_strict_final_answer, rescore_response_rows


@dataclass(frozen=True)
class SFTConfig:
    seed: int = 42
    train_examples: int = 256
    validation_examples: int = 64
    max_sequence_length: int = 1024
    epochs: int = 1
    microbatch_size: int = 1
    accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.1
    validation_interval: int = 8

    def __post_init__(self):
        for name in (
            "train_examples",
            "validation_examples",
            "max_sequence_length",
            "epochs",
            "microbatch_size",
            "accumulation_steps",
            "validation_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_sequence_length < 3:
            raise ValueError("Need space for a prompt, response, and end token.")
        for value in (self.learning_rate, self.max_grad_norm):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Learning rate and clipping norm must be finite and positive."
                )
        if (
            not 0 <= self.warmup_fraction < 1
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid warmup fraction or weight decay.")


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def clean_solution(answer: str) -> tuple[str, str]:
    """Remove calculator markup, but retain visible calculations and final line."""
    cleaned = re.sub(r"<<[^<>]*>>", "", answer).strip()
    number = extract_strict_final_answer(cleaned)
    if number is None or "<<" in cleaned or ">>" in cleaned:
        raise ValueError("Reference must contain one valid final #### answer.")
    return cleaned, str(number)


def encode_example(
    tokenizer, question: str, answer: str, protocol: dict, max_length: int
) -> dict:
    target, number = clean_solution(answer)
    prompt = protocol["prompt_template"].format(question=question)
    # Joint tokenization preserves normal text tokenization. Verify the prompt
    # boundary instead of assuming separately tokenized BPE pieces concatenate.
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + target, add_special_tokens=False)["input_ids"]
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("unstable_token_boundary")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an end token.")
    response_ids = full_ids[len(prompt_ids) :]
    if not response_ids:
        raise ValueError("empty_response")
    full_ids = full_ids + [tokenizer.eos_token_id]
    if len(full_ids) > max_length:
        raise ValueError("overlong_sequence")
    labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    assert len(labels) == len(full_ids)
    return {
        "question": question,
        "prompt": prompt,
        "target_text": target,
        "expected_answer": number,
        "input_ids": full_ids,
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "target_length": len(response_ids) + 1,
    }


def prepare_splits(dataset, tokenizer, protocol: dict, config: SFTConfig) -> dict:
    """Select disjoint, length-safe examples from the pinned TRAIN split only."""
    blocked = {normalize_question(e["question"]) for e in protocol["examples"]}
    seen = set(blocked)
    indices = list(range(len(dataset)))
    random.Random(config.seed).shuffle(indices)
    selected, exclusions = [], []
    required = config.train_examples + config.validation_examples
    for index in indices:
        row = dataset[index]
        key = normalize_question(row["question"])
        reason = (
            "reserved_test_overlap"
            if key in blocked
            else "duplicate_question"
            if key in seen
            else None
        )
        if reason is None:
            try:
                example = encode_example(
                    tokenizer,
                    row["question"],
                    row["answer"],
                    protocol,
                    config.max_sequence_length,
                )
            except ValueError as error:
                reason = str(error)
        if reason is not None:
            exclusions.append({"source_index": index, "reason": reason})
            continue
        seen.add(key)
        selected.append(
            {"example_id": f"gsm8k/train/{index}", "source_index": index, **example}
        )
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(f"Only {len(selected)} eligible examples; need {required}.")
    train = selected[: config.train_examples]
    validation = selected[config.train_examples :]
    assert not (
        {e["source_index"] for e in train} & {e["source_index"] for e in validation}
    )
    assert not ({normalize_question(e["question"]) for e in selected} & blocked)
    return {
        "train": train,
        "validation": validation,
        "exclusions": exclusions,
        "source_split": "train",
        "dataset_rows": len(dataset),
    }


def collate_examples(examples: list[dict], pad_token_id: int) -> dict:
    """Right-pad inputs; ignore padding by POSITION, never by token ID."""
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    width = max(len(e["input_ids"]) for e in examples)
    inputs, labels, masks = [], [], []
    for example in examples:
        ids, targets = example["input_ids"], example["labels"]
        if len(ids) != len(targets) or not any(t != -100 for t in targets[1:]):
            raise ValueError("Each example needs aligned IDs and response targets.")
        padding = width - len(ids)
        inputs.append(ids + [pad_token_id] * padding)
        labels.append(targets + [-100] * padding)
        masks.append([1] * len(ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def precision_context(device, dtype):
    import torch

    return (
        torch.autocast(device_type="cuda", dtype=dtype)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def response_loss_sum(model, batch: dict):
    """Return summed response NLL and count, with exactly one causal shift."""
    import torch

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits
    # [B, L, V] -> [B, L-1, V]; target j+1 pairs with logits j.
    targets = batch["labels"][:, 1:]
    count = int((targets != -100).sum().item())
    if count == 0:
        raise ValueError("No response targets remain after shifting.")
    loss_sum = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    if not torch.isfinite(loss_sum):
        raise FloatingPointError("Response loss is not finite; no update is safe.")
    return loss_sum, count


def training_groups(examples: list[dict], config: SFTConfig, epoch: int):
    """Deterministic shuffle; retain the last partial accumulation group."""
    indices = list(range(len(examples)))
    random.Random(config.seed + epoch).shuffle(indices)
    batches = [
        [examples[i] for i in indices[start : start + config.microbatch_size]]
        for start in range(0, len(indices), config.microbatch_size)
    ]
    for start in range(0, len(batches), config.accumulation_steps):
        yield batches[start : start + config.accumulation_steps]


def backward_group(model, group, pad_token_id, device, dtype, scaler) -> dict:
    """Backpropagate one group's TOKEN MEAN without retaining multiple graphs.

    The caller zeroes gradients, unscales/clips them, then steps the optimizer.
    Dividing each microbatch mean by eight is incorrect when lengths differ.
    """
    total_tokens = sum(
        sum(t != -100 for t in e["labels"][1:]) for batch in group for e in batch
    )
    if total_tokens == 0:
        raise ValueError("Empty accumulation group.")
    total_loss = 0.0
    for examples in group:
        batch = {
            k: v.to(device) for k, v in collate_examples(examples, pad_token_id).items()
        }
        with precision_context(device, dtype):
            loss_sum, _ = response_loss_sum(model, batch)
        # Every response token has the same weight, including in a partial group.
        scaler.scale(loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())
    return {
        "loss": total_loss / total_tokens,
        "response_tokens": total_tokens,
        "examples": sum(len(batch) for batch in group),
    }


def validation_loss(model, examples, pad_token_id, device, dtype) -> float:
    import torch

    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        with torch.inference_mode():
            for example in examples:
                batch = {
                    k: v.to(device)
                    for k, v in collate_examples([example], pad_token_id).items()
                }
                with precision_context(device, dtype):
                    loss_sum, count = response_loss_sum(model, batch)
                total_loss += float(loss_sum)
                total_tokens += count
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("No validation response targets.")
    return total_loss / total_tokens


def generate_evaluation(
    model, tokenizer, examples, generation_policy, device, dtype
) -> dict:
    """Generate from question-only prompts; never supply worked solutions."""
    import torch
    from transformers import GenerationConfig

    policy = GenerationConfig.from_dict(generation_policy)
    if policy.do_sample or policy.num_beams != 1:
        raise ValueError("Use Notebook 02's greedy one-beam evaluation policy.")
    eos = policy.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index, example in enumerate(examples, 1):
                print(
                    f"Evaluating {index}/{len(examples)}: {example['example_id']}",
                    flush=True,
                )
                inputs = tokenizer(
                    example["prompt"],
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                ).to(device)
                with precision_context(device, dtype):
                    complete = model.generate(
                        **inputs, generation_config=policy, use_cache=True
                    )
                if complete.ndim != 2 or complete.shape[0] != 1:
                    raise ValueError("Expected [1, complete_length] generated IDs.")
                prompt_ids = inputs["input_ids"][0].tolist()
                response_ids = validate_generation_ids(
                    prompt_ids, complete[0].tolist(), policy.max_new_tokens
                )
                ended = response_ids[-1] in eos_ids
                rows.append(
                    {
                        "example_id": example["example_id"],
                        "source_index": example["source_index"],
                        "question": example["question"],
                        "prompt": example["prompt"],
                        "expected_answer": example["expected_answer"],
                        "response_text": tokenizer.decode(
                            response_ids, skip_special_tokens=True
                        ),
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        "prompt_length": len(prompt_ids),
                        "response_length": len(response_ids),
                        "ended_with_eos": ended,
                        "hit_token_limit": len(response_ids) == policy.max_new_tokens
                        and not ended,
                    }
                )
    finally:
        model.train(was_training)
    annotations, scores = rescore_response_rows(rows)
    return {"rows": rows, "annotations": annotations, "scores": scores}


def evaluation_summary(evaluation: dict, stage: str, split: str) -> dict:
    scores = evaluation["scores"]
    rows = evaluation["rows"]
    return {
        "stage": stage,
        "split": split,
        "questions": len(rows),
        "strict_accuracy": scores["strict"]["accuracy"],
        "format_compliance": scores["strict"]["parse_rate"],
        "prose_numeric_accuracy": scores["prose"]["accuracy"],
        "prose_parse_rate": scores["prose"]["parse_rate"],
        "token_limit_stops": sum(r["hit_token_limit"] for r in rows),
    }


def new_run_directory(project_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(project_root) / "results" / f"sft_cold_start_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_report(
    root,
    directory,
    config,
    protocol,
    splits,
    history,
    evaluations,
    runtime,
    checkpoint_verified,
):
    """Self-contained lightweight report; model weights live in a separate folder."""
    from src.evaluation import PROSE_CHECKER_VERSION

    directory, root = Path(directory), Path(root)
    report = directory / "report"
    report.mkdir(exist_ok=False)
    write_json(report / "configuration.json", asdict(config))
    write_json(report / "protocol.json", protocol)
    write_json(report / "data_splits.json", splits)
    write_json(report / "training_history.json", history)
    for name, evaluation in evaluations.items():
        write_json(report / f"{name}.json", evaluation)
    source = report / "source"
    source.mkdir()
    for name in ("sft.py", "evaluation.py", "baseline.py"):
        shutil.copyfile(root / "src" / name, source / name)
    notebook = json.loads((root / "notebooks/03_sft_cold_start.ipynb").read_text())
    write_json(
        source / "notebook03_source.json",
        [
            {"cell_type": cell["cell_type"], "source": cell["source"]}
            for cell in notebook["cells"]
        ],
    )
    # Include the observed chart and the checkpoint's fingerprints, not its weights.
    if (directory / "observed_results.png").is_file():
        shutil.copyfile(
            directory / "observed_results.png", report / "observed_results.png"
        )
    if (directory / "checkpoint/SHA256SUMS.json").is_file():
        shutil.copyfile(
            directory / "checkpoint/SHA256SUMS.json",
            report / "checkpoint_SHA256SUMS.json",
        )
    manifest = {
        "status": "completed",
        "method": "reference_solution_sft",
        "student_model_id": protocol["student_model_id"],
        "student_revision": protocol["student_revision"],
        "teacher_loaded": False,
        "seed": config.seed,
        "checker_version": PROSE_CHECKER_VERSION,
        "runtime": runtime,
        "analysis_environment": analysis_environment(),
        "checkpoint_reload_verified": checkpoint_verified,
        "checkpoint_directory": "../checkpoint",
        "checkpoint_in_report_zip": False,
        "limitations": [
            "Small custom experiment",
            "Exploratory prose metric",
            "Numeric agreement does not verify reasoning",
            "No distillation yet",
        ],
    }
    write_json(report / "manifest.json", manifest)
    hashes = {
        str(p.relative_to(report)): sha256_file(p)
        for p in sorted(report.rglob("*"))
        if p.is_file()
    }
    write_json(report / "SHA256SUMS.json", hashes)
    archive = Path(
        shutil.make_archive(str(directory / "report"), "zip", root_dir=report)
    )
    return archive


def verify_report(directory: Path):
    directory = Path(directory)
    hashes = json.loads((directory / "SHA256SUMS.json").read_text())
    files = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    if files != set(hashes):
        raise ValueError("Report inventory differs from its checksum list.")
    for name, digest in hashes.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Report checksum mismatch: {name}")
    return json.loads((directory / "manifest.json").read_text())


def save_checkpoint(model, tokenizer, directory: Path):
    """Save final weights for further training, not mid-epoch optimizer resume."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Never overwrite an earlier checkpoint.")
    old_cache = model.config.use_cache
    try:
        model.config.use_cache = True
        model.save_pretrained(directory, safe_serialization=True)
        tokenizer.save_pretrained(directory)
    finally:
        model.config.use_cache = old_cache
    write_json(
        directory / "SHA256SUMS.json",
        {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()},
    )


def verify_checkpoint(
    model, tokenizer, directory: Path, token_ids: list[int], device, dtype
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = Path(directory)
    hashes = json.loads((directory / "SHA256SUMS.json").read_text())
    files = {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.json"
    }
    if set(hashes) != files:
        raise ValueError("Checkpoint inventory differs from its checksum list.")
    for name, digest in hashes.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    restored_tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if restored_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Reloaded tokenizer mapping differs.")
    restored = AutoModelForCausalLM.from_pretrained(
        directory, dtype=torch.float32, local_files_only=True
    ).to(device)
    model.eval()
    restored.eval()
    # Compare identical small prefixes to avoid a second long generation pass.
    inputs = torch.tensor([token_ids[:32]], device=device)
    with torch.inference_mode(), precision_context(device, dtype):
        original_logits = model(inputs, use_cache=False).logits
        restored_logits = restored(inputs, use_cache=False).logits
    if not torch.allclose(
        original_logits.float(), restored_logits.float(), rtol=1e-4, atol=1e-5
    ):
        raise ValueError("Reloaded checkpoint logits differ from the trained model.")
    del restored
    return True
