"""Notebook 02's replay validation, optional GPU generation, and analysis export.

Importing this module never imports PyTorch or downloads a model. The default
portfolio path replays a checksummed recorded experiment entirely on CPU.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.evaluation import (
    PROSE_CHECKER_VERSION,
    extract_strict_final_answer,
    load_saved_baseline,
    rescore_response_rows,
    save_rescore_report,
)

RUN_FILES = ("manifest.json", "responses.jsonl", "results.csv", "summary.json")


def sha256_file(path: Path) -> str:
    """Fingerprint bytes, rather than a path tied to one person's computer."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_recorded_run(directory: Path) -> dict:
    """Validate checksums, protocol identity, token lengths, stops, and all metrics.

    The token lists can establish stored shapes and stop flags. They do not
    independently reproduce the original CUDA forward pass or prove its logits.
    """
    directory = Path(directory)
    checksums = json.loads((directory / "SHA256SUMS.json").read_text())
    if set(checksums) != set(RUN_FILES):
        raise ValueError("Checksums must cover exactly the four original run files.")
    for name in RUN_FILES:
        if sha256_file(directory / name) != checksums[name]:
            raise ValueError(f"Recorded artifact checksum mismatch: {name}")

    rows, provenance = load_saved_baseline(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    original_summary = json.loads((directory / "summary.json").read_text())
    protocol = manifest["protocol"]
    generation = protocol["generation_config"]
    budget = generation["max_new_tokens"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("Generation token budget must be a positive integer.")
    if generation["do_sample"] is not False or generation["num_beams"] != 1:
        raise ValueError("This baseline is defined for single-beam greedy evaluation.")
    eos_ids = generation["eos_token_id"]
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    if not eos_ids:
        raise ValueError("The recorded stopping policy must define end tokens.")

    for row in rows:
        for ids_key, length_key in [
            ("prompt_token_ids", "prompt_length"),
            ("response_token_ids", "response_length"),
        ]:
            ids, length = row[ids_key], row[length_key]
            if (
                not isinstance(ids, list)
                or type(length) is not int
                or length != len(ids)
                or not ids
            ):
                raise ValueError(
                    f"Invalid recorded token shape: {row['example_id']} / {ids_key}"
                )
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Token IDs must be nonnegative integers.")
        for flag in [
            "answer_parsed",
            "is_correct",
            "hit_token_limit",
            "ended_with_eos",
        ]:
            if type(row[flag]) is not bool:
                raise ValueError(f"Expected a Boolean stop/scoring flag: {flag}")
        length = row["response_length"]
        ended = row["response_token_ids"][-1] in eos_ids
        if not 0 < length <= budget:
            raise ValueError("Response exceeded the configured token budget.")
        if row["ended_with_eos"] != ended or row["hit_token_limit"] != (
            length == budget and not ended
        ):
            raise ValueError("Stored stop flags disagree with response IDs and budget.")
        if row["answer_parsed"] != (
            extract_strict_final_answer(row["response_text"]) is not None
        ):
            raise ValueError("Stored strict parse flag is inconsistent.")
        if row["prompt"] != protocol["prompt_template"].format(
            question=row["question"]
        ):
            raise ValueError(
                "A prompt does not match the fixed question-only template."
            )
    if protocol["selected_indices"] != [row["source_index"] for row in rows]:
        raise ValueError("Selected dataset indices do not match response order.")
    indices = protocol["selected_indices"]
    if any(type(index) is not int or index < 0 for index in indices) or len(
        set(indices)
    ) != len(indices):
        raise ValueError("Selected indices must be unique, nonnegative integers.")

    annotations, scores = rescore_response_rows(rows)
    strict = scores["strict"]
    expected_summary = {
        "number_of_examples": len(rows),
        "number_correct": strict["number_correct"],
        "accuracy": strict["accuracy"],
        "number_parsed": strict["number_parsed"],
        "answer_parse_rate": strict["parse_rate"],
        "number_at_token_limit": sum(row["hit_token_limit"] for row in rows),
        "mean_response_tokens": sum(row["response_length"] for row in rows) / len(rows),
    }
    if original_summary != expected_summary:
        raise ValueError(
            "Stored summary does not match independently recomputed metrics."
        )
    with (directory / "results.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(rows):
        raise ValueError("CSV row count differs from JSONL.")
    for csv_row, row in zip(csv_rows, rows, strict=True):
        if any(
            value != ("" if row[key] is None else str(row[key]))
            for key, value in csv_row.items()
        ):
            raise ValueError("CSV and JSONL contain different records.")
    return {
        "manifest": manifest,
        "rows": rows,
        "original_summary": original_summary,
        "annotations": annotations,
        "scores": scores,
        "provenance": provenance,
        "artifact_sha256": checksums,
    }


def metric_rows(annotations: list[dict], scores: dict) -> list[dict]:
    """Report both policies with the same all-question denominator."""
    table = []
    for name, label in [
        ("strict", "Strict format (original)"),
        ("prose", "Prose (exploratory)"),
    ]:
        score = scores[name]
        table.append(
            {
                "checker": label,
                "questions": score["number_of_examples"],
                "correct": score["number_correct"],
                "parsed_wrong": score["number_parsed"] - score["number_correct"],
                "unparsed": score["number_of_examples"] - score["number_parsed"],
                "accuracy": score["accuracy"],
                "parse_rate": score["parse_rate"],
            }
        )
    for row in table:
        if row["correct"] + row["parsed_wrong"] + row["unparsed"] != len(annotations):
            raise ValueError(
                "Metric categories must partition all evaluation examples."
            )
    return table


def analysis_environment() -> dict:
    """Separate this CPU analysis environment from the recorded GPU environment."""
    versions = {}
    for name in [
        "numpy",
        "pandas",
        "matplotlib",
        "nbformat",
        "nbclient",
        "nbconvert",
        "ipykernel",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": versions,
    }


def export_analysis(project_root: Path, run: dict, *, execution_mode: str) -> Path:
    """Create a new evidence report; inputs and the old strict metric are immutable."""
    if execution_mode not in {"replay", "generate"}:
        raise ValueError("Execution mode must be replay or generate.")
    root = Path(project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "results" / f"baseline_analysis_{stamp}"
    source = {**run["provenance"], "artifact_sha256": run["artifact_sha256"]}
    archive = save_rescore_report(
        destination, run["annotations"], run["scores"], source
    )
    metadata = {
        "execution_mode": execution_mode,
        "inference_performed_this_execution": execution_mode == "generate",
        "original_run_id": run["manifest"]["run_id"],
        "recorded_inference_environment": run["manifest"]["environment"],
        "analysis_environment": analysis_environment(),
        "checker_version": PROSE_CHECKER_VERSION,
        "code_sha256": {
            name: sha256_file(root / "src" / name)
            for name in ["baseline.py", "evaluation.py", "baseline_plots.py"]
        },
        "limitations": [
            "32-question sample",
            "prose checker designed post hoc",
            "numeric agreement does not certify units or reasoning",
            "no teacher logits or training curve in these artifacts",
        ],
    }
    notebook = json.loads((root / "notebooks/02_baseline_evaluation.ipynb").read_text())
    sources = [(cell["cell_type"], cell["source"]) for cell in notebook["cells"]]
    metadata["notebook_source_sha256"] = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False).encode()
    ).hexdigest()
    (destination / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    # Include the final metadata inside the downloadable archive too.
    shutil.make_archive(str(destination), "zip", root_dir=destination)
    return archive


def validate_generation_ids(
    prompt_ids: list[int], complete_ids: list[int], budget: int
) -> list[int]:
    """Check the causal prompt/response boundary before any answer scoring."""
    if not prompt_ids or complete_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Generation must preserve the exact input prompt IDs.")
    response_ids = complete_ids[len(prompt_ids) :]
    if not 0 < len(response_ids) <= budget:
        raise ValueError("Generated response must contain 1..max_new_tokens tokens.")
    return response_ids


def generate_fresh_baseline(protocol: dict, output_directory: Path) -> Path:
    """Optional GPU-only inference using the recorded fixed protocol.

    This path is intentionally explicit and never runs in replay mode. It loads
    only the original student checkpoint. It performs no distillation or training.
    """
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-generation.txt in a CUDA-enabled runtime first."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Fresh inference requires a working CUDA GPU. Use MODE='replay' on CPU."
        )
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError("Choose a new output directory for a fresh model run.")
    policy = copy.deepcopy(protocol)
    if policy["generation_config"]["do_sample"] is not False:
        raise ValueError("This baseline requires deterministic greedy decoding.")
    seed = policy["selection_seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset(
        policy["dataset_id"],
        policy["dataset_config"],
        split=policy["dataset_split"],
        revision=policy["dataset_revision"],
    )
    policy["dataset_fingerprint"] = dataset._fingerprint
    selected = sorted(
        random.Random(seed).sample(range(len(dataset)), policy["number_of_examples"])
    )
    if selected != policy["selected_indices"]:
        raise ValueError(
            "Dataset revision or selected question set differs from the recorded protocol."
        )
    for example in policy["examples"]:
        reference = dataset[example["source_index"]]
        answer = extract_strict_final_answer(reference["answer"])
        if reference["question"] != example["question"] or answer != Decimal(
            example["expected_answer"]
        ):
            raise ValueError(
                "Pinned dataset content differs from the saved evaluation examples."
            )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        policy["student_model_id"], revision=policy["student_revision"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    student = AutoModelForCausalLM.from_pretrained(
        policy["student_model_id"],
        revision=policy["student_revision"],
        dtype=dtype,
        device_map={"": 0},
    )
    student.eval()
    if len(tokenizer) > student.config.vocab_size:
        raise ValueError("Tokenizer IDs exceed the model's vocabulary capacity.")
    generation_config = GenerationConfig.from_dict(policy["generation_config"])
    if generation_config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("The recorded padding ID differs from the pinned tokenizer.")
    # Keep the actual runtime configuration, including its library version.
    policy["generation_config"] = generation_config.to_dict()
    eos = generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos_ids:
        raise ValueError("The generation policy must define end tokens.")
    rows = []
    for example in policy["examples"]:
        # No reference answer enters the input. Batch size is always one.
        prompt = policy["prompt_template"].format(question=example["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Expected input_ids shape [1, prompt_length].")
        with torch.inference_mode():
            sequence = student.generate(**inputs, generation_config=generation_config)
        if sequence.ndim != 2 or sequence.shape[0] != 1:
            raise ValueError("Expected generated IDs shape [1, complete_length].")
        prompt_ids = inputs["input_ids"][0].cpu().tolist()
        response_ids = validate_generation_ids(
            prompt_ids, sequence[0].cpu().tolist(), generation_config.max_new_tokens
        )
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        predicted = extract_strict_final_answer(response)
        ended = response_ids[-1] in eos_ids
        rows.append(
            {
                **example,
                "response_text": response,
                "predicted_answer": str(predicted) if predicted is not None else None,
                "answer_parsed": predicted is not None,
                "is_correct": predicted is not None
                and predicted == Decimal(example["expected_answer"]),
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "ended_with_eos": ended,
                "hit_token_limit": len(response_ids) == generation_config.max_new_tokens
                and not ended,
            }
        )
    if any(parameter.grad is not None for parameter in student.parameters()):
        raise RuntimeError("Evaluation must not create parameter gradients.")
    _, scores = rescore_response_rows(rows)
    strict = scores["strict"]
    summary = {
        "number_of_examples": len(rows),
        "number_correct": strict["number_correct"],
        "accuracy": strict["accuracy"],
        "number_parsed": strict["number_parsed"],
        "answer_parse_rate": strict["parse_rate"],
        "number_at_token_limit": sum(row["hit_token_limit"] for row in rows),
        "mean_response_tokens": math.fsum(row["response_length"] for row in rows)
        / len(rows),
    }
    manifest = {
        "run_id": output.name,
        "status": "completed",
        "protocol": policy,
        "protocol_sha256": hashlib.sha256(
            json.dumps(policy, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "torch",
                    "transformers",
                    "datasets",
                    "accelerate",
                    "huggingface_hub",
                ]
            },
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "model_dtype": str(dtype),
            "seed": seed,
            "attention_implementation": student.config._attn_implementation,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "responses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            key
            for key in rows[0]
            if key not in {"prompt_token_ids", "response_token_ids"}
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    checksums = {name: sha256_file(output / name) for name in RUN_FILES}
    (output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2) + "\n")
    verify_recorded_run(output)
    return output
