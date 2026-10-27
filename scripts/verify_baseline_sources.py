"""Optional network check of the pinned tokenizer and GSM8K content, without weights.

Install transformers/datasets/huggingface_hub from requirements-generation.txt.
The default notebook replay needs none of these downloads.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def verify_sources(root: Path) -> dict:
    # Explicit task-local caching avoids writes to user-wide credential/cache locations.
    os.environ["HF_HOME"] = str(root / ".cache/source-verification")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from datasets import disable_progress_bars, load_dataset
    from transformers import AutoTokenizer

    from src.baseline import verify_recorded_run
    from src.evaluation import extract_strict_final_answer

    disable_progress_bars()
    assets = root / "notebooks/assets/02_baseline_evaluation"
    run = verify_recorded_run(assets / "recorded_run")
    protocol = run["manifest"]["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["student_model_id"],
        revision=protocol["student_revision"],
        token=False,
    )
    for row in run["rows"]:
        if tokenizer(row["prompt"])["input_ids"] != row["prompt_token_ids"]:
            raise ValueError(
                "Pinned tokenizer does not reproduce prompt IDs: " + row["example_id"]
            )
        if (
            tokenizer.decode(row["response_token_ids"], skip_special_tokens=True)
            != row["response_text"]
        ):
            raise ValueError(
                "Pinned tokenizer does not reproduce response text: "
                + row["example_id"]
            )
    print(
        "Pinned tokenizer verified: all 32 prompts and response-ID decodes.", flush=True
    )
    dataset = load_dataset(
        protocol["dataset_id"],
        protocol["dataset_config"],
        split=protocol["dataset_split"],
        revision=protocol["dataset_revision"],
        token=False,
    )
    selection = sorted(
        random.Random(protocol["selection_seed"]).sample(
            range(len(dataset)), protocol["number_of_examples"]
        )
    )
    if selection != protocol["selected_indices"]:
        raise ValueError("Selected dataset indices are not reproducible.")
    for row in run["rows"]:
        reference = dataset[row["source_index"]]
        if reference["question"] != row["question"]:
            raise ValueError(
                "Question differs from the pinned dataset: " + row["example_id"]
            )
        if extract_strict_final_answer(reference["answer"]) != Decimal(
            row["expected_answer"]
        ):
            raise ValueError(
                "Reference number differs from the pinned dataset: " + row["example_id"]
            )
    print(
        "Pinned dataset verified: selected question text and all reference numbers.",
        flush=True,
    )
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": protocol["student_revision"],
        "dataset_revision": protocol["dataset_revision"],
        "prompt_tokenizations_checked": len(run["rows"]),
        "response_decodes_checked": len(run["rows"]),
        "reference_answers_checked": len(run["rows"]),
        "dataset_test_rows": len(dataset),
        "selection_reproduced": True,
        "model_weights_loaded": False,
        "fresh_inference": False,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["transformers", "datasets", "huggingface_hub", "tokenizers"]
        },
        "verification_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (assets / "source_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    print(json.dumps(verify_sources(Path(__file__).resolve().parents[1]), indent=2))
