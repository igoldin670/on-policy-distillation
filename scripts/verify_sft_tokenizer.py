"""Reproduce the local-Qwen tokenizer warning without loading student weights."""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path


def verify(root: Path):
    os.environ.setdefault("HF_HOME", str(root / ".cache/source-verification"))
    sys.path.insert(0, str(root))
    from datasets import load_dataset
    from transformers import AutoTokenizer, __version__

    from src.baseline import verify_recorded_run
    from src.sft import (
        SFTConfig,
        load_checkpoint_tokenizer,
        prepare_splits,
        verify_tokenizer_roundtrip,
    )

    protocol = verify_recorded_run(root / "notebooks/assets/02_baseline_evaluation/recorded_run")[
        "manifest"
    ]["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["student_model_id"], revision=protocol["student_revision"], token=False
    )
    dataset = load_dataset(
        protocol["dataset_id"],
        protocol["dataset_config"],
        split="train",
        revision=protocol["dataset_revision"],
        token=False,
    )
    splits = prepare_splits(dataset, tokenizer, protocol, SFTConfig())
    probes = [
        text
        for e in splits["train"] + splits["validation"]
        for text in (e["prompt"], e["target_text"])
    ]
    probes += [e["prompt"] for e in protocol["examples"]]
    probes += [
        "\t  spaces\n\n",
        "£12.50 − 3 = 9.50",
        "Café 中文 🙂 １２３",
        "3 + 2 = 5.\n#### 5",
    ]
    with tempfile.TemporaryDirectory(prefix="qwen-tokenizer-check-") as folder:
        directory = Path(folder)
        tokenizer.save_pretrained(directory)
        # Only the identifying fields used by the library's regex detector.
        # This is a tokenizer-only simulation of a local Qwen3 save, not weights.
        (directory / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "transformers_version": __version__})
        )
        logger = logging.getLogger("transformers.tokenization_utils_base")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            AutoTokenizer.from_pretrained(directory, local_files_only=True)
        finally:
            logger.removeHandler(handler)
        warning_seen = "incorrect regex pattern" in stream.getvalue()
        restored = load_checkpoint_tokenizer(directory)
        verify_tokenizer_roundtrip(tokenizer, restored, probes, splits["train"][0]["input_ids"])
    return {
        "transformers": __version__,
        "model_revision": protocol["student_revision"],
        "mistral_warning_reproduced": warning_seen,
        "probe_texts": len(probes),
        "encoding_and_decoding_checks_passed": True,
        "student_weights_loaded": False,
        "actual_colab_checkpoint_checked": False,
        "scope": "Pinned tokenizer saved locally beside minimal Qwen3 identifying config; not the trained checkpoint.",
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = verify(root)
    (root / "notebooks/assets/03_sft_cold_start/tokenizer_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
