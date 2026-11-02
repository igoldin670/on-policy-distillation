"""CPU tests use a tiny RANDOM model, never the pretrained Qwen student."""

import copy
import io
import json
import tempfile
import time
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
except ImportError as error:
    raise unittest.SkipTest(
        "SFT CPU tests need PyTorch and requirements-generation.txt"
    ) from error

from src.sft import (
    SFTConfig,
    backward_group,
    clean_solution,
    collate_examples,
    encode_example,
    generate_evaluation,
    normalize_question,
    prepare_splits,
    response_loss_sum,
    save_checkpoint,
    save_report,
    training_groups,
    validation_loss,
    verify_checkpoint,
    verify_report,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 2 for c in text]}


def tiny_model():
    torch.manual_seed(42)
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            attention_dropout=0.0,
            use_cache=False,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
    )


def sample(ids=(2, 3, 4, 1), prompt_length=2):
    return {
        "input_ids": list(ids),
        "labels": [-100] * prompt_length + list(ids[prompt_length:]),
    }


class SFTDataTests(unittest.TestCase):
    def test_reference_cleanup_preserves_reasoning_and_strict_answer(self):
        target, number = clean_solution("2+3=<<2+3=5>>5.\n#### 5")
        self.assertEqual(target, "2+3=5.\n#### 5")
        self.assertEqual(number, "5")
        with self.assertRaises(ValueError):
            clean_solution("The answer is 5.")

    def test_prompt_mask_boundary_and_real_eos(self):
        encoded = encode_example(
            CharacterTokenizer(),
            "Q",
            "Reason.\n#### 5",
            {"prompt_template": "Question: {question}\nAnswer:\n"},
            100,
        )
        p = encoded["prompt_length"]
        self.assertEqual(encoded["labels"][:p], [-100] * p)
        self.assertEqual(encoded["labels"][p:], encoded["input_ids"][p:])
        self.assertEqual(encoded["labels"][-1], 1)
        self.assertEqual(encoded["labels"][1:][p - 1], encoded["input_ids"][p])

    def test_overlong_example_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "overlong_sequence"):
            encode_example(
                CharacterTokenizer(),
                "Q",
                "Reason.\n#### 5",
                {"prompt_template": "{question}\n"},
                3,
            )

    def test_unstable_bpe_boundary_is_rejected(self):
        class Unstable(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [9] if text.endswith("Answer:\n") else [10, 11]}

        with self.assertRaisesRegex(ValueError, "unstable_token_boundary"):
            encode_example(
                Unstable(),
                "Q",
                "#### 5",
                {"prompt_template": "{question}Answer:\n"},
                100,
            )

    def test_split_exclusion_deduplication_and_reproducibility(self):
        dataset = [
            {"question": " Q  ONE ", "answer": "#### 1"},
            {"question": "q one", "answer": "#### 1"},
            {"question": "ＨＥＬＤ  out", "answer": "#### 2"},
        ]
        dataset += [{"question": f"Unique {i}", "answer": "#### 3"} for i in range(10)]
        protocol = {
            "prompt_template": "{question}\n",
            "examples": [{"question": "held out"}],
        }
        config = SFTConfig(
            train_examples=5, validation_examples=5, max_sequence_length=100
        )
        splits = prepare_splits(dataset, CharacterTokenizer(), protocol, config)
        self.assertEqual(
            splits, prepare_splits(dataset, CharacterTokenizer(), protocol, config)
        )
        train = {normalize_question(e["question"]) for e in splits["train"]}
        validation = {normalize_question(e["question"]) for e in splits["validation"]}
        self.assertFalse(train & validation)
        self.assertNotIn("held out", train | validation)
        # Requesting more than the eligible pool must fail explicitly.
        with self.assertRaisesRegex(ValueError, "eligible examples"):
            prepare_splits(
                dataset,
                CharacterTokenizer(),
                protocol,
                SFTConfig(
                    train_examples=20, validation_examples=1, max_sequence_length=100
                ),
            )

    def test_padding_does_not_mask_real_end_tokens(self):
        batch = collate_examples([sample(), sample((2, 4, 1), 1)], pad_token_id=1)
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"][1].tolist(), [-100, 4, 1, -100])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 0])


class SFTTensorTests(unittest.TestCase):
    def test_actual_notebook_loop_runs_on_cpu_with_a_tiny_model(self):
        # Execute the actual notebook loop. Only CUDA housekeeping is mocked;
        # the forward/backward passes, optimizer, scheduler, and validation run.
        import math

        from transformers import get_linear_schedule_with_warmup

        notebook = json.loads((ROOT / "notebooks/03_sft_cold_start.ipynb").read_text())
        source = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["id"] == "sft-train-loop"
        )
        model = tiny_model()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        examples = [sample(), sample((2, 3, 4, 5, 1)), sample((2, 6, 1), 1)]
        with tempfile.TemporaryDirectory() as folder:
            namespace = {
                "MODE": "train",
                "torch": torch,
                "math": math,
                "time": time,
                "config": SFTConfig(
                    train_examples=3,
                    validation_examples=1,
                    accumulation_steps=2,
                    learning_rate=1e-3,
                ),
                "student": model,
                "device": "cpu",
                "compute_dtype": None,
                "train_examples": examples,
                "validation_examples": examples[:1],
                "tokenizer": CharacterTokenizer(),
                "run_directory": Path(folder),
                "history": [],
                "backward_group": backward_group,
                "training_groups": training_groups,
                "validation_loss": validation_loss,
                "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
                "write_json": write_json,
            }
            namespace["tokenizer"].pad_token_id = 1
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.empty_cache"),
                redirect_stdout(io.StringIO()),
            ):
                exec(compile(source, "notebook03_training_cell", "exec"), namespace)
            self.assertEqual(namespace["step"], 2)
            self.assertEqual(
                len(json.loads((Path(folder) / "training_history.json").read_text())), 2
            )
            self.assertTrue(
                any(
                    not torch.equal(p, original[name])
                    for name, p in model.named_parameters()
                )
            )

    def test_generation_uses_only_prompt_not_reference_solution(self):
        from tokenizers.pre_tokenizers import Whitespace

        backend = Tokenizer(
            WordLevel(
                {"<unk>": 0, "<eos>": 1, "question": 2, "reference": 3},
                unk_token="<unk>",
            )
        )
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        model = tiny_model().train()
        model.generation_config.use_cache = True
        example = {
            "example_id": "unit/0",
            "source_index": 0,
            "question": "question",
            "prompt": "question",
            "target_text": "reference",
            "expected_answer": "3",
        }
        with (
            patch.object(model, "generate", wraps=model.generate) as generate,
            redirect_stdout(io.StringIO()),
        ):
            evaluation = generate_evaluation(
                model,
                tokenizer,
                [example],
                {
                    "max_new_tokens": 2,
                    "do_sample": False,
                    "num_beams": 1,
                    "pad_token_id": 1,
                    "eos_token_id": 1,
                    "bos_token_id": 0,
                },
                "cpu",
                None,
            )
        self.assertEqual(generate.call_args.kwargs["input_ids"].tolist(), [[2]])
        self.assertEqual(evaluation["rows"][0]["prompt_token_ids"], [2])
        self.assertTrue(model.training)

    def test_response_loss_matches_huggingface_internal_shift(self):
        model = tiny_model().eval()
        batch = collate_examples([sample(), sample((2, 3, 5, 6, 1))], 1)
        loss_sum, count = response_loss_sum(model, batch)
        reference = model(**batch).loss
        self.assertTrue(torch.allclose(loss_sum / count, reference, atol=1e-6))
        self.assertEqual(count, 5)
        # Changing ignored prompt labels must not turn them into supervised tokens.
        self.assertEqual(int((batch["labels"][:, 1:] != -100).sum()), count)

    def test_accumulated_gradients_match_single_batch_with_unequal_lengths(self):
        accumulated = tiny_model().train()
        combined = copy.deepcopy(accumulated)
        examples = [sample(), sample((2, 3, 4, 5, 6, 1))]
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metrics = backward_group(
            accumulated, [[examples[0]], [examples[1]]], 1, "cpu", None, scaler
        )
        loss_sum, count = response_loss_sum(combined, collate_examples(examples, 1))
        (loss_sum / count).backward()
        self.assertEqual(metrics["response_tokens"], count)
        for left, right in zip(
            accumulated.parameters(), combined.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=2e-6, rtol=2e-4))

    def test_partial_group_and_optimizer_update(self):
        config = SFTConfig(
            train_examples=3, validation_examples=1, accumulation_steps=2
        )
        examples = [sample(), sample((2, 4, 1), 1), sample((2, 5, 1), 1)]
        groups = list(training_groups(examples, config, 0))
        self.assertEqual([len(group) for group in groups], [2, 1])
        model = tiny_model().train()
        original = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            metrics = backward_group(model, group, 1, "cpu", None, scaler)
            self.assertGreater(metrics["loss"], 0)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            self.assertTrue(torch.isfinite(norm))
            scaler.step(optimizer)
            scaler.update()
        self.assertTrue(
            any(
                not torch.equal(p, original[name])
                for name, p in model.named_parameters()
            )
        )
        score = validation_loss(model, examples, 1, "cpu", None)
        self.assertGreater(score, 0)
        self.assertTrue(model.training)

    def test_no_response_targets_fail(self):
        with self.assertRaises(ValueError):
            collate_examples([{"input_ids": [2, 3], "labels": [-100, -100]}], 1)

    def test_checkpoint_reload_and_tamper_detection(self):
        backend = Tokenizer(
            WordLevel({"<unk>": 0, "<eos>": 1, "hi": 2}, unk_token="<unk>")
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            eos_token="<eos>",
            pad_token="<eos>",
        )
        model = tiny_model()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint"
            save_checkpoint(model, tokenizer, checkpoint)
            self.assertTrue(
                verify_checkpoint(model, tokenizer, checkpoint, [2, 2, 1], "cpu", None)
            )
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, tokenizer, checkpoint)
            (checkpoint / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_checkpoint(model, tokenizer, checkpoint, [2], "cpu", None)

    def test_report_zip_is_self_contained_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            archive = save_report(
                ROOT,
                directory,
                SFTConfig(),
                {
                    "student_model_id": "test-only/random-qwen3",
                    "student_revision": "none",
                },
                {"train": [], "validation": []},
                [],
                {},
                {"kind": "unit_test_only"},
                True,
            )
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                restored = directory / "restored"
                bundle.extractall(restored)
            manifest = verify_report(restored)
            self.assertEqual(manifest["runtime"]["kind"], "unit_test_only")
            self.assertFalse(manifest["checkpoint_in_report_zip"])
            self.assertTrue((restored / "source/sft.py").is_file())
            (restored / "training_history.json").write_text("[1]")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_report(restored)


if __name__ == "__main__":
    unittest.main()
