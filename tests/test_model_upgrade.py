"""New-family checks use synthetic logits and tiny random models, never Qwen weights."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.opd_loss import forward_kl_loss
from src.qwen38.model_upgrade import (
    generation_policy,
    load_text_weights,
    load_upgrade_profile,
    text_config_values,
    upgrade_cache_preflight,
    upgrade_teacher_prompt_ids,
    verify_upgrade_tokenizers,
)

ROOT = Path(__file__).resolve().parents[1]


class ModelUpgradeTests(unittest.TestCase):
    def test_cache_budget_counts_only_the_pinned_revision(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        profile = load_upgrade_profile(ROOT)
        profile["teacher_weight_bytes"] = 12 * 2**30
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "models--Qwen--Qwen3.8-27B"
            unrelated = repository / "snapshots/old"
            unrelated.mkdir(parents=True)
            (unrelated / "model.safetensors").write_bytes(b"old")
            with (
                patch.dict("os.environ", {"HF_HUB_CACHE": temporary}),
                patch(
                    "src.qwen38.model_upgrade.shutil.disk_usage",
                    return_value=SimpleNamespace(free=19 * 2**30),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "20.0 GiB"):
                    upgrade_cache_preflight(profile)
                pinned = repository / "snapshots" / profile["teacher_revision"]
                pinned.mkdir()
                # A sparse fixture represents one completed cached shard.
                with (pinned / "model.safetensors").open("wb") as stream:
                    stream.truncate(2 * 2**30)
                result = upgrade_cache_preflight(profile)
                self.assertEqual(result["estimated_additional_gib"], 18)
                self.assertFalse(result["personal_quota_verified"])

    def test_teacher_template_explicitly_requests_flat_ids(self):
        from unittest.mock import Mock

        tokenizer = Mock(chat_template="pinned", apply_chat_template=Mock(return_value=[3, 4, 5]))
        self.assertEqual(upgrade_teacher_prompt_ids(tokenizer, "Question?"), [3, 4, 5])
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["return_dict"])
        tokenizer.apply_chat_template.return_value = {"input_ids": [3, 4, 5]}
        with self.assertRaisesRegex(ValueError, "flat list"):
            upgrade_teacher_prompt_ids(tokenizer, "Question?")

    def test_pinned_pair_and_new_stopping_ids(self):
        profile = load_upgrade_profile(ROOT)
        self.assertEqual(profile["student_model_id"], "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(profile["teacher_model_id"], "Qwen/Qwen3.8-27B")
        self.assertEqual(profile["eos_token_ids"], [248044, 248046])
        self.assertEqual(profile["tokenizer_vocab_size"], 248077)

    def test_old_checkpoint_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "Qwen3 checkpoint"):
            text_config_values({"model_type": "qwen3", "vocab_size": 151936})

    def test_text_config_extracted_without_vision(self):
        text = {"model_type": "qwen3_5_text", "vocab_size": 64}
        self.assertEqual(text_config_values({"text_config": text}), text)
        self.assertEqual(text_config_values(text), text)

    def test_policy_preserves_unfiltered_sampling(self):
        profile = load_upgrade_profile(ROOT)
        policy = generation_policy(profile, 248044)
        self.assertEqual(policy.eos_token_id, [248044, 248046])
        self.assertIsNone(policy.suppress_tokens)
        self.assertEqual(policy.top_k, 0)
        self.assertEqual(policy.temperature, 1.0)

    def test_grouped_loss_still_preserves_original_eos_ids(self):
        student = torch.tensor([[[0.0, 4.0, -4.0, 1.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, -4.0, 4.0, 1.0]]])
        grouped = forward_kl_loss(
            student, teacher, teacher_temperature=1.0, recompute=False, eos_token_ids=[1, 2]
        )
        raw = forward_kl_loss(student, teacher, teacher_temperature=1.0, recompute=False)
        self.assertAlmostEqual(grouped.item(), 0.0, places=5)
        self.assertGreater(raw.item(), 1.0)
        grouped.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_tokenizer_meaning_mismatch_rejected(self):
        from types import SimpleNamespace

        profile = {"tokenizer_vocab_size": 3, "output_vocab_size": 4, "eos_token_ids": [1, 2]}
        student = SimpleNamespace(get_vocab=lambda: {"a": 0, "end": 1, "stop": 2})
        teacher = SimpleNamespace(get_vocab=lambda: {"a": 1, "end": 0, "stop": 2})
        with self.assertRaisesRegex(ValueError, "different meanings"):
            verify_upgrade_tokenizers(student, teacher, {}, {}, profile, [])


class NewArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError:
            raise unittest.SkipTest("Run the new-architecture suite with requirements-qwen38.txt")
        cls.config_type = Qwen3_5Config
        cls.model_type = Qwen3_5ForConditionalGeneration

    def test_multimodal_checkpoint_text_reload_and_gradients(self):
        # Tiny random text+vision weights verify the exact conversion performed
        # when text-only models load the official multimodal repository format.
        from src.sft import collate_examples, response_loss_sum

        config = self.config_type(
            text_config={
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "max_position_embeddings": 128,
                "eos_token_id": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 2],
                },
                "tie_word_embeddings": False,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 16,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [],
            },
        )
        # The base student ties its head to embeddings; the teacher does not.
        # Both must load published multimodal keys without initializing new rows.
        for tied in (False, True):
            with self.subTest(tie_word_embeddings=tied):
                config.tie_word_embeddings = tied
                config.text_config.tie_word_embeddings = tied
                torch.manual_seed(42)
                full = self.model_type(config).eval()
                ids = torch.tensor([[3, 4, 5, 2]])
                with torch.no_grad():
                    expected = full(
                        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                    ).logits
                with tempfile.TemporaryDirectory() as temporary:
                    full.save_pretrained(temporary)
                    restored = load_text_weights(
                        temporary,
                        config=config,
                        dtype=torch.float32,
                        device_map={"": "cpu"},
                    ).eval()
                    with torch.no_grad():
                        actual = restored(
                            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                        ).logits
                    self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
                    # Hybrid cached decoding must preserve the prompt and agree with
                    # full-sequence scores at the first generated-token boundary.
                    from transformers import GenerationConfig

                    from src.logprobs import response_prediction_logits

                    generation = GenerationConfig(
                        max_new_tokens=3,
                        do_sample=False,
                        num_beams=1,
                        eos_token_id=2,
                        pad_token_id=2,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_logits=True,
                    )
                    with torch.no_grad():
                        generated = restored.generate(
                            input_ids=ids[:, :2],
                            attention_mask=torch.ones_like(ids[:, :2]),
                            generation_config=generation,
                        )
                        full_scores = restored(
                            input_ids=generated.sequences, use_cache=False
                        ).logits
                    self.assertTrue(torch.equal(generated.sequences[:, :2], ids[:, :2]))
                    aligned = response_prediction_logits(full_scores, generated.sequences, 2)
                    self.assertTrue(
                        torch.allclose(aligned[:, 0, :], generated.logits[0], atol=1e-5, rtol=1e-5)
                    )
                    batch = collate_examples(
                        [
                            {
                                "input_ids": ids[0].tolist(),
                                "labels": [-100, -100, 5, 2],
                            }
                        ],
                        pad_token_id=2,
                    )
                    restored.train()
                    restored.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    loss, targets = response_loss_sum(restored, batch)
                    self.assertEqual(targets, 2)
                    (loss / targets).backward()
                    self.assertTrue(
                        all(
                            torch.isfinite(p.grad).all()
                            for p in restored.parameters()
                            if p.grad is not None
                        )
                    )
                    self.assertTrue(
                        any(
                            p.grad is not None and p.grad.abs().sum() > 0
                            for p in restored.parameters()
                        )
                    )
                    # Saving a TEXT-only checkpoint must reload through the same loader.
                    text_directory = Path(temporary) / "text_checkpoint"
                    restored.eval().save_pretrained(text_directory)
                    again = load_text_weights(
                        text_directory,
                        config=copy.deepcopy(restored.config),
                        dtype=torch.float32,
                        device_map={"": "cpu"},
                    ).eval()
                    with torch.no_grad():
                        self.assertTrue(
                            torch.allclose(
                                again(input_ids=ids, use_cache=False).logits,
                                actual,
                                atol=1e-6,
                                rtol=1e-5,
                            )
                        )
                    # Exercise the full new scoring path with tiny CPU models.
                    # Only CUDA autocast is replaced; real pretrained weights
                    # and actual GPU execution remain a separate qualification.
                    from contextlib import nullcontext
                    from types import SimpleNamespace
                    from unittest.mock import Mock, patch

                    from src.qwen38.model_upgrade import score_upgrade_trajectory

                    frozen_teacher = copy.deepcopy(again).requires_grad_(False).eval()
                    frozen_teacher.zero_grad(set_to_none=True)
                    tokenizer = SimpleNamespace(
                        encode=lambda *args, **kwargs: [3, 4],
                        decode=lambda tokens, **kwargs: repr(tokens),
                        pad_token_id=2,
                    )
                    teacher_tokenizer = Mock(
                        chat_template="pinned", apply_chat_template=Mock(return_value=[3, 4, 5])
                    )
                    with patch("torch.autocast", side_effect=lambda *args, **kwargs: nullcontext()):
                        result = score_upgrade_trajectory(
                            again,
                            frozen_teacher,
                            {"student": tokenizer, "teacher": teacher_tokenizer},
                            {
                                "tokenizer_vocab_size": 64,
                                "output_vocab_size": 64,
                                "eos_token_ids": [1, 2],
                            },
                            "Question?",
                            max_new_tokens=3,
                        )
                    self.assertEqual(result["status"], "new_model_scoring_passed")
                    self.assertEqual(result["student_predicting_first_response_position"], 1)
                    self.assertEqual(result["teacher_predicting_first_response_position"], 2)
                    self.assertEqual(result["aligned_logits_shape"][1], result["response_length"])
                    self.assertFalse(result["weights_updated"])
                    self.assertTrue(all(p.grad is None for p in frozen_teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
