"""The public pipeline is exercised with tiny random hybrid models on CPU."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_opd as fixtures
import torch

from src import pipeline
from src.opd import OPDConfig
from src.profiles import load_profile_student
from src.sft import SFTConfig
from src.utils import verify_inventory


class PipelineContracts(unittest.TestCase):
    def test_gpu_qualification_requires_matching_completed_evidence(self):
        from src.qwen38.qualification import require_qualification

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("src.qwen38.qualification.qualification_fingerprint", return_value="expected"),
        ):
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Notebook 05"):
                require_qualification(root, {})
            folder = root / "results/qwen38_qualification_test"
            folder.mkdir(parents=True)
            report = folder / "qualification.json"
            report.write_text(
                json.dumps(
                    {
                        "status": "disposable_gpu_qualification_passed",
                        "qualification_fingerprint": "different",
                    }
                )
            )
            with self.assertRaises(RuntimeError):
                require_qualification(root, {})
            report.write_text(
                json.dumps(
                    {
                        "status": "disposable_gpu_qualification_passed",
                        "qualification_fingerprint": "expected",
                    }
                )
            )
            self.assertEqual(require_qualification(root, {}), report)

    def test_gpu_qualification_does_not_claim_cpu_execution(self):
        from src.qwen38.qualification import qualify_gpu

        with self.assertRaisesRegex(RuntimeError, "allocated CUDA"):
            qualify_gpu(fixtures.tiny_model(), None, {}, {}, fixtures.ROOT)

    def test_execution_controls_and_markdown_do_not_change_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notebooks").mkdir()
            for name in ("03_sft_cold_start", "04_on_policy_distillation"):
                (root / "notebooks" / f"{name}.ipynb").write_text(
                    (fixtures.ROOT / "notebooks" / f"{name}.ipynb").read_text()
                )
            before = pipeline.source_hashes(root)
            path = root / "notebooks/04_on_policy_distillation.ipynb"
            notebook = json.loads(path.read_text())
            for cell in notebook["cells"]:
                if cell["id"] == "opd-setup-runtime":
                    cell["source"] = [
                        "RUN_STAGE = 'full'\nRESUME_RUN_DIRECTORY = '/new/location'\n"
                    ]
                elif cell["cell_type"] == "markdown":
                    cell["source"].append("\nAn added explanation.\n")
            path.write_text(json.dumps(notebook))
            self.assertEqual(pipeline.source_hashes(root), before)
            next(cell for cell in notebook["cells"] if cell["id"] == "opd-run-on-policy")[
                "source"
            ].append("\nchanged_method = True\n")
            path.write_text(json.dumps(notebook))
            self.assertNotEqual(pipeline.source_hashes(root), before)

    def test_unmapped_sample_is_preserved_before_failure(self):
        state = {"tokenizer": fixtures.tiny_tokenizer(), "profile": {"tokenizer_vocab_size": 6}}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                pipeline, "sample_trajectory", return_value=fixtures.trajectory(response=(31,))
            ),
        ):
            with self.assertRaisesRegex(ValueError, "unmapped"):
                pipeline.profiled_sample(state, None, fixtures.example(), directory=directory)
            saved = list(Path(directory).glob("invalid_trajectory_*.json"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text())["response_token_ids"], [31])

    def test_reference_control_budget_preserves_complete_answers(self):
        from test_sft import CharacterTokenizer

        from src.data import prepare_opd_splits

        class EncodableTokenizer(CharacterTokenizer):
            def encode(self, text, **kwargs):
                return self(text, **kwargs)["input_ids"]

        result = prepare_opd_splits(
            [
                {"question": "short", "answer": "#### 5"},
                {"question": "long", "answer": "A lengthy explanation.\n#### 5"},
            ],
            [{"question": "reserved", "answer": "#### 5"}],
            EncodableTokenizer(),
            {"prompt_template": "Question {question}\nAnswer:\n", "examples": []},
            {"train": [], "validation": []},
            train_examples=1,
            test_examples=1,
            max_response_length=8,
        )
        self.assertEqual(result["train"][0]["question"], "short")
        self.assertIn(
            "overlong_reference_response", [row["reason"] for row in result["exclusions"]]
        )

    def test_preparation_keeps_sft_evidence_immutable(self):
        cold = {"train": [fixtures.example(0)], "validation": [fixtures.example(1)]}
        original = copy.deepcopy(cold)
        state = {
            "protocol": {"dataset_id": "toy", "dataset_config": "toy", "dataset_revision": "pin"},
            "cold_splits": cold,
            "source_train": [],
            "tokenizer": fixtures.tiny_tokenizer(),
            "tokenizers": {"teacher": fixtures.tiny_tokenizer()},
        }

        def prepared(train, test, tokenizer, protocol, cold_splits, **kwargs):
            return {"train": [], "validation": cold_splits["validation"], "primary_test": []}

        with (
            patch("datasets.load_dataset", return_value=[]),
            patch.object(pipeline, "prepare_opd_splits", side_effect=prepared),
            patch.object(pipeline, "teacher_prompt_token_ids", return_value=[3, 2]),
        ):
            prepared_state = pipeline.prepare_distillation(state)
        self.assertEqual(cold, original)
        self.assertEqual(
            prepared_state["splits"]["validation"][0]["teacher_prompt_token_ids"], [3, 2]
        )

    def test_final_optimizer_boundary_can_finish_interrupted_publication(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            directory = Path(folder)
            namespace = fixtures.StagedTrainingTests().namespace(
                directory, fixtures.tiny_config(train_examples=2, accumulation_steps=1)
            )
            verify = namespace["verify_checkpoint"]
            namespace["verify_checkpoint"] = lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("interrupted final reload")
            )
            with self.assertRaises(OSError):
                namespace["train_arm"]("sft_control", recover_pause_validation=True)
            namespace["verify_checkpoint"] = verify
            result = namespace["train_arm"]("sft_control", recover_pause_validation=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_steps"], 2)

    def test_source_fingerprints_cover_nested_model_helpers(self):
        hashes = pipeline.source_hashes(fixtures.ROOT)
        self.assertIn("src/qwen38/model_upgrade.py", hashes)
        self.assertIn("requirements-qwen38.txt", hashes)
        self.assertIn("notebooks/03_sft_cold_start.ipynb", hashes)

    def test_checkpoint_profile_mismatch_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            fixtures.save_checkpoint(fixtures.tiny_model(), fixtures.tiny_tokenizer(), path)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_profile_student(
                    {"model_type": "qwen3_5_text", "output_vocab_size": 32}, "cpu", path
                )

    def test_full_stage_requires_existing_qualified_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "qualified pilot"):
                pipeline._run_distillation({}, stage="full", run_directory=directory)


class HybridPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        except ImportError:
            raise unittest.SkipTest("The hybrid integration runs in requirements-qwen38.txt")
        cls.model_type, cls.config_type = Qwen3_5ForCausalLM, Qwen3_5TextConfig

    def model(self, seed=42):
        config = self.config_type(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            max_position_embeddings=128,
            eos_token_id=1,
            pad_token_id=1,
            tie_word_embeddings=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return self.model_type(config)

    def test_sft_two_pilots_full_controls_export_and_reopen(self):
        from safetensors.torch import load_file

        tokenizer = fixtures.tiny_tokenizer()
        profile = {
            "profile": "qwen38",
            "student_model_id": "tiny-random-student",
            "student_revision": "a" * 40,
            "teacher_model_id": "tiny-random-teacher",
            "teacher_revision": "b" * 40,
            "model_type": "qwen3_5_text",
            "output_vocab_size": 32,
            "tokenizer_vocab_size": 32,
            "eos_token_ids": [1, 5],
        }
        protocol = {
            "student_model_id": profile["student_model_id"],
            "student_revision": profile["student_revision"],
            "model_profile": "qwen38",
            "prompt_template": "{question}",
            "examples": [],
            "generation_config": {
                "max_new_tokens": 3,
                "do_sample": False,
                "num_beams": 1,
                "eos_token_id": [1, 5],
                "pad_token_id": 1,
                "bos_token_id": None,
            },
        }
        cold = {
            "train": [fixtures.example(i) for i in range(4)],
            "validation": [fixtures.example(8)],
        }
        for row in cold["train"] + cold["validation"]:
            row["teacher_prompt_token_ids"] = [3, 2]
        teacher = self.model(77).requires_grad_(False).eval()

        def factory(selected_profile, device, checkpoint=None):
            if checkpoint is not None:
                return load_profile_student(selected_profile, device, checkpoint)
            model = self.model()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            return model

        # Keep this synthetic qualification independent of whether a random LM
        # happens to stop. Actual cached generation is covered by the model suite.
        def sampled(model, tokenizer, example, **settings):
            return fixtures.trajectory(
                example["source_index"], policy_step=settings["policy_step"]
            ) | {"teacher_prompt_token_ids": example["teacher_prompt_token_ids"]}

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            state = dict(
                root=fixtures.ROOT,
                profile=profile,
                protocol=protocol,
                tokenizer=tokenizer,
                cold_splits=cold,
                sft_config=SFTConfig(train_examples=4, validation_examples=1, accumulation_steps=2),
                model_configs={},
            )
            with (
                patch.object(pipeline, "_runtime", return_value=(torch.device("cpu"), None)),
                patch.object(pipeline, "load_profile_student", side_effect=factory),
                patch.object(pipeline, "_load_teacher", return_value=teacher),
                patch.object(pipeline, "sample_trajectory", side_effect=sampled),
            ):
                sft_run = pipeline.run_cold_start(state, root / "sft")
                self.assertEqual(sft_run["result"]["completed_steps"], 2)
                manifest = json.loads((Path(sft_run["report"]) / "manifest.json").read_text())
                self.assertEqual(manifest["final_test_status"], "deferred")
                self.assertEqual(manifest["runtime"]["kind"], "tiny_random_cpu_test_only")
                verify_inventory(Path(sft_run["report"]))
                before = load_file(Path(sft_run["checkpoint"]) / "model.safetensors")
                state["splits"] = {
                    "train": [fixtures.example(i) for i in range(10, 14)],
                    "validation": cold["validation"],
                    "primary_test": [fixtures.example(19)],
                    "historical_test": [],
                }
                for rows in state["splits"].values():
                    for row in rows:
                        row["teacher_prompt_token_ids"] = [3, 2]
                config = OPDConfig(
                    train_examples=4,
                    accumulation_steps=2,
                    pilot_updates=1,
                    pilot_check_interval=1,
                    validation_interval=1,
                    checkpoint_interval=1,
                    max_prompt_length=8,
                    max_response_length=3,
                    max_sequence_length=16,
                    diagnostic_examples=1,
                    eos_loss_mode="semantic_stop",
                    eos_token_ids=(1, 5),
                    teacher_context="qwen3_5_nonthinking_v1",
                )
                run = root / "gkd"
                pilot = pipeline.run_distillation(
                    state, sft_run_directory=sft_run["directory"], run_directory=run, config=config
                )
                self.assertEqual(pilot["status"], "qualified_pilot")
                self.assertFalse(list(run.glob("primary_*.json")))
                full = pipeline.run_distillation(
                    state,
                    sft_run_directory=sft_run["directory"],
                    run_directory=run,
                    stage="full",
                    config=config,
                )
                self.assertEqual(full["status"], "completed")
                self.assertEqual(
                    set(full["metrics"]),
                    {"original_student", "starting_sft", "on_policy", "supervised_kd", "teacher"},
                )
                after = load_file(run / "supervised_kd/checkpoints/supervised_kd/model.safetensors")
                self.assertTrue(
                    any(not torch.equal(value, after[name]) for name, value in before.items())
                )
                rows = [
                    json.loads(line)
                    for line in (run / "supervised_kd/reference_trajectories.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(r["sampling_source"] == "reference" for r in rows))
                self.assertTrue(
                    all(p.grad is None and not p.requires_grad for p in teacher.parameters())
                )
                verify_inventory(run / "report")
                self.assertTrue(
                    (run / "report/source/project/src/qwen38/model_upgrade.py").is_file()
                )
                again = pipeline.run_distillation(
                    state,
                    sft_run_directory=sft_run["directory"],
                    run_directory=run,
                    stage="full",
                    config=config,
                )
                self.assertEqual(again["metrics"], full["metrics"])
                self.assertEqual(
                    len(
                        (run / "supervised_kd/reference_trajectories.jsonl")
                        .read_text()
                        .splitlines()
                    ),
                    4,
                )
