"""04 correctness tests: tiny RANDOM models are test machinery, not Qwen results."""

import copy
import io
import json
import math
import random
import tempfile
import time
import unittest
import zipfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3Config,
        Qwen3ForCausalLM,
        get_linear_schedule_with_warmup,
    )
except ImportError as error:
    raise unittest.SkipTest("04 tests need CPU torch and requirements-opd.txt") from error

from src.data import prepare_opd_splits
from src.generation import sample_trajectory
from src.logprobs import (
    response_prediction_logits,
    selected_log_probs,
    teacher_prompt_token_ids,
    verify_shared_tokenizer,
)
from src.opd import (
    OPDConfig,
    append_trajectories,
    backward_opd_group,
    backward_sft_group,
    cuda_preflight,
    experiment_signature,
    export_report,
    fixed_trajectory_diagnostics,
    load_student,
    optimizer_update,
    read_resume,
    release_cuda_memory,
    resolve_cold_start,
    restore_training_state,
    save_resume,
    seed_everything,
    training_health,
    trajectory_kl,
    trajectory_tensors,
    truncate_uncommitted_trajectories,
    validate_trajectory_log,
)
from src.opd_loss import forward_kl_loss, forward_kl_per_token
from src.sft import (
    collate_examples,
    evaluation_summary,
    generate_evaluation,
    precision_context,
    response_loss_sum,
    save_checkpoint,
    validation_loss,
    verify_checkpoint,
)
from src.utils import (
    artifact_directory,
    verified_zip,
    verify_inventory,
    write_inventory,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def tiny_model(seed=42):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
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


def tiny_tokenizer():
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<eos>": 1, "question": 2, "reason": 3, "5": 4, "6": 5}, unk_token="<unk>"
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )


def example(index=0, response=(3, 4, 1)):
    return {
        "example_id": f"unit/{index}",
        "source_index": index,
        "question": "question",
        "prompt": "question",
        "teacher_prompt_token_ids": [6, 2],
        "expected_answer": "5",
        "target_text": "reason 5",
        "input_ids": [2] + list(response),
        "labels": [-100] + list(response),
        "prompt_length": 1,
        "target_length": len(response),
    }


def trajectory(index=0, response=(3, 4, 1), policy_step=0):
    return {
        "example_id": f"unit/{index}",
        "policy_step": policy_step,
        "prompt_token_ids": [2],
        "teacher_prompt_token_ids": [6, 2],
        "teacher_prompt_length": 2,
        "response_token_ids": list(response),
        "complete_sequence_ids": [2] + list(response),
        "prompt_length": 1,
        "response_length": len(response),
        "response_text": "reason 5",
        "ended_with_eos": True,
        "hit_token_limit": False,
    }


class KLTests(unittest.TestCase):
    def test_direction_and_entropy_term(self):
        teacher = torch.tensor([[[0.8, 0.15, 0.05]]])
        student = torch.tensor([[[0.1, 0.7, 0.2]]])
        observed = forward_kl_per_token(student.log(), teacher.log(), 1.0)
        expected = (teacher * (teacher.log() - student.log())).sum(-1)
        self.assertTrue(torch.allclose(observed, expected))
        self.assertFalse(
            torch.allclose(observed, forward_kl_per_token(teacher.log(), student.log(), 1.0))
        )
        ce = -(teacher * student.log()).sum(-1)
        entropy = -(teacher * teacher.log()).sum(-1)
        self.assertTrue(torch.allclose(observed, ce - entropy))

    def test_chunked_value_and_gradient_match_dense_reference(self):
        torch.manual_seed(14)
        dense_logits = torch.randn(1, 7, 13, requires_grad=True)
        chunked_logits = dense_logits.detach().clone().requires_grad_()
        teacher = torch.randn_like(dense_logits, requires_grad=True)
        dense_loss = forward_kl_per_token(dense_logits, teacher, 0.1).mean()
        dense_loss.backward()
        loss = forward_kl_loss(chunked_logits, teacher, teacher_temperature=0.1, chunk_size=3)
        loss.backward()
        self.assertTrue(torch.allclose(loss, dense_loss, atol=2e-6))
        self.assertTrue(torch.allclose(chunked_logits.grad, dense_logits.grad, atol=2e-6))
        self.assertIsNone(teacher.grad)

    def test_full_vocabulary_includes_unsampled_columns(self):
        student = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 0.0, 0.0, 8.0]]])
        loss = forward_kl_loss(student, teacher, teacher_temperature=1, chunk_size=1)
        loss.backward()
        self.assertLess(student.grad[0, 0, 3].item(), 0)
        self.assertGreater(loss.item(), 1)

    def test_invalid_inputs_and_nonfinite_values_stop(self):
        logits = torch.ones(1, 2, 3)
        for temperature in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                forward_kl_per_token(logits, logits, temperature)
        with self.assertRaises(ValueError):
            forward_kl_loss(logits, logits[:, :1])
        with self.assertRaises(ValueError):
            forward_kl_loss(logits, logits, chunk_size=0)
        with self.assertRaises(FloatingPointError):
            forward_kl_loss(logits * float("nan"), logits)


class AlignmentAndSamplingTests(unittest.TestCase):
    def test_prompt_minus_one_shift_and_response_only_gradients(self):
        ids = torch.tensor([[2, 3, 4, 5, 1]])
        logits = torch.randn(1, 5, 7, requires_grad=True)
        response_logits = response_prediction_logits(logits, ids, 2)
        self.assertEqual(response_logits.shape, (1, 3, 7))
        self.assertTrue(torch.equal(response_logits, logits[:, 1:4]))
        scores = selected_log_probs(response_logits, ids[:, 2:])
        self.assertEqual(scores.shape, (1, 3))
        scores.sum().backward()
        self.assertTrue(torch.equal(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0])))
        self.assertTrue(torch.equal(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1])))
        self.assertGreater(logits.grad[:, 1:4].abs().sum().item(), 0)

    def test_bad_boundaries_fail(self):
        logits, ids = torch.zeros(1, 3, 8), torch.tensor([[2, 3, 1]])
        for boundary in (0, 3, -1):
            with self.assertRaises(ValueError):
                response_prediction_logits(logits, ids, boundary)

    def test_generation_policy_is_fresh_unfiltered_and_reference_blind(self):
        model, tokenizer = tiny_model().train(), tiny_tokenizer()
        captured = []

        def fake_generate(**kwargs):
            captured.append(kwargs)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(model.training)
            return torch.cat([kwargs["input_ids"], torch.tensor([[3, 4, 1]])], dim=1)

        with patch.object(model, "generate", side_effect=fake_generate):
            first = sample_trajectory(
                model,
                tokenizer,
                example(),
                device="cpu",
                dtype=None,
                max_new_tokens=3,
                stop_token_ids=[1, 5],
                policy_step=7,
            )
            altered_reference = {
                **example(),
                "target_text": "secret reference",
                "expected_answer": "999",
            }
            second = sample_trajectory(
                model,
                tokenizer,
                altered_reference,
                device="cpu",
                dtype=None,
                max_new_tokens=3,
                stop_token_ids=[1, 5],
                policy_step=7,
            )
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        policy = captured[0]["generation_config"]
        self.assertEqual((policy.top_k, policy.top_p, policy.temperature), (0, 1.0, 1.0))
        self.assertTrue(policy.do_sample)
        self.assertEqual(policy.eos_token_id, [1, 5])
        self.assertEqual(captured[0]["input_ids"].tolist(), [[2]])
        self.assertEqual(first["policy_step"], 7)
        self.assertFalse(first["hit_token_limit"])

    def test_capped_trajectory_does_not_get_an_invented_eos(self):
        model, tokenizer = tiny_model(), tiny_tokenizer()
        with patch.object(model, "generate", return_value=torch.tensor([[2, 3, 4]])):
            result = sample_trajectory(
                model,
                tokenizer,
                example(),
                device="cpu",
                dtype=None,
                max_new_tokens=2,
                stop_token_ids=[1, 5],
                policy_step=0,
            )
        self.assertTrue(result["hit_token_limit"])
        self.assertEqual(result["response_token_ids"], [3, 4])

    def test_effective_generation_does_not_inherit_model_sampling_defaults(self):
        model, tokenizer = tiny_model(), tiny_tokenizer()
        model.generation_config.temperature = 0.2
        model.generation_config.top_p = 0.6
        model.generation_config.repetition_penalty = 1.2
        with patch.object(
            model, "_get_logits_processor", wraps=model._get_logits_processor
        ) as processors:
            sample_trajectory(
                model,
                tokenizer,
                example(),
                device="cpu",
                dtype=None,
                max_new_tokens=3,
                stop_token_ids=[1, 5],
                policy_step=0,
            )
        effective_policy = processors.call_args.kwargs["generation_config"]
        self.assertEqual(effective_policy.temperature, 1.0)
        self.assertEqual(effective_policy.top_p, 1.0)
        self.assertEqual(effective_policy.top_k, 0)
        self.assertEqual(effective_policy.repetition_penalty, 1.0)

    def test_mapping_size_alone_does_not_hide_encoding_difference(self):
        left, right = tiny_tokenizer(), tiny_tokenizer()
        with patch.object(right, "encode", return_value=[4]):
            with self.assertRaisesRegex(ValueError, "tokenize/decode"):
                verify_shared_tokenizer(left, right, ["question"], 32)


class TrainingTests(unittest.TestCase):
    def test_teacher_context_preserves_response_ids_and_frozen_weights(self):
        student, teacher = tiny_model(), tiny_model(77).requires_grad_(False).eval()
        before_teacher = {k: v.clone() for k, v in teacher.state_dict().items()}
        before_student = {k: v.clone() for k, v in student.state_dict().items()}
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with patch.object(teacher, "forward", wraps=teacher.forward) as teacher_forward:
            metrics = backward_opd_group(
                student, teacher, [trajectory()], "cpu", None, OPDConfig(), scaler, 0
            )
            self.assertEqual(
                teacher_forward.call_args.kwargs["input_ids"].tolist(), [[6, 2, 3, 4, 1]]
            )
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        optimizer.step()
        self.assertTrue(math.isfinite(metrics["loss"]))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(
            all(torch.equal(v, before_teacher[k]) for k, v in teacher.state_dict().items())
        )
        self.assertTrue(
            any(not torch.equal(v, before_student[k]) for k, v in student.state_dict().items())
        )

    def test_accumulation_is_mean_of_response_means(self):
        first, second, teacher = (
            tiny_model(),
            tiny_model(),
            tiny_model(77).requires_grad_(False).eval(),
        )
        trajectories = [trajectory(), trajectory(1, (4, 1))]
        config = OPDConfig()
        dense = sum(trajectory_kl(first, teacher, t, "cpu", None, config) for t in trajectories) / 2
        dense.backward()
        backward_opd_group(
            second,
            teacher,
            trajectories,
            "cpu",
            None,
            config,
            torch.amp.GradScaler("cuda", enabled=False),
            0,
        )
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-6))

    def test_sft_control_uses_same_response_weighting_and_partial_group(self):
        first, second = tiny_model(), tiny_model()
        examples = [example(), example(1, (4, 1))]
        reference = (
            sum(
                response_loss_sum(first, collate_examples([e], 1))[0] / e["target_length"]
                for e in examples
            )
            / 2
        )
        reference.backward()
        backward_sft_group(
            second, examples, 1, "cpu", None, torch.amp.GradScaler("cuda", enabled=False)
        )
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-6))

    def test_stale_trajectory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "current"):
            backward_opd_group(
                tiny_model(),
                tiny_model(77),
                [trajectory(policy_step=0)],
                "cpu",
                None,
                OPDConfig(),
                torch.amp.GradScaler("cuda", enabled=False),
                1,
            )

    def test_diagnostics_preserve_student_mode_and_weights(self):
        student, teacher = tiny_model().train(), tiny_model(77).requires_grad_(False)
        before = {k: v.clone() for k, v in student.state_dict().items()}
        result = fixed_trajectory_diagnostics(
            student, teacher, [trajectory()], "cpu", None, OPDConfig()
        )
        self.assertTrue(student.training)
        self.assertTrue(math.isfinite(result["mean_forward_kl_nats"]))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in student.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in student.parameters()))

    def test_nonfinite_gradient_cannot_update_weights_or_scheduler(self):
        model = tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 2)
        next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("nan"))
        original = copy.deepcopy(model.state_dict())
        epoch = scheduler.last_epoch
        with self.assertRaises(RuntimeError):
            optimizer_update(
                model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False), 1
            )
        self.assertEqual(scheduler.last_epoch, epoch)
        self.assertTrue(all(torch.equal(v, original[k]) for k, v in model.state_dict().items()))

    def test_actual_notebook_loop_executes_both_arms_with_random_models(self):
        notebook = json.loads((ROOT / "notebooks/04_on_policy_distillation.ipynb").read_text())
        loop = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "opd-train-arm")
        with tempfile.TemporaryDirectory() as folder:
            directory, tokenizer = Path(folder), tiny_tokenizer()
            checkpoint = directory / "initial"
            save_checkpoint(tiny_model(), tokenizer, checkpoint)
            examples = [example(i, (3, 4, 1) if i % 2 else (4, 1)) for i in range(3)]
            config = OPDConfig(
                train_examples=3,
                test_examples=1,
                accumulation_steps=2,
                validation_interval=1,
                checkpoint_interval=1,
                max_prompt_length=8,
                max_response_length=3,
                max_sequence_length=16,
                diagnostic_examples=1,
                learning_rate=1e-3,
            )
            evaluated = []

            def evaluation(model, rows, name):
                result = generate_evaluation(
                    model,
                    tokenizer,
                    rows,
                    {
                        "max_new_tokens": 3,
                        "do_sample": False,
                        "num_beams": 1,
                        "eos_token_id": [1, 5],
                        "pad_token_id": 1,
                        "use_cache": True,
                        "bos_token_id": 0,
                    },
                    "cpu",
                    None,
                    use_model_defaults=False,
                )
                evaluated.append(name)
                write_json(directory / f"{name}.json", result)
                return result

            namespace = dict(
                globals(),
                config=config,
                starting_validation_loss=4.0,
                signature="tiny-qa",
                run_directory=directory,
                starting_checkpoint_directory=checkpoint,
                tokenizer=tokenizer,
                device="cpu",
                compute_dtype=None,
                train_examples=examples,
                validation_examples=examples[:1],
                diagnostic_trajectories=[trajectory()],
                stop_token_ids=[1, 5],
                evaluation_policy={
                    "max_new_tokens": 3,
                    "do_sample": False,
                    "num_beams": 1,
                    "eos_token_id": [1, 5],
                    "pad_token_id": 1,
                    "bos_token_id": 0,
                    "use_cache": True,
                },
                evaluate_and_save=evaluation,
                time=time,
                release_cuda_memory=release_cuda_memory,
                evaluation_summary=evaluation_summary,
                validation_loss=validation_loss,
                verify_checkpoint=verify_checkpoint,
                validate_trajectory_log=validate_trajectory_log,
            )
            with redirect_stdout(io.StringIO()):
                exec(compile(loop, "notebook04_training_cell", "exec"), namespace)
                control = namespace["train_arm"]("sft_control")
                teacher = tiny_model(77).requires_grad_(False).eval()
                on_policy = namespace["train_arm"]("on_policy", teacher=teacher)
                reused = namespace["train_arm"]("on_policy", teacher=teacher)
                # Recover an interruption during finalization, after saving weights.
                (directory / "on_policy_training.json").unlink()
                recovered = namespace["train_arm"]("on_policy", teacher=teacher)
            self.assertEqual((control["completed_steps"], on_policy["completed_steps"]), (2, 2))
            self.assertTrue(
                control["checkpoint_reload_verified"] and on_policy["checkpoint_reload_verified"]
            )
            self.assertEqual(reused, on_policy)
            self.assertEqual(recovered["resumed_from_step"], 2)
            self.assertEqual(recovered["completed_steps"], 2)
            samples = [
                json.loads(line)
                for line in (directory / "training_trajectories.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["policy_step"] for r in samples], [0, 0, 1])
            self.assertEqual(len(samples), 3)
            saved_policy = json.loads(
                (directory / "checkpoints/on_policy/generation_config.json").read_text()
            )
            self.assertEqual(saved_policy["eos_token_id"], [1, 5])
            self.assertFalse(saved_policy.get("do_sample", False))
            self.assertEqual(
                evaluated,
                ["sft_control_validation", "on_policy_validation", "on_policy_validation"],
            )
            # Execute the actual final evaluation/plot/export cells too. Everything
            # remains inside a temporary directory and is labelled random-model QA.
            import importlib.metadata
            import sys

            import pandas as pd

            from src.opd import paired_accuracy_difference
            from src.opd_plots import plot_run

            namespace.update(
                MODE="train",
                FULL_RUN_COMPLETE=True,
                teacher_template_sha256="tiny-template",
                PROJECT_ROOT=ROOT,
                control_result=control,
                on_policy_result=on_policy,
                pd=pd,
                display=lambda value: None,
                splits={"primary_test": examples, "historical_test": examples[:1]},
                paired_accuracy_difference=paired_accuracy_difference,
                plot_run=plot_run,
                runtime={"kind": "tiny_random_cpu_test_only"},
                protocol={"student_model_id": "tiny-random-qa", "student_revision": "test-only"},
                TEACHER_ID="tiny-random-qa-teacher",
                TEACHER_REVISION="test-only",
                checkpoint_hashes=json.loads((checkpoint / "SHA256SUMS.json").read_text()),
                source_train=SimpleNamespace(_fingerprint="synthetic-train-fixtures"),
                source_test=SimpleNamespace(_fingerprint="synthetic-test-fixtures"),
                sys=sys,
                importlib=importlib,
                shutil=__import__("shutil"),
                export_report=export_report,
            )
            with redirect_stdout(io.StringIO()):
                for cell_id in ("opd-final-evaluation", "opd-changed-answers", "opd-export"):
                    source = next(
                        "".join(c["source"]) for c in notebook["cells"] if c["id"] == cell_id
                    )
                    exec(compile(source, cell_id, "exec"), namespace)
            self.assertTrue((directory / "primary_test_comparison.png").is_file())
            self.assertTrue((directory / "training_objectives.svg").is_file())
            self.assertTrue((directory / "report.zip").is_file())
            self.assertEqual(
                json.loads((directory / "report/manifest.json").read_text())["runtime"]["kind"],
                "tiny_random_cpu_test_only",
            )
            verify_inventory(directory / "report")

    def test_checkpoint_verification_preserves_training_mode(self):
        student, tokenizer = tiny_model().train(), tiny_tokenizer()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "checkpoint"
            save_checkpoint(student, tokenizer, directory)
            self.assertTrue(
                verify_checkpoint(student, tokenizer, directory, [2, 3, 4, 1], "cpu", None)
            )
        self.assertTrue(student.training)


class ArtifactAndProtocolTests(unittest.TestCase):
    def test_config_rejects_invalid_budgets(self):
        for options in (
            {"train_examples": 0},
            {"max_sequence_length": 100},
            {"teacher_temperature": float("inf")},
            {"warmup_fraction": 1},
        ):
            with self.assertRaises(ValueError):
                OPDConfig(**options)

    def test_no_checkpoint_never_falls_back_to_base_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "evidence, not weights"):
                resolve_cold_start(Path(folder))

    def test_cpu_preflight_stops_before_teacher_load(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Colab CUDA"):
                cuda_preflight(ROOT)

    def test_zip_inventory_round_trip_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"weights")
            hashes = write_inventory(artifact)
            archive = root / "artifact.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                for path in artifact.iterdir():
                    stream.write(path, path.name)
            expanded = artifact_directory(archive, root / "cache")
            self.assertEqual(verify_inventory(expanded), hashes)
            self.assertEqual(artifact_directory(archive, root / "cache"), expanded)
            (expanded / "weights").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_inventory(expanded)

    def test_zip_rejects_traversal_duplicates_and_size_overruns(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("../escape", "/absolute", "nested/../../escape", "a\\b"):
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as stream:
                    stream.writestr(name, b"bad")
                with self.assertRaises(ValueError):
                    artifact_directory(archive, root / "cache")
            with zipfile.ZipFile(root / "large.zip", "w") as stream:
                stream.writestr("bytes", b"12345")
            with self.assertRaisesRegex(ValueError, "size limit"):
                artifact_directory(root / "large.zip", root / "cache", max_bytes=2)

    def test_saved_state_restores_rng_and_the_next_optimizer_update(self):
        tokenizer, student = tiny_tokenizer(), tiny_model()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        seed_everything(42)

        def step(model, opt, sched, scale):
            opt.zero_grad(set_to_none=True)
            backward_sft_group(model, [example()], 1, "cpu", None, scale)
            optimizer_update(model, opt, sched, scale, 1)

        step(student, optimizer, scheduler, scaler)
        with tempfile.TemporaryDirectory() as folder:
            snapshot = save_resume(
                folder,
                student,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                arm="sft_control",
                step=1,
                signature="fixed",
                history=[{"step": 1}],
            )
            metadata = read_resume(snapshot, arm="sft_control", signature="fixed")
            expected_python, expected_torch = random.random(), torch.rand(4)
            step(student, optimizer, scheduler, scaler)
            restored = load_student(snapshot / "student", "cpu")
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_scheduler = get_linear_schedule_with_warmup(restored_optimizer, 0, 4)
            restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
            restore_training_state(
                snapshot, restored_optimizer, restored_scheduler, restored_scaler
            )
            self.assertEqual(random.random(), expected_python)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch))
            step(restored, restored_optimizer, restored_scheduler, restored_scaler)
            for a, b in zip(student.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(metadata["completed_steps"], 1)
            with self.assertRaises(ValueError):
                read_resume(snapshot, arm="on_policy", signature="fixed")

    def test_resume_removes_only_rollouts_beyond_its_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectories.jsonl"
            append_trajectories(path, [trajectory(policy_step=i) for i in range(4)])
            with path.open("a") as stream:
                stream.write('{"partial":')
            truncate_uncommitted_trajectories(path, 2)
            self.assertEqual(
                [json.loads(line)["policy_step"] for line in path.read_text().splitlines()], [0, 1]
            )

    def test_audit_log_rejects_missing_or_reordered_rollouts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectories.jsonl"
            rows = [trajectory(0), trajectory(1), trajectory(2, policy_step=1)]
            append_trajectories(path, rows)
            self.assertEqual(validate_trajectory_log(path, [example(i) for i in range(3)], 2, 2), 3)
            with self.assertRaisesRegex(ValueError, "order"):
                validate_trajectory_log(path, [example(1), example(0), example(2)], 2, 2)
            path.write_text(path.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_trajectory_log(path, [example(i) for i in range(3)], 2, 2)

    def test_snapshot_retention_preserves_unrelated_directories(self):
        student, tokenizer = tiny_model(), tiny_tokenizer()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder) / "resume/sft_control"
            unrelated = parent / "step_0000"
            unrelated.mkdir(parents=True)
            write_json(unrelated / "resume.json", {"arm": "sft_control", "signature": "unrelated"})
            for step in range(1, 4):
                save_resume(
                    folder,
                    student,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    arm="sft_control",
                    step=step,
                    signature="owned",
                    history=[{"step": i} for i in range(1, step + 1)],
                )
            self.assertTrue(unrelated.exists())
            self.assertFalse((parent / "step_0001").exists())
            self.assertTrue((parent / "step_0002").exists())
            self.assertTrue((parent / "step_0003").exists())

    def test_resume_signature_tracks_protocol_and_code_changes(self):
        signature = experiment_signature(OPDConfig(), {}, {}, {}, {}, {"opd.py": "abc"})
        self.assertNotEqual(
            signature,
            experiment_signature(replace(OPDConfig(), seed=7), {}, {}, {}, {}, {"opd.py": "abc"}),
        )
        self.assertNotEqual(
            signature, experiment_signature(OPDConfig(), {}, {}, {}, {}, {"opd.py": "def"})
        )

    def test_report_export_excludes_weights_and_keeps_auditable_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_json(directory / "experiment.json", {"seed": 42})
            (directory / "checkpoints").mkdir()
            (directory / "checkpoints/model.safetensors").write_bytes(b"not real weights")
            archive = export_report(ROOT, directory, {"status": "test_machinery"})
            with zipfile.ZipFile(archive) as stream:
                self.assertIn("source/opd_loss.py", stream.namelist())
                self.assertIn("source/notebook04_source.json", stream.namelist())
                self.assertFalse(any(name.endswith(".safetensors") for name in stream.namelist()))
            verify_inventory(directory / "report")
            first_inventory = (directory / "report/SHA256SUMS.json").read_bytes()
            self.assertEqual(export_report(ROOT, directory, {"status": "test_machinery"}), archive)
            self.assertEqual((directory / "report/SHA256SUMS.json").read_bytes(), first_inventory)
            with self.assertRaises(FileExistsError):
                export_report(ROOT, directory, {"status": "a_different_run"})

    def test_paired_comparison_preserves_question_identity(self):
        from src.opd import paired_accuracy_difference

        first = {"annotations": [{"example_id": str(i), "strict_correct": False} for i in range(4)]}
        second = {
            "annotations": [
                {"example_id": str(i), "strict_correct": True} for i in reversed(range(4))
            ]
        }
        comparison = paired_accuracy_difference(first, second, repetitions=20)
        self.assertEqual(comparison["difference_second_minus_first"], 1)
        self.assertEqual(comparison["paired_bootstrap_95_interval"], [1, 1])
        with self.assertRaises(ValueError):
            paired_accuracy_difference(first, {"annotations": []})


class CorrectedContextTests(unittest.TestCase):
    def test_template_uses_verbatim_question_and_nonthinking_assistant_prefix(self):
        tokenizer = SimpleNamespace(chat_template="pinned template")
        from unittest.mock import Mock

        tokenizer.apply_chat_template = Mock(return_value=[7, 8, 2])
        prompt = "Question: Café 中文 🙂 3 + 2?\n\nAnswer:\n"
        self.assertEqual(teacher_prompt_token_ids(tokenizer, prompt), [7, 8, 2])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokenizer.chat_template = None
        with self.assertRaises(ValueError):
            teacher_prompt_token_ids(tokenizer, prompt)

    def test_separate_context_lengths_align_identical_response_ids(self):
        row = trajectory(response=(3, 4, 1))
        student_ids, _ = trajectory_tensors(row, "cpu")
        teacher_ids, _ = trajectory_tensors(row, "cpu", context="teacher")
        self.assertEqual(student_ids.tolist(), [[2, 3, 4, 1]])
        self.assertEqual(teacher_ids.tolist(), [[6, 2, 3, 4, 1]])
        student_logits = torch.randn(1, 4, 32, requires_grad=True)
        teacher_logits = torch.randn(1, 5, 32, requires_grad=True)
        student_scores = response_prediction_logits(student_logits, student_ids, 1)
        teacher_scores = response_prediction_logits(teacher_logits, teacher_ids, 2)
        self.assertEqual(student_scores.shape, teacher_scores.shape)
        loss = forward_kl_loss(student_scores, teacher_scores)
        loss.backward()
        self.assertIsNone(teacher_logits.grad)
        self.assertTrue(
            torch.equal(student_logits.grad[:, -1], torch.zeros_like(student_logits.grad[:, -1]))
        )
        legacy = {k: v for k, v in row.items() if not k.startswith("teacher_prompt")}
        with self.assertRaisesRegex(ValueError, "Legacy"):
            trajectory_tensors(legacy, "cpu", context="teacher")

    def test_actual_generation_stops_at_the_teachers_end_id(self):
        model, tokenizer = tiny_model(), tiny_tokenizer()
        original_forward = model.forward

        def favor_teacher_end(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            result.logits[..., 5] = 1000
            return result

        with patch.object(model, "forward", side_effect=favor_teacher_end):
            row = sample_trajectory(
                model,
                tokenizer,
                example(),
                device="cpu",
                dtype=None,
                max_new_tokens=3,
                stop_token_ids=[1, 5],
                policy_step=0,
            )
        self.assertEqual(row["response_token_ids"], [5])
        self.assertTrue(row["ended_with_eos"])
        self.assertFalse(row["hit_token_limit"])
        self.assertEqual(row["teacher_prompt_token_ids"], [6, 2])

    def test_health_window_and_reference_loss_warning_are_independent(self):
        config = OPDConfig()
        healthy = [{"response_limit_flags": [True] * 23 + [False] * 9}]
        self.assertFalse(training_health(healthy, config, 1.0)["pause"])
        stopped = [{"response_limit_flags": [True] * 24 + [False] * 8}]
        self.assertTrue(training_health(stopped, config, 1.0)["pause"])
        short = [{"response_limit_flags": [True] * 31}]
        self.assertFalse(training_health(short, config, 1.0)["pause"])
        warning = [{"validation_reference_nll": 1.3}]
        result = training_health(warning, config, 1.0)
        self.assertFalse(result["pause"])
        self.assertEqual(len(result["warnings"]), 1)

    def test_full_stage_notebook_requires_reviewed_pilot_before_training(self):
        notebook = json.loads((ROOT / "notebooks/04_on_policy_distillation.ipynb").read_text())
        source = next(
            "".join(c["source"]) for c in notebook["cells"] if c["id"] == "opd-run-on-policy"
        )
        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(MODE="train", RUN_STAGE="full", run_directory=Path(folder))
            with self.assertRaisesRegex(AssertionError, "pilot snapshot"):
                exec(compile(source, "stage_gate", "exec"), namespace)

    def test_interrupted_json_write_preserves_the_previous_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "history.json"
            write_json(path, {"step": 1})
            with patch.object(Path, "write_text", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    write_json(path, {"step": 2})
            self.assertEqual(json.loads(path.read_text()), {"step": 1})
            self.assertFalse(list(Path(folder).glob(".json-*")))

    def test_checkpoint_zip_reuse_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "checkpoint"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"test weights")
            write_inventory(artifact)
            archive = verified_zip(artifact, root / "checkpoint.zip")
            original_mtime = archive.stat().st_mtime_ns
            self.assertEqual(verified_zip(artifact, archive), archive)
            self.assertEqual(archive.stat().st_mtime_ns, original_mtime)
            archive.write_bytes(b"interrupted download")
            with self.assertRaisesRegex(ValueError, "damaged ZIP"):
                verified_zip(artifact, archive)
            self.assertEqual((artifact / "weights").read_bytes(), b"test weights")

    def test_interrupted_packaging_never_exposes_a_final_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "checkpoint"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"test weights")
            write_inventory(artifact)
            with patch("zipfile.ZipFile.write", side_effect=OSError("simulated interrupted write")):
                with self.assertRaises(OSError):
                    verified_zip(artifact, root / "checkpoint.zip")
            self.assertFalse((root / "checkpoint.zip").exists())
            self.assertFalse(list(root.glob(".archive-*")))

    def test_concurrent_checkpoint_packaging_publishes_one_verified_archive(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "checkpoint"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"concurrent test weights")
            write_inventory(artifact)
            archive = root / "checkpoint.zip"
            barrier = Barrier(2)
            original = Path.hardlink_to

            def publish(path, target):
                barrier.wait(timeout=10)
                return original(path, target)

            with (
                patch.object(Path, "hardlink_to", publish),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                futures = [pool.submit(verified_zip, artifact, archive) for _ in range(2)]
                self.assertEqual([future.result() for future in futures], [archive, archive])
            self.assertEqual(verified_zip(artifact, archive), archive)
            self.assertFalse(list(root.glob(".archive-*")))

    def test_zip_checks_entry_hashes_beyond_valid_crc(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "checkpoint"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"actual weights")
            write_inventory(artifact)
            archive = root / "tampered.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr("weights", b"different weights")
                stream.write(artifact / "SHA256SUMS.json", "SHA256SUMS.json")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verified_zip(artifact, archive)

    def test_named_pilot_report_survives_a_later_full_export(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_json(directory / "experiment.json", {"signature": "same-run"})
            pilot = export_report(
                ROOT,
                directory,
                {"status": "pilot_complete", "signature": "same-run"},
                report_name="report_pilot_complete_0032",
            )
            original = pilot.read_bytes()
            full = export_report(ROOT, directory, {"status": "completed", "signature": "same-run"})
            self.assertNotEqual(pilot, full)
            self.assertEqual(pilot.read_bytes(), original)
            with self.assertRaises(ValueError):
                export_report(ROOT, directory, {"status": "bad"}, report_name="../escape")


class StagedTrainingTests(unittest.TestCase):
    def namespace(self, directory, config):
        tokenizer = tiny_tokenizer()
        checkpoint = directory / "initial"
        save_checkpoint(tiny_model(), tokenizer, checkpoint)
        examples = [example(i) for i in range(config.train_examples)]

        def evaluate(model, rows, name):
            result = generate_evaluation(
                model,
                tokenizer,
                rows,
                {
                    "max_new_tokens": 3,
                    "do_sample": False,
                    "num_beams": 1,
                    "eos_token_id": [1, 5],
                    "pad_token_id": 1,
                    "bos_token_id": 0,
                },
                "cpu",
                None,
                use_model_defaults=False,
            )
            write_json(directory / f"{name}.json", result)
            return result

        namespace = dict(
            globals(),
            config=config,
            precision_context=precision_context,
            starting_validation_loss=4.0,
            signature="staged-test-only",
            run_directory=directory,
            starting_checkpoint_directory=checkpoint,
            tokenizer=tokenizer,
            device="cpu",
            compute_dtype=None,
            train_examples=examples,
            validation_examples=examples[:1],
            diagnostic_trajectories=[trajectory()],
            stop_token_ids=[1, 5],
            evaluation_policy={
                "max_new_tokens": 3,
                "do_sample": False,
                "num_beams": 1,
                "eos_token_id": [1, 5],
                "pad_token_id": 1,
                "bos_token_id": 0,
            },
            evaluate_and_save=evaluate,
        )
        notebook = json.loads((ROOT / "notebooks/04_on_policy_distillation.ipynb").read_text())
        source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "opd-train-arm")
        exec(compile(source, "actual_staged_notebook_loop", "exec"), namespace)
        return namespace

    def test_pilot_resume_matches_uninterrupted_weights_rng_and_schedule(self):
        from safetensors.torch import load_file

        config = OPDConfig(
            train_examples=8,
            accumulation_steps=2,
            pilot_updates=2,
            pilot_check_interval=1,
            checkpoint_interval=2,
            validation_interval=2,
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            diagnostic_examples=1,
            cap_window_responses=100,
        )
        teacher = tiny_model(77).requires_grad_(False).eval()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            staged, continuous = root / "staged", root / "continuous"
            staged.mkdir()
            continuous.mkdir()
            namespace = self.namespace(staged, config)
            pilot = namespace["train_arm"]("on_policy", teacher, stop_after_steps=2)
            self.assertEqual(pilot["status"], "pilot_complete")
            self.assertEqual(pilot["completed_steps"], 2)
            self.assertFalse((staged / "checkpoints/on_policy").exists())
            self.assertTrue(Path(pilot["resume_snapshot"]).is_dir())
            resumed = namespace["train_arm"]("on_policy", teacher)
            other = self.namespace(continuous, config)
            uninterrupted = other["train_arm"]("on_policy", teacher)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["resumed_from_step"], 2)
            left = load_file(staged / "checkpoints/on_policy/model.safetensors")
            right = load_file(continuous / "checkpoints/on_policy/model.safetensors")
            self.assertTrue(all(torch.equal(left[k], right[k]) for k in left))
            self.assertEqual(
                [r["learning_rate"] for r in resumed["history"]],
                [r["learning_rate"] for r in uninterrupted["history"]],
            )
            self.assertEqual(
                (staged / "training_trajectories.jsonl").read_text(),
                (continuous / "training_trajectories.jsonl").read_text(),
            )
            for key in ("torch_rng", "python_rng", "scheduler"):
                first = torch.load(
                    staged / "resume/on_policy/step_0004/training_state.pt", weights_only=True
                )[key]
                second = torch.load(
                    continuous / "resume/on_policy/step_0004/training_state.pt", weights_only=True
                )[key]
                if isinstance(first, torch.Tensor):
                    self.assertTrue(torch.equal(first, second))
                else:
                    self.assertEqual(first, second)

    def test_default_pilot_boundary_is_32_of_128_updates(self):
        # Random-model responses need no quality qualification in this workflow test.
        config = OPDConfig(
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            cap_window_responses=10000,
        )
        self.assertEqual(config.optimizer_steps, 128)
        teacher = tiny_model(77).requires_grad_(False).eval()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            namespace = self.namespace(Path(folder), config)
            result = namespace["train_arm"](
                "on_policy", teacher, stop_after_steps=config.pilot_updates
            )
            self.assertEqual(result["completed_steps"], 32)
            self.assertEqual(result["status"], "pilot_complete")
            self.assertEqual(result["resumed_from_step"], 0)
            rows = (Path(folder) / "training_trajectories.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 256)
            metadata = read_resume(
                Path(result["resume_snapshot"]), arm="on_policy", signature="staged-test-only"
            )
            self.assertEqual(metadata["completed_steps"], 32)

    def test_actual_teacher_alignment_cell_uses_separate_contexts(self):
        # Execute the notebook's entire alignment/qualification cell using random models.
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            config = OPDConfig(
                train_examples=2,
                accumulation_steps=1,
                max_prompt_length=8,
                max_response_length=3,
                max_sequence_length=16,
            )
            namespace = self.namespace(Path(folder), config)
            teacher = tiny_model(77).requires_grad_(False).eval()
            for row in namespace["validation_examples"]:
                # These two IDs decode to 'reason question' in the tiny tokenizer.
                row["teacher_prompt_token_ids"] = [3, 2]
            namespace["diagnostic_trajectories"][0]["teacher_prompt_token_ids"] = [3, 2]
            namespace.update(
                MODE="train",
                pd=__import__("pandas"),
                display=lambda frame: None,
                teacher_tokenizer=namespace["tokenizer"],
                load_teacher=lambda device: teacher,
                starting_validation={"scores": {"strict": {"accuracy": 0.0}}},
            )
            notebook = json.loads((ROOT / "notebooks/04_on_policy_distillation.ipynb").read_text())
            source = next(
                "".join(c["source"])
                for c in notebook["cells"]
                if c["id"] == "opd-teacher-alignment"
            )
            exec(compile(source, "actual_teacher_alignment_cell", "exec"), namespace)
            boundary = json.loads((Path(folder) / "teacher_boundary_probe.json").read_text())
            self.assertTrue(boundary["sampled_ended_with_eos"])
            self.assertTrue(0 <= boundary["teacher_target_stop_probability"] <= 1)
            self.assertTrue(
                all(p.grad is None and not p.requires_grad for p in teacher.parameters())
            )
            self.assertIn("starting_fixed_diagnostics", namespace)

    def test_health_pause_saves_nonperiodic_boundary_and_blocks_resume(self):
        config = OPDConfig(
            train_examples=8,
            accumulation_steps=2,
            pilot_updates=4,
            pilot_check_interval=3,
            checkpoint_interval=3,
            validation_interval=3,
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            diagnostic_examples=1,
            cap_window_responses=4,
        )
        teacher = tiny_model(77).requires_grad_(False).eval()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            namespace = self.namespace(Path(folder), config)

            def capped(model, tokenizer, row, **kwargs):
                return trajectory(row["source_index"], (3, 4, 4), kwargs["policy_step"]) | {
                    "ended_with_eos": False,
                    "hit_token_limit": True,
                }

            namespace["sample_trajectory"] = capped
            result = namespace["train_arm"]("on_policy", teacher)
            self.assertEqual(result["status"], "health_paused")
            self.assertEqual(result["completed_steps"], 2)
            self.assertTrue((Path(folder) / "resume/on_policy/step_0002/resume.json").exists())
            self.assertFalse((Path(folder) / "checkpoints/on_policy").exists())
            with self.assertRaisesRegex(RuntimeError, "health-paused"):
                namespace["train_arm"]("on_policy", teacher)


class DataTests(unittest.TestCase):
    def test_new_training_and_test_selections_are_disjoint_and_deterministic(self):
        class CharacterTokenizer:
            eos_token_id = 1

            def __call__(self, text, **kwargs):
                return {"input_ids": self.encode(text)}

            def encode(self, text, **kwargs):
                return [ord(char) + 2 for char in text]

        source_train = [
            {"question": f"Practice {i}", "answer": "Reason.\n#### 5"} for i in range(20)
        ]
        source_test = [{"question": f"Exam {i}", "answer": "#### 5"} for i in range(10)]
        source_train += [{"question": "ＥＸＡＭ  1", "answer": "#### 5"}]
        cold = {"train": [{"question": "Practice 0"}], "validation": [{"question": "Practice 1"}]}
        protocol = {
            "prompt_template": "{question}\nAnswer:\n",
            "examples": [{"question": "Exam 0"}],
        }
        kwargs = dict(
            train_examples=4, test_examples=3, max_sequence_length=100, max_prompt_length=50
        )
        result = prepare_opd_splits(
            source_train, source_test, CharacterTokenizer(), protocol, cold, **kwargs
        )
        self.assertEqual(
            result,
            prepare_opd_splits(
                source_train, source_test, CharacterTokenizer(), protocol, cold, **kwargs
            ),
        )
        self.assertFalse(
            {r["question"] for r in result["train"]} & {"Practice 0", "Practice 1", "ＥＸＡＭ  1"}
        )
        self.assertNotIn("Exam 0", {r["question"] for r in result["primary_test"]})
        self.assertEqual(len(result["train"]), 4)
        self.assertEqual(len(result["primary_test"]), 3)


if __name__ == "__main__":
    unittest.main()
