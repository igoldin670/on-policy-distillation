"""EOS/recovery tests use tiny random models and synthetic records, never Qwen results."""

import io
import json
import random
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_opd as fixtures
import torch
from test_opd import (
    example,
    tiny_config,
    tiny_model,
    tiny_tokenizer,
    trajectory,
)

from src.opd import OPDConfig, experiment_signature, trajectory_kl
from src.opd_diagnostics import probe_boundaries, select_probe_trajectories, stopping_statistics
from src.opd_experiment import (
    SlurmBudget,
    WalltimePause,
    comparison_gates,
    evaluate_incrementally,
    publish_pause,
    run_eos_comparison,
    verify_historical_run,
)
from src.opd_loss import forward_kl_loss, forward_kl_per_token
from src.sft import generate_evaluation


class SemanticLossTests(unittest.TestCase):
    def test_swapping_eos_alias_mass_changes_only_raw_objective(self):
        student = torch.tensor([[[0.79, 0.01, 0.2]]]).log().requires_grad_()
        teacher = torch.tensor([[[0.01, 0.79, 0.2]]]).log().requires_grad_()
        self.assertGreater(forward_kl_loss(student, teacher, teacher_temperature=1).item(), 1)
        loss = forward_kl_loss(student, teacher, teacher_temperature=1, eos_token_ids=(0, 1))
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0, places=6)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student), atol=1e-6))
        self.assertIsNone(teacher.grad)

    def test_dense_reference_value_and_gradient_after_temperature_scaling(self):
        torch.manual_seed(3)
        student = torch.randn(1, 7, 9, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        original = student.detach().clone().requires_grad_()
        p = original.softmax(-1)
        q = (teacher.detach() / 0.1).softmax(-1)
        p = torch.cat([p[..., :2].sum(-1, keepdim=True), p[..., 2:]], -1)
        q = torch.cat([q[..., :2].sum(-1, keepdim=True), q[..., 2:]], -1)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(1, 7)))
        self.assertTrue(torch.allclose(q.sum(-1), torch.ones(1, 7)))
        expected = (q * (q.log() - p.log())).sum(-1).mean()
        expected.backward()
        result = forward_kl_loss(student, teacher, eos_token_ids=(0, 1), chunk_size=3)
        result.backward()
        self.assertTrue(torch.allclose(result, expected, atol=2e-6))
        self.assertTrue(torch.allclose(student.grad, original.grad, atol=2e-6))
        self.assertIsNone(teacher.grad)

    def test_empty_and_singleton_groups_preserve_raw_values_and_gradients(self):
        for ids in (None, (), (1,)):
            student = torch.randn(1, 3, 5, requires_grad=True)
            teacher = torch.randn_like(student)
            original = student.detach().clone().requires_grad_()
            expected = forward_kl_per_token(original, teacher).mean()
            expected.backward()
            result = forward_kl_loss(student, teacher, eos_token_ids=ids, chunk_size=2)
            result.backward()
            self.assertTrue(torch.allclose(result, expected, atol=2e-6))
            self.assertTrue(torch.allclose(student.grad, original.grad, atol=2e-6))

    def test_invalid_groups_are_rejected(self):
        logits = torch.ones(1, 2, 4)
        for ids in ((1, 1), (-1, 2), (1, 4), (True, 2), (1.0, 2)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                forward_kl_loss(logits, logits, eos_token_ids=ids)

    def test_extreme_bfloat16_logits_have_finite_loss_and_gradients(self):
        logits = torch.tensor(
            [[[10000, -10000, 20], [-10000, 10000, -20]]], dtype=torch.bfloat16
        ).requires_grad_()
        teacher = -logits.detach()
        loss = forward_kl_loss(logits, teacher, eos_token_ids=(0, 1), chunk_size=1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mode_is_part_of_experiment_identity(self):
        config = OPDConfig()
        self.assertNotEqual(
            experiment_signature(config, {}, {}, {}, {}),
            experiment_signature(replace(config, eos_loss_mode="semantic_stop"), {}, {}, {}, {}),
        )
        with self.assertRaises(ValueError):
            OPDConfig(eos_loss_mode="typo")

    def test_original_ending_ids_and_causal_offsets_survive_both_objectives(self):
        student = tiny_model().eval()
        teacher = tiny_model(77).requires_grad_(False).eval()
        for ending_id in (1, 5):
            row = trajectory(response=(3, 4, ending_id))
            student_ids = torch.tensor([[2, 3, 4, ending_id]])
            teacher_ids = torch.tensor([[6, 2, 3, 4, ending_id]])
            with torch.no_grad():
                student_logits = student(student_ids).logits[:, 0:3]
                teacher_logits = teacher(teacher_ids).logits[:, 1:4]
            for mode in ("raw_token", "semantic_stop"):
                with (
                    patch.object(student, "forward", wraps=student.forward) as student_forward,
                    patch.object(teacher, "forward", wraps=teacher.forward) as teacher_forward,
                ):
                    loss = trajectory_kl(
                        student, teacher, row, "cpu", None, tiny_config(eos_loss_mode=mode)
                    )
                self.assertTrue(
                    torch.equal(student_forward.call_args.kwargs["input_ids"], student_ids)
                )
                self.assertTrue(
                    torch.equal(teacher_forward.call_args.kwargs["input_ids"], teacher_ids)
                )
                expected = forward_kl_loss(
                    student_logits,
                    teacher_logits,
                    eos_token_ids=(1, 5) if mode == "semantic_stop" else None,
                )
                self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
                self.assertEqual(row["response_token_ids"][-1], ending_id)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


class IncrementalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "validation.json"
        self.model, self.tokenizer = tiny_model(), tiny_tokenizer()
        self.examples = [example(0), example(1)]
        self.policy = {
            "max_new_tokens": 3,
            "do_sample": False,
            "num_beams": 1,
            "eos_token_id": [1, 5],
            "pad_token_id": 1,
            "bos_token_id": 0,
        }

    def evaluate(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return evaluate_incrementally(
                self.model,
                self.tokenizer,
                self.examples,
                self.policy,
                "cpu",
                None,
                path=self.path,
                identity="tiny-checkpoint",
                **kwargs,
            )

    def test_interrupt_and_resume_match_uninterrupted_answers(self):
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args[2][0]["example_id"])
            if len(calls) == 2:
                raise RuntimeError("simulated kernel loss")
            return generate_evaluation(*args, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "kernel loss"):
            self.evaluate(generator=interrupted)
        partial = json.loads(self.path.with_suffix(".partial.json").read_text())
        self.assertEqual(len(partial["rows"]), 1)
        self.assertFalse(self.path.exists())
        remaining = []

        def resumed(*args, **kwargs):
            remaining.append(args[2][0]["example_id"])
            return generate_evaluation(*args, **kwargs)

        result = self.evaluate(generator=resumed)
        self.assertEqual(remaining, [self.examples[1]["example_id"]])
        with redirect_stdout(io.StringIO()):
            reference = generate_evaluation(
                self.model,
                self.tokenizer,
                self.examples,
                self.policy,
                "cpu",
                None,
                use_model_defaults=False,
            )
        self.assertEqual(result["rows"], reference["rows"])
        self.assertEqual(result["scores"], reference["scores"])
        self.assertEqual(
            self.evaluate(generator=lambda *a, **k: self.fail("cache regenerated")), result
        )

    def test_changed_policy_checkpoint_and_cached_tokens_are_rejected(self):
        self.evaluate()
        self.policy["max_new_tokens"] = 4
        with self.assertRaisesRegex(ValueError, "changed"):
            self.evaluate()
        self.policy["max_new_tokens"] = 3
        saved = json.loads(self.path.read_text())
        saved["rows"][0]["response_length"] = 999
        self.path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "tokens"):
            self.evaluate()

    def test_validation_does_not_consume_rollout_rng(self):
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()

        def consuming(*args, **kwargs):
            random.random()
            torch.rand(2)
            return generate_evaluation(*args, **kwargs)

        self.evaluate(generator=consuming)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_deadline_stops_before_generation(self):
        with self.assertRaises(WalltimePause):
            self.evaluate(budget=SlurmBudget(time.time() + 599))
        self.assertFalse(self.path.exists())


class DeadlineAndProbeTests(unittest.TestCase):
    def test_slurm_deadline_and_checkpoint_reserve(self):
        result = type("Result", (), {"stdout": "JobId=42 EndTime=2026-11-08T22:00:00"})()
        with (
            patch.dict("os.environ", {"SLURM_JOB_ID": "42"}),
            patch("subprocess.run", return_value=result) as call,
        ):
            budget = SlurmBudget.from_environment()
        self.assertIn("42", call.call_args.args[0])
        with patch("time.time", return_value=budget.end_time - 601):
            budget.require_time()
        with (
            patch("time.time", return_value=budget.end_time - 600),
            self.assertRaises(WalltimePause),
        ):
            budget.require_time()

    def test_unknown_slurm_deadline_is_not_ignored(self):
        result = type("Result", (), {"stdout": "JobId=42 EndTime=Unknown"})()
        with (
            patch.dict("os.environ", {"SLURM_JOB_ID": "42"}),
            patch("subprocess.run", return_value=result),
            self.assertRaises(RuntimeError),
        ):
            SlurmBudget.from_environment()

    def test_capped_boundary_is_after_all_original_ids(self):
        row = trajectory(response=(3, 4, 4)) | {"ended_with_eos": False, "hit_token_limit": True}
        points = probe_boundaries(row, tiny_tokenizer())
        after = next(point for point in points if "after_truncation" in point["boundaries"])
        self.assertEqual(after["response_prefix_length"], 3)
        self.assertIsNone(after["observed_next_token_id"])
        natural = probe_boundaries(trajectory(), tiny_tokenizer())
        before = next(point for point in natural if "before_observed_eos" in point["boundaries"])
        self.assertEqual(before["response_prefix_length"], 2)
        self.assertEqual(before["observed_next_token_id"], 1)

    def test_stopping_probe_distinguishes_native_tokens_from_total_stop(self):
        logits = torch.tensor([[0.79, 0.01, 0.2]]).log()
        teacher = torch.tensor([[0.01, 0.79, 0.2]]).log()
        record = stopping_statistics(logits, teacher, eos_token_ids=(0, 1), temperature=1)[0]
        self.assertAlmostEqual(record["student_stop_probability"], 0.8, places=6)
        self.assertAlmostEqual(record["teacher_T1_stop_probability"], 0.8, places=6)
        self.assertAlmostEqual(record["semantic_stop_kl_nats"], 0, places=6)
        self.assertGreater(record["raw_student_eos_logit_gradients"]["0"], 0)
        self.assertAlmostEqual(record["grouped_student_eos_logit_gradients"]["0"], 0, places=6)
        self.assertAlmostEqual(
            record["raw_eos_contribution_nats"] + record["ordinary_token_contribution_nats"],
            record["raw_token_kl_nats"],
            places=6,
        )
        self.assertAlmostEqual(
            record["grouped_eos_contribution_nats"] + record["ordinary_token_contribution_nats"],
            record["semantic_stop_kl_nats"],
            places=6,
        )
        expected_entropy = -(logits.exp() * logits).sum().item()
        self.assertAlmostEqual(record["student_entropy_nats"], expected_entropy, places=6)

    def test_probe_selection_is_deterministic_and_covers_the_three_groups(self):
        fixed = [trajectory(i) for i in range(8)]
        early = [trajectory(i, policy_step=0) for i in range(5)]
        late = [
            trajectory(i, response=(3, 4, 4), policy_step=21)
            | {"ended_with_eos": False, "hit_token_limit": True}
            for i in range(5)
        ]
        selected = select_probe_trajectories(fixed, list(reversed(early + late)))
        self.assertEqual(len(selected), 16)
        self.assertEqual(
            [row["example_id"] for row in selected[8:12]], [row["example_id"] for row in early[:4]]
        )
        self.assertEqual(selected[-1]["probe_group"], "late_capped")

    def test_pause_status_is_durable_before_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            record = publish_pause(
                directory,
                arm="on_policy",
                status="health_paused",
                step=22,
                snapshot=Path(directory) / "step_0022",
                signature="tiny",
                config=tiny_config(),
                health={"pause": True},
                reason="cap guard",
            )
            self.assertEqual(
                json.loads((Path(directory) / "manifest.json").read_text()),
                json.loads(json.dumps(record)),
            )
            self.assertEqual(record["validation_status"], "pending")


class ActualComparisonTests(unittest.TestCase):
    def fixture(self, directory, **overrides):
        config = tiny_config(
            train_examples=8,
            accumulation_steps=2,
            pilot_updates=2,
            pilot_check_interval=2,
            checkpoint_interval=2,
            validation_interval=2,
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            diagnostic_examples=1,
            cap_window_responses=100,
            **overrides,
        )
        return fixtures.StagedTrainingTests().namespace(directory, config), config

    def test_two_actual_pilots_have_independent_checkpoints_and_same_initial_rollouts(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace, config = self.fixture(root)
            teacher = tiny_model(77).requires_grad_(False).eval()
            before = {key: value.clone() for key, value in teacher.state_dict().items()}
            result = run_eos_comparison(
                namespace["train_arm"],
                teacher,
                directory=root,
                config=config,
                protocol={},
                splits={},
                checkpoint_hashes={},
                evaluation_policy=namespace["evaluation_policy"],
                code_hashes={"fixture": "random-model"},
                starting_validation_nll=4,
            )
            self.assertEqual(set(result["results"]), {"raw_token", "semantic_stop"})
            self.assertTrue(result["gates"]["initial_rollout_groups_identical"])
            self.assertNotEqual(
                result["gates"]["arm_signatures"]["raw_token"],
                result["gates"]["arm_signatures"]["semantic_stop"],
            )
            for mode, record in result["results"].items():
                self.assertEqual(record["completed_steps"], 2)
                self.assertEqual(record["validation_status"], "completed")
                self.assertTrue(
                    (root / "eos_comparison" / mode / "resume/on_policy/step_0002/student").is_dir()
                )
                self.assertEqual(record["configuration"]["eos_loss_mode"], mode)
                progress_path = root / "eos_comparison" / mode / "progress.jsonl"
                last_entry = json.loads(progress_path.read_text().splitlines()[-1])
                self.assertEqual(last_entry["update"], 2)
                self.assertEqual(last_entry["stage"], "validation")
                self.assertEqual(last_entry["event"], "completed")
            paths = [
                root / "eos_comparison" / mode / "training_trajectories.jsonl"
                for mode in ("raw_token", "semantic_stop")
            ]
            first = [
                [json.loads(line) for line in path.read_text().splitlines()][:2] for path in paths
            ]
            self.assertEqual(first[0], first[1])
            self.assertTrue(
                all(torch.equal(value, before[key]) for key, value in teacher.state_dict().items())
            )
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_progress_is_visible_when_teacher_scoring_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace, _ = self.fixture(root)

            def interrupt_scoring(*args, **kwargs):
                # Read the file while the stage is still running. A buffered
                # message that appears only after cleanup would miss a killed job.
                rows = [
                    json.loads(line) for line in (root / "progress.jsonl").read_text().splitlines()
                ]
                self.assertEqual(
                    [(r["update"], r["stage"], r["event"]) for r in rows],
                    [
                        (1, "student_generation", "started"),
                        (1, "teacher_scoring_and_backward", "started"),
                    ],
                )
                raise RuntimeError("simulated scoring interruption")

            namespace["backward_opd_group"] = interrupt_scoring
            with self.assertRaisesRegex(RuntimeError, "simulated scoring interruption"):
                namespace["train_arm"](
                    "on_policy", tiny_model(77).requires_grad_(False), stop_after_steps=2
                )
            self.assertFalse((root / "training_trajectories.jsonl").exists())

    def test_walltime_pause_saves_optimizer_boundary_before_validation(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace, config = self.fixture(root)

            class Budget:
                calls = 0

                def require_time(self):
                    self.calls += 1
                    if self.calls > 1:
                        raise WalltimePause("simulated walltime")

            with patch.object(SlurmBudget, "from_environment", return_value=Budget()):
                result = namespace["train_arm"](
                    "on_policy", tiny_model(77).requires_grad_(False), stop_after_steps=2
                )
            self.assertEqual(result["status"], "walltime_paused")
            self.assertEqual(result["completed_steps"], 1)
            self.assertEqual(result["validation_status"], "pending")
            self.assertTrue((root / "resume/on_policy/step_0001/resume.json").is_file())

    def test_pilot_status_is_published_before_slow_boundary_validation(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace, config = self.fixture(root)
            original_evaluate = namespace["evaluate_and_save"]
            original_release = namespace["release_cuda_memory"]
            cleanup_states = []

            def check_cleanup():
                # The pause must already be durable before optimizer cleanup,
                # which happens before the slower validation pass.
                cleanup_states.append(json.loads((root / "manifest.json").read_text()))
                original_release()

            def check_manifest(*args, **kwargs):
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "pilot_complete")
                self.assertEqual(manifest["completed_steps"], 2)
                self.assertEqual(manifest["validation_status"], "pending")
                return original_evaluate(*args, **kwargs)

            namespace["evaluate_and_save"] = check_manifest
            namespace["release_cuda_memory"] = check_cleanup
            result = namespace["train_arm"](
                "on_policy", tiny_model(77).requires_grad_(False), stop_after_steps=2
            )
            self.assertEqual(result["validation_status"], "completed")
            self.assertEqual(cleanup_states[0]["status"], "pilot_complete")
            self.assertEqual(cleanup_states[0]["validation_status"], "pending")

    def test_health_paused_validation_recovers_without_new_updates(self):
        config = tiny_config(
            train_examples=8,
            accumulation_steps=2,
            pilot_updates=4,
            pilot_check_interval=3,
            checkpoint_interval=3,
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            diagnostic_examples=1,
            cap_window_responses=4,
            cap_pause_fraction=1,
        )
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = fixtures.StagedTrainingTests().namespace(root, config)
            original_evaluation = namespace["evaluate_and_save"]
            namespace["sample_trajectory"] = lambda model, tokenizer, row, **kwargs: (
                trajectory(row["source_index"], (3, 4, 4), kwargs["policy_step"])
                | {"ended_with_eos": False, "hit_token_limit": True}
            )
            namespace["evaluate_and_save"] = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("validation interrupted")
            )
            teacher = tiny_model(77).requires_grad_(False).eval()
            with self.assertRaisesRegex(RuntimeError, "validation interrupted"):
                namespace["train_arm"]("on_policy", teacher, stop_after_steps=4)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "health_paused")
            self.assertEqual(manifest["completed_steps"], 2)
            self.assertEqual(manifest["validation_status"], "pending")
            original_log = (root / "training_trajectories.jsonl").read_bytes()
            namespace["evaluate_and_save"] = original_evaluation
            result = namespace["train_arm"](
                "on_policy", teacher, stop_after_steps=4, recover_pause_validation=True
            )
            self.assertEqual(result["status"], "health_paused")
            self.assertEqual(result["validation_status"], "completed")
            self.assertEqual((root / "training_trajectories.jsonl").read_bytes(), original_log)
            with self.assertRaisesRegex(RuntimeError, "health-paused"):
                namespace["train_arm"]("on_policy", teacher, stop_after_steps=4)

    def test_engineering_gates_require_a_measured_contrast_and_complete_validation(self):
        def result(mode):
            return {
                "status": "pilot_complete",
                "completed_steps": 32,
                "validation_status": "completed",
                "history": [
                    {
                        "step": i,
                        "response_limit_flags": [mode == "raw_token"] * 8,
                        "loss": 1.0,
                        "grad_norm_before_clipping": 1.0,
                    }
                    for i in range(1, 33)
                ],
                "final_validation_reference_nll": 0.7,
                "final_validation": {"strict_accuracy": 0.5},
            }

        results = {mode: result(mode) for mode in ("raw_token", "semantic_stop")}
        self.assertTrue(comparison_gates(results, OPDConfig(), 0.667)["ready_for_full_run"])
        results["raw_token"] = result("semantic_stop")
        self.assertFalse(comparison_gates(results, OPDConfig(), 0.667)["ready_for_full_run"])
        results["semantic_stop"]["validation_status"] = "pending"
        self.assertEqual(comparison_gates(results, OPDConfig(), 0.667)["status"], "incomplete")


class ProbeAndEvidenceRoundTripTests(unittest.TestCase):
    def test_fresh_colab_control_setup_needs_only_03_inputs(self):
        from src.pipeline import resolve_profile_cold_start

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "Notebook 03"):
                resolve_profile_cold_start(
                    Path(folder),
                    {"student_model_id": "new-student", "student_revision": "new-revision"},
                )

    def test_fresh_colab_probe_runs_without_historical_snapshots(self):
        from src.opd_diagnostics import probe_checkpoint_set

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            namespace, config = ActualComparisonTests().fixture(root)
            teacher = tiny_model(77).requires_grad_(False).eval()
            records = [dict(trajectory(), probe_group="fixed_validation")]
            result = probe_checkpoint_set(
                teacher,
                namespace["tokenizer"],
                records,
                {"starting_03": namespace["starting_checkpoint_directory"]},
                "cpu",
                None,
                config,
            )
            self.assertEqual({row["checkpoint"] for row in result["rows"]}, {"starting_03"})
            self.assertFalse((root / "historical_step_0022_metrics.json").exists())

    def test_historical_history_and_sampled_ids_are_checked_before_reuse(self):
        from src.opd import append_trajectories
        from src.utils import write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config(train_examples=44, accumulation_steps=2)
            splits = {"train": [example(i) for i in range(44)]}
            splits.update(validation=[], primary_test=[], historical_test=[])
            history = [{"step": i, "loss": 1.0} for i in range(1, 23)]
            write_json(
                root / "experiment.json",
                {
                    "signature": "synthetic-history",
                    "configuration": {"accumulation_steps": 2},
                    "protocol": {},
                    "starting_checkpoint_hashes": {},
                },
            )
            write_json(root / "data_splits.json", splits)
            write_json(root / "on_policy_history.json", history)
            write_json(
                root / "sft_control_training.json",
                {"status": "completed", "checkpoint_reload_verified": True},
            )
            log = root / "training_trajectories.jsonl"
            append_trajectories(log, [trajectory(i, policy_step=i // 2) for i in range(44)])

            def snapshot(path, **kwargs):
                step = int(path.name.split("_")[1])
                return {"completed_steps": step, "history": history[:step]}

            with (
                patch("src.opd_experiment.read_resume", side_effect=snapshot),
                patch("src.opd_experiment.verify_inventory", return_value={}),
            ):
                result = verify_historical_run(root, {}, {}, splits, config)
                self.assertEqual(result["verified_trajectory_count"], 44)
                damaged = json.loads(json.dumps(history))
                damaged[-1]["loss"] = 2.0
                write_json(root / "on_policy_history.json", damaged)
                with self.assertRaisesRegex(ValueError, "optimizer boundary"):
                    verify_historical_run(root, {}, {}, splits, config)
                write_json(root / "on_policy_history.json", history)
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                rows[-1]["complete_sequence_ids"][-1] = 9
                log.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaisesRegex(ValueError, "sampled response"):
                    verify_historical_run(root, {}, {}, splits, config)

    def test_final_answer_boundary_uses_original_ids_without_encoding(self):
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return "".join(
                    {3: "Reason.\n", 4: "#### ", 5: "10", 6: "\nContinue", 1: ""}[item]
                    for item in ids
                )

            def encode(self, *args, **kwargs):
                raise AssertionError("probe must not retokenize")

        row = trajectory(response=(3, 4, 5, 6)) | {"ended_with_eos": False, "hit_token_limit": True}
        points = probe_boundaries(row, Tokenizer())
        boundary = next(
            point for point in points if "after_first_final_answer_line" in point["boundaries"]
        )
        self.assertEqual(boundary["response_prefix_length"], 3)
        self.assertEqual(boundary["observed_next_token_id"], 6)

    def test_actual_frozen_probe_forwards_preserve_weights_and_prefix_positions(self):
        from src.opd_diagnostics import probe_checkpoint_set
        from src.sft import save_checkpoint

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "student"
            tokenizer = tiny_tokenizer()
            save_checkpoint(tiny_model(), tokenizer, path)
            teacher = tiny_model(77).requires_grad_(False).eval()
            before = {key: value.clone() for key, value in teacher.state_dict().items()}
            row = trajectory() | {"probe_group": "tiny-fixture"}
            result = probe_checkpoint_set(
                teacher, tokenizer, [row], {"tiny_checkpoint": path}, "cpu", None, tiny_config()
            )
            ending = next(
                record for record in result["rows"] if "before_observed_eos" in record["boundaries"]
            )
            self.assertEqual(ending["student_predicting_position"], 2)
            self.assertEqual(ending["teacher_predicting_position"], 3)
            self.assertTrue(
                all(torch.equal(before[key], value) for key, value in teacher.state_dict().items())
            )
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_collector_verifies_source_weights_but_exports_only_small_evidence(self):
        import zipfile

        from scripts.collect_opd_evidence import collect
        from src.opd import save_resume
        from src.sft import save_checkpoint
        from src.utils import verify_inventory, write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            tokenizer = tiny_tokenizer()
            model = tiny_model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            scaler = torch.amp.GradScaler("cuda", enabled=False)
            for name in (
                "data_splits.json",
                "fixed_validation_trajectories.json",
                "sft_control_training.json",
                "on_policy_history.json",
            ):
                write_json(source / name, {"fixture": "synthetic-test-only"})
            write_json(source / "experiment.json", {"signature": "synthetic-source"})
            (source / "training_trajectories.jsonl").write_text("{}\n")
            for step in (16, 22):
                save_resume(
                    source,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    arm="on_policy",
                    step=step,
                    signature="synthetic-source",
                    history=[{"step": i} for i in range(1, step + 1)],
                )
            save_checkpoint(model, tokenizer, source / "checkpoints/sft_control")
            archive = collect(source, root / "collected")
            verify_inventory(root / "collected")
            with zipfile.ZipFile(archive) as zipped:
                self.assertTrue(any("step_0022/resume.json" in name for name in zipped.namelist()))
                self.assertFalse(
                    any(name.endswith((".pt", ".safetensors")) for name in zipped.namelist())
                )
            (source / "checkpoints/sft_control/model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                collect(source, root / "corrupt_collection")

    def test_comparison_can_be_reopened_with_json_tuple_configuration(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = tiny_config(
                train_examples=4,
                accumulation_steps=2,
                pilot_updates=1,
                max_prompt_length=8,
                max_response_length=3,
                max_sequence_length=16,
                cap_window_responses=100,
            )
            namespace = fixtures.StagedTrainingTests().namespace(root, config)
            teacher = tiny_model(77).requires_grad_(False).eval()
            arguments = dict(
                directory=root,
                config=config,
                protocol={},
                splits={},
                checkpoint_hashes={},
                evaluation_policy=namespace["evaluation_policy"],
                code_hashes={},
                starting_validation_nll=4,
            )
            first = run_eos_comparison(namespace["train_arm"], teacher, **arguments)
            logs = {
                mode: (root / "eos_comparison" / mode / "training_trajectories.jsonl").read_bytes()
                for mode in first["results"]
            }
            second = run_eos_comparison(namespace["train_arm"], teacher, **arguments)
            self.assertEqual(set(second["results"]), set(first["results"]))
            for mode, content in logs.items():
                self.assertEqual(
                    (root / "eos_comparison" / mode / "training_trajectories.jsonl").read_bytes(),
                    content,
                )

    def test_comparison_report_contains_both_arms_and_excludes_tensors(self):
        import zipfile

        from src.opd import export_report
        from src.utils import write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("raw_token", "semantic_stop"):
                child = root / "eos_comparison" / mode
                child.mkdir(parents=True)
                write_json(child / "on_policy_history.json", [{"fixture": "synthetic"}])
                (child / "model.safetensors").write_bytes(b"excluded")
            archive = export_report(
                fixtures.ROOT, root, {"signature": "synthetic", "status": "comparison_complete"}
            )
            with zipfile.ZipFile(archive) as zipped:
                for mode in ("raw_token", "semantic_stop"):
                    self.assertIn(
                        f"eos_comparison/{mode}/on_policy_history.json", zipped.namelist()
                    )
                self.assertFalse(any(name.endswith(".safetensors") for name in zipped.namelist()))


if __name__ == "__main__":
    unittest.main()
