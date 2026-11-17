"""Guard regressions use synthetic records and tiny CPU models."""

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_opd as fixtures
import torch
from transformers import get_linear_schedule_with_warmup

from scripts.collect_opd_evidence import collect
from src.opd import optimizer_update, read_resume, save_resume
from src.training_safety import (
    TrainingSafetyConfig,
    accept_checkpoint,
    assess_training_safety,
    restore_boundary_measurements,
    save_boundary_measurements,
)
from src.utils import write_inventory, write_json


class GuardThresholdTests(unittest.TestCase):
    def test_actual_pilots_would_pause_at_update_eight(self):
        directory = (
            fixtures.ROOT
            / "notebooks/assets/04_on_policy_distillation/recorded_eos_comparison/audited"
        )
        summary = json.loads((directory / "audit_summary.json").read_text())
        for mode in ("raw_token", "semantic_stop"):
            history = json.loads((directory / f"{mode}_history.json").read_text())
            pauses = [
                row["step"]
                for index, row in enumerate(history)
                if assess_training_safety(
                    history[: index + 1],
                    summary["baseline_nll"],
                    summary["baseline_accuracy"],
                    TrainingSafetyConfig(),
                )["pause"]
            ]
            self.assertEqual(pauses[0], 8)

    def test_exact_thresholds_pass_and_crossings_pause(self):
        policy = TrainingSafetyConfig()
        record = {
            "step": 8,
            "response_limit_flags": [True] * 8 + [False] * 24,
            "validation_reference_nll": 1.1,
            "generated_validation": {"strict_accuracy": 0.75},
        }
        self.assertFalse(assess_training_safety([record], 1.0, 0.8, policy)["pause"])
        for field, value, metric in (
            ("validation_reference_nll", 1.10001, "validation_reference_nll"),
            ("generated_validation", {"strict_accuracy": 0.7499}, "strict_accuracy"),
            ("response_limit_flags", [True] * 9 + [False] * 23, "recent_cap_fraction"),
        ):
            with self.subTest(field=field):
                result = assess_training_safety([{**record, field: value}], 1.0, 0.8, policy)
                self.assertTrue(result["pause"])
                self.assertIn(metric, [r["metric"] for r in result["reasons"]])

    def test_reported_grouped_pilot_is_rejected(self):
        history = [
            {
                "step": 32,
                "response_limit_flags": [True] * 19 + [False] * 13,
                "validation_reference_nll": 0.963979733522468,
                "generated_validation": {"strict_accuracy": 23 / 64},
            }
        ]
        result = assess_training_safety(
            history, 0.6676512348477289, 33 / 64, TrainingSafetyConfig()
        )
        self.assertEqual(len(result["reasons"]), 3)

    def test_partial_window_and_missing_generated_metrics(self):
        policy = TrainingSafetyConfig()
        self.assertFalse(
            assess_training_safety([{"response_limit_flags": [True] * 31}], 1.0, 0.5, policy)[
                "pause"
            ]
        )
        self.assertTrue(
            assess_training_safety([{"validation_reference_nll": 1.2}], 1.0, 0.5, policy)["pause"]
        )

    def test_nonfinite_metrics_and_invalid_limits_fail(self):
        with self.assertRaises(FloatingPointError):
            assess_training_safety(
                [{"validation_reference_nll": float("nan")}], 1.0, 0.5, TrainingSafetyConfig()
            )
        for changes in (
            {"max_nll_ratio": 1.0},
            {"max_accuracy_drop": -1.0},
            {"response_window": 0},
        ):
            with self.assertRaises(ValueError):
                TrainingSafetyConfig(**changes)


class GuardedLoopTests(unittest.TestCase):
    def test_supervised_control_keeps_its_existing_policy_and_pointer(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = self.fixture(root)
            result = namespace["train_arm"]("sft_control", stop_after_steps=1)
            self.assertEqual(result["status"], "pilot_complete")
            self.assertFalse((root / "last_acceptable.json").exists())

    def test_report_packaging_failure_preserves_quality_pause(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = self.fixture(root)

            def fail_export(*args, **kwargs):
                raise OSError("simulated full disk during ZIP creation")

            namespace["export_report"] = fail_export
            result = namespace["train_arm"](
                "on_policy", fixtures.tiny_model(77).requires_grad_(False), stop_after_steps=4
            )
            self.assertEqual(result["status"], "quality_paused")
            self.assertEqual(result["validation_status"], "completed")
            self.assertTrue(Path(result["resume_snapshot"]).is_dir())
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["status"], "quality_paused"
            )

    def test_mid_group_deadline_restores_boundary_rng(self):
        from src.opd_experiment import WalltimePause

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = self.fixture(root)
            namespace["config"] = replace(namespace["config"], pilot_check_interval=4)
            namespace["validation_loss"] = lambda *args, **kwargs: 4.0
            original = namespace["sample_trajectory"]
            boundary_rng = []

            def interrupt(model, tokenizer, example, **kwargs):
                if kwargs["policy_step"] == 1:
                    boundary_rng.append(torch.get_rng_state().clone())
                    torch.rand(17)
                    raise WalltimePause("simulated mid-group deadline")
                return original(model, tokenizer, example, **kwargs)

            namespace["sample_trajectory"] = interrupt
            result = namespace["train_arm"](
                "on_policy",
                fixtures.tiny_model(77).requires_grad_(False),
                stop_after_steps=4,
            )
            self.assertEqual(result["status"], "walltime_paused")
            self.assertEqual(result["completed_steps"], 1)
            state = torch.load(
                Path(result["resume_snapshot"]) / "training_state.pt", weights_only=True
            )
            self.assertTrue(torch.equal(state["torch_rng"], boundary_rng[0]))
            self.assertTrue(torch.equal(torch.get_rng_state(), boundary_rng[0]))

    def fixture(self, root):
        config = fixtures.tiny_config(
            train_examples=8,
            accumulation_steps=2,
            pilot_check_interval=1,
            pilot_updates=4,
            max_prompt_length=8,
            max_response_length=3,
            max_sequence_length=16,
            cap_window_responses=100,
        )
        namespace = fixtures.StagedTrainingTests().namespace(root, config)
        namespace.update(
            PROJECT_ROOT=fixtures.ROOT,
            safety_policy=TrainingSafetyConfig(response_window=100),
            starting_validation={"scores": {"strict": {"accuracy": 0.0}}},
            validation_loss=lambda *args, **kwargs: 4.8,
        )
        return namespace

    def test_quality_pause_prevents_next_update_and_recovers_validation(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            namespace = self.fixture(root)
            teacher = fixtures.tiny_model(77).requires_grad_(False).eval()
            result = namespace["train_arm"]("on_policy", teacher, stop_after_steps=4)
            self.assertEqual(result["status"], "quality_paused")
            self.assertEqual(result["completed_steps"], 1)
            self.assertEqual(result["validation_status"], "completed")
            self.assertTrue(list(root.glob("report_quality_paused_*.zip")))
            accepted = json.loads((root / "last_acceptable.json").read_text())
            self.assertEqual(accepted["step"], 0)
            trajectory_bytes = (root / "training_trajectories.jsonl").read_bytes()
            namespace["validation_loss"] = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Completed measurements were recomputed")
            )
            recovered = namespace["train_arm"](
                "on_policy",
                teacher,
                stop_after_steps=4,
                recover_pause_validation=True,
            )
            self.assertEqual(recovered["completed_steps"], 1)
            self.assertEqual((root / "training_trajectories.jsonl").read_bytes(), trajectory_bytes)
            with self.assertRaisesRegex(RuntimeError, "health-paused"):
                namespace["train_arm"]("on_policy", teacher, stop_after_steps=4)

    def test_final_boundary_quality_failure_cannot_be_pilot_complete(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            namespace = self.fixture(Path(directory))
            result = namespace["train_arm"](
                "on_policy",
                fixtures.tiny_model(77).requires_grad_(False),
                stop_after_steps=1,
            )
            self.assertEqual(result["status"], "quality_paused")

    def test_post_update_nonfinite_parameters_stop_scheduler(self):
        student = fixtures.tiny_model()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        for parameter in student.parameters():
            parameter.grad = torch.ones_like(parameter)
        previous = scheduler.last_epoch

        def corrupt():
            with torch.no_grad():
                next(student.parameters()).fill_(float("inf"))

        with patch.object(optimizer, "step", side_effect=corrupt):
            with self.assertRaises(FloatingPointError):
                optimizer_update(student, optimizer, scheduler, scaler, 1.0)
        self.assertEqual(scheduler.last_epoch, previous)


class RecoveryAndCollectionTests(unittest.TestCase):
    def state(self):
        student = fixtures.tiny_model()
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, 8)
        return (
            student,
            fixtures.tiny_tokenizer(),
            optimizer,
            scheduler,
            torch.amp.GradScaler("cuda", enabled=False),
        )

    def test_retention_pins_last_acceptable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.state()
            for step in range(1, 5):
                snapshot = save_resume(
                    root,
                    *state,
                    arm="on_policy",
                    step=step,
                    signature="fixed",
                    history=[{"step": i} for i in range(1, step + 1)],
                )
                if step == 1:
                    accept_checkpoint(
                        root,
                        snapshot,
                        {
                            "step": 1,
                            "generated_validation": {"strict_accuracy": 0.8},
                            "validation_reference_nll": 1.0,
                        },
                        "fixed",
                        TrainingSafetyConfig(),
                    )
            self.assertEqual(
                [p.name for p in sorted((root / "resume/on_policy").glob("step_*"))],
                ["step_0001", "step_0003", "step_0004"],
            )
            read_resume(root / "resume/on_policy/step_0001", arm="on_policy", signature="fixed")

    def test_boundary_metrics_require_same_snapshot_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "weight").write_bytes(b"fixture")
            write_inventory(snapshot)
            policy = TrainingSafetyConfig()
            record = {"step": 1, "validation_reference_nll": 1.2}
            save_boundary_measurements(root, record, snapshot, "fixed", policy, complete=False)
            history = [{"step": 1}]
            restore_boundary_measurements(root, history, snapshot, "fixed", policy)
            self.assertEqual(history[-1]["validation_reference_nll"], 1.2)
            with self.assertRaises(ValueError):
                restore_boundary_measurements(
                    root, [{"step": 1}], snapshot, "fixed", replace(policy, max_nll_ratio=1.2)
                )

    def test_collector_discovers_comparison_steps_without_sft_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            write_json(run / "experiment.json", {"signature": "parent"})
            for mode, step in (("raw_token", 18), ("semantic_stop", 32)):
                child = run / "eos_comparison" / mode
                child.mkdir(parents=True)
                write_json(child / "experiment.json", {"signature": mode})
                save_resume(
                    child,
                    *self.state(),
                    arm="on_policy",
                    step=step,
                    signature=mode,
                    history=[{"step": i} for i in range(1, step + 1)],
                )
                write_json(child / "on_policy_pause.json", {"completed_steps": step})
            archive = collect(run, root / "evidence")
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertTrue(any("step_0018/resume.json" in name for name in names))
                self.assertTrue(any("step_0032/resume.json" in name for name in names))
                self.assertFalse(any(name.endswith((".pt", ".safetensors")) for name in names))
                self.assertEqual(
                    json.loads(bundle.read("collection.json"))["layout"], "eos_comparison"
                )


if __name__ == "__main__":
    unittest.main()
