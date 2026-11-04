"""Synthetic rollout audits, independent of CUDA and pretrained model weights."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_opd_run import audit_run


def rollout(response, *, policy_step=0):
    ended = response[-1] in {151643, 151645}
    return {
        "example_id": "synthetic/0",
        "policy_step": policy_step,
        "prompt_token_ids": [12],
        "prompt_length": 1,
        "response_token_ids": response,
        "response_length": len(response),
        "complete_sequence_ids": [12] + response,
        "ended_with_eos": ended,
        "hit_token_limit": len(response) == 8 and not ended,
        "response_text": "Illustrative text only",
    }


class RolloutAuditTests(unittest.TestCase):
    def test_eos_at_limit_is_not_a_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [rollout([3] * 8), rollout([3] * 7 + [151645])]
            (root / "training_trajectories.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            report = audit_run(root, response_limit=8)
            self.assertEqual(report["trajectory_count"], 2)
            self.assertEqual(report["updates"][0]["token_limit_stops"], 1)
            self.assertEqual(report["updates"][0]["eos_stops"], 1)
            self.assertEqual(report["trajectory_inconsistencies"], [])
            self.assertGreater(
                report["most_repetitive_responses"][0]["repeated_four_token_fraction"], 0
            )

    def test_incorrect_metadata_and_history_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = rollout([3, 151643])
            row["ended_with_eos"] = False
            (root / "training_trajectories.jsonl").write_text(json.dumps(row) + "\n")
            (root / "on_policy_history.json").write_text(
                json.dumps([{"step": 1, "response_tokens": 900, "token_limit_stops": 0}])
            )
            report = audit_run(root, response_limit=8)
            self.assertEqual(len(report["trajectory_inconsistencies"]), 1)
            self.assertEqual(report["history_count_mismatches"], [1])

    def test_unfinished_append_is_explicit_and_inputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            text = json.dumps(rollout([3, 151643])) + '\n{"example_id":'
            path = root / "training_trajectories.jsonl"
            path.write_text(text)
            report = audit_run(root, response_limit=8)
            self.assertTrue(report["incomplete_last_line"])
            self.assertEqual(report["trajectory_count"], 1)
            self.assertEqual(path.read_text(), text)

    def test_teacher_probe_preserves_responses_and_model_state(self):
        from types import SimpleNamespace

        import torch

        from scripts.probe_opd_teacher import probe_teacher_context

        class FakeTokenizer:
            chat_template = "synthetic template"

            def decode(self, ids, **kwargs):
                return str(ids)

            def apply_chat_template(self, messages, **kwargs):
                self.options = kwargs
                return [8, 9]

        class FakeTeacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                self.calls = []

            def forward(self, input_ids, attention_mask, use_cache):
                self.calls.append((input_ids.tolist(), torch.is_grad_enabled(), self.training))
                # Synthetic vocabulary includes the real two Qwen stop-token IDs.
                logits = torch.zeros(1, input_ids.shape[1], 151646)
                logits[..., 151643] = 4
                return SimpleNamespace(logits=logits)

        model, tokenizer = FakeTeacher().train(), FakeTokenizer()
        row = rollout([3, 151643])
        result = probe_teacher_context(model, tokenizer, row, None)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(tokenizer.options["enable_thinking"])
        self.assertEqual(model.calls[0][0], [[12, 3, 151643]])
        self.assertEqual(model.calls[1][0], [[8, 9, 3, 151643]])
        self.assertTrue(all(not grad and not training for _, grad, training in model.calls))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(0 <= r["stop_probability_T1"] <= 1 for r in result))
        self.assertTrue(
            all(r["stop_target_probability"] > r["stop_probability_T1"] for r in result)
        )

    def test_malformed_complete_line_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "training_trajectories.jsonl").write_text("bad json\n")
            with self.assertRaises(json.JSONDecodeError):
                audit_run(root)


if __name__ == "__main__":
    unittest.main()
