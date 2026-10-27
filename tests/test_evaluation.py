"""Tests for reference-blind extraction, metric separation, and immutable rescoring."""

import copy
import hashlib
import json
import math
import tempfile
import unittest
import zipfile
from decimal import Decimal
from pathlib import Path

from src.evaluation import (
    categorical_kl,
    extract_prose_final_answer,
    extract_strict_final_answer,
    load_saved_baseline,
    rescore_response_rows,
    save_rescore_report,
)


class AnswerExtractionTests(unittest.TestCase):
    def test_clear_answer_styles(self):
        examples = {
            "#### 42": "42",
            "Reasoning: 3 + 4 = 7.\nFinal answer: 7": "7",
            "The final answer is $1,234.00.": "1234.00",
            "Therefore, the final answer is -2.5.": "-2.5",
            "Final answer:\n21 years": "21",
            "Final answer:\nThe shop has 8 books left.": "8",
            "So, each child gets 5 books.": "5",
            "So, the class spends 12 hours on TV and reading.": "12",
            "Therefore, the total cost is **\\$93**.": "93",
            "7. The total amount is 3 + 6 + 20 = 29 liters.": "29",
            "The number of students is 60.": "60",
            "The first calculation is 63 + 21. The answer is 21.": "21",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                extracted = extract_prose_final_answer(text)
                self.assertEqual(Decimal(extracted["answer"]), Decimal(expected))
                self.assertTrue(extracted["evidence"])

    def test_intermediate_ambiguous_and_malformed_numbers_are_rejected(self):
        for text in [
            "",
            "There are 42 pencils in the question.",
            "Step 1: 2 + 3 = 5",
            "42",
            "The answer might be 42.",
            "Final answer: 5 or 7",
            "Final answer: between 5 and 7",
            "Final answer: 5/7",
            "Final answer: 2**3",
            "Final answer: 5**2**3",
            "Final answer: 1,23",
            "Final answer: NaN",
            "Final answer: infinity",
            "Final answer: 1e3",
            "Final answer: 5 is wrong",
            "Final answer: 5\nFinal answer: 6",
            "So, the total is 5.\nTherefore, the total is 6.",
            "Final answer:\n5\nFinal answer:\n6",
            "#### 5\n#### 6",
            "#### 5\nFinal answer: 6",
            "Final answer: 5\n#### 6",
            "Final answer: 5\nQuestion: What is 3 + 4?",
            "We got 5.\nNow calculate 6 + 7 = 13.",
            "So, the total is 1,23.",
            "So, 3 + 4 = 7.",
            "So, the answer isn't 5.",
            "So, she spends 12 hours in 4 weeks.",
            "So, the total is 5/",
            "So, the total is 5-",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(extract_prose_final_answer(text)["answer"])

    def test_budget_stops_do_not_recover_numbers_from_unfinished_reasoning(self):
        extracted = extract_prose_final_answer("Final answer: 66", hit_token_limit=True)
        self.assertIsNone(extracted["answer"])
        self.assertEqual(extracted["reason"], "token_limit_stop")

    def test_reference_changes_only_correctness_not_extraction(self):
        row = {
            "example_id": "a",
            "response_text": "Final answer: 7",
            "expected_answer": "7",
        }
        good, _ = rescore_response_rows([row])
        bad, _ = rescore_response_rows([{**row, "expected_answer": "9"}])
        self.assertEqual(good[0]["prose_answer"], bad[0]["prose_answer"])
        self.assertTrue(good[0]["prose_correct"])
        self.assertFalse(bad[0]["prose_correct"])

    def test_strict_metric_stays_independent_and_all_rows_count(self):
        rows = [
            {
                "example_id": "a",
                "response_text": "Final answer: 7",
                "expected_answer": "7",
            },
            {"example_id": "b", "response_text": "#### 8", "expected_answer": "9"},
            {
                "example_id": "c",
                "response_text": "Step 1: 2 + 3 = 5",
                "expected_answer": "5",
            },
            {
                "example_id": "d",
                "response_text": "So, the answer is 5",
                "expected_answer": "5",
                "hit_token_limit": True,
            },
        ]
        original = copy.deepcopy(rows)
        _, summary = rescore_response_rows(rows)
        self.assertEqual(rows, original)
        self.assertEqual(summary["strict"]["accuracy"], 0)
        self.assertEqual(summary["prose"]["accuracy"], 1 / 4)
        self.assertEqual(summary["prose"]["parse_rate"], 2 / 4)
        self.assertEqual(summary["prose"]["status"], "exploratory_post_hoc")
        self.assertIsNone(extract_strict_final_answer("Final answer: 7"))

    def test_duplicate_ids_and_changed_original_scores_are_rejected(self):
        row = {"example_id": "a", "response_text": "#### 7", "expected_answer": "7"}
        with self.assertRaises(ValueError):
            rescore_response_rows([row, row])
        with self.assertRaises(ValueError):
            rescore_response_rows([{**row, "is_correct": False}])
        with self.assertRaises(ValueError):
            rescore_response_rows([{**row, "predicted_answer": "9"}])


class RescoreReportTests(unittest.TestCase):
    def test_saved_run_and_new_report_round_trip(self):
        example = {
            "example_id": "fixture/0",
            "question": "A fixture?",
            "expected_answer": "7",
        }
        rows = [
            {
                **example,
                "response_text": "So, the answer is 7.",
                "predicted_answer": None,
                "is_correct": False,
            }
        ]
        protocol = {"number_of_examples": 1, "examples": [example]}
        manifest = {
            "run_id": "fixture",
            "protocol": protocol,
            "protocol_sha256": hashlib.sha256(
                json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            source_directory = Path(directory) / "original"
            source_directory.mkdir()
            (source_directory / "manifest.json").write_text(json.dumps(manifest))
            (source_directory / "responses.jsonl").write_text(
                json.dumps(rows[0]) + "\n"
            )
            before = {p.name: p.read_bytes() for p in source_directory.iterdir()}
            loaded, provenance = load_saved_baseline(source_directory)
            scored, summary = rescore_response_rows(loaded)
            archive = save_rescore_report(
                Path(directory) / "report", scored, summary, provenance
            )
            with zipfile.ZipFile(archive) as report:
                self.assertIsNone(report.testzip())
                self.assertEqual(
                    json.loads(report.read("rescore_summary.json")), summary
                )
                saved_manifest = json.loads(report.read("rescore_manifest.json"))
                self.assertEqual(saved_manifest["source"]["original_run_id"], "fixture")
                self.assertEqual(
                    hashlib.sha256(report.read("checker_source.py")).hexdigest(),
                    saved_manifest["checker_source_sha256"],
                )
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in source_directory.iterdir()}
            )
            with self.assertRaises(FileExistsError):
                save_rescore_report(source_directory, scored, summary, provenance)
            manifest["protocol"]["number_of_examples"] = 2
            (source_directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_saved_baseline(source_directory)


class KLDivergenceTests(unittest.TestCase):
    def test_known_value_and_self_divergence(self):
        self.assertAlmostEqual(
            categorical_kl([0.5, 0.5], [0.25, 0.75]), 0.5 * math.log(4 / 3)
        )
        self.assertEqual(categorical_kl([0.3, 0.7], [0.3, 0.7]), 0)

    def test_divergence_is_asymmetric(self):
        self.assertNotAlmostEqual(
            categorical_kl([0.5, 0.5], [0.1, 0.9]),
            categorical_kl([0.1, 0.9], [0.5, 0.5]),
        )

    def test_support_boundaries(self):
        self.assertEqual(categorical_kl([0, 1], [0, 1]), 0)
        self.assertEqual(categorical_kl([1, 0], [0, 1]), math.inf)

    def test_invalid_distributions(self):
        for p, q in [
            ([], []),
            ([1], [0.5, 0.5]),
            ([-0.1, 1.1], [0.5, 0.5]),
            ([math.nan, 0], [0.5, 0.5]),
            ([math.inf, 0], [0.5, 0.5]),
            ([0.4, 0.5], [0.5, 0.5]),
        ]:
            with self.subTest(p=p, q=q), self.assertRaises(ValueError):
                categorical_kl(p, q)


if __name__ == "__main__":
    unittest.main()
