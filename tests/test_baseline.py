"""Portable tests use the committed recorded run, not personal Downloads paths."""

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.baseline import (
    export_analysis,
    generate_fresh_baseline,
    metric_rows,
    validate_generation_ids,
    verify_recorded_run,
)
from src.evaluation import extract_prose_final_answer

ROOT = Path(__file__).resolve().parents[1]
RECORDED = ROOT / "notebooks/assets/02_baseline_evaluation/recorded_run"


class RecordedRunTests(unittest.TestCase):
    def test_original_and_exploratory_results_recompute(self):
        run = verify_recorded_run(RECORDED)
        self.assertEqual(len(run["rows"]), 32)
        self.assertEqual(run["scores"]["strict"]["number_correct"], 0)
        self.assertEqual(run["scores"]["prose"]["number_correct"], 17)
        self.assertEqual(run["scores"]["prose"]["number_parsed"], 28)
        self.assertEqual(run["original_summary"]["number_at_token_limit"], 2)

    def test_partition_retains_all_questions(self):
        run = verify_recorded_run(RECORDED)
        metrics = metric_rows(run["annotations"], run["scores"])
        self.assertEqual(
            [(r["correct"], r["parsed_wrong"], r["unparsed"]) for r in metrics],
            [(0, 0, 32), (17, 11, 4)],
        )

    def test_file_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            shutil.copytree(RECORDED, target)
            with (target / "responses.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_recorded_run(target)

    def test_invalid_token_metadata_is_detected_even_if_file_hash_is_updated(self):
        for mutation in [
            lambda row: row.update(response_length=999),
            lambda row: row["response_token_ids"].__setitem__(0, -1),
            lambda row: row.update(ended_with_eos=not row["ended_with_eos"]),
            lambda row: row.update(answer_parsed="False"),
        ]:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as folder,
            ):
                target = Path(folder) / "run"
                shutil.copytree(RECORDED, target)
                path = target / "responses.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                mutation(rows[0])
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                )
                checksums = json.loads((target / "SHA256SUMS.json").read_text())
                checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (target / "SHA256SUMS.json").write_text(json.dumps(checksums))
                with self.assertRaises(ValueError):
                    verify_recorded_run(target)

    def test_changing_reference_labels_does_not_change_extraction(self):
        run = verify_recorded_run(RECORDED)
        for row in run["rows"]:
            # The extractor accepts no gold-answer argument in the first place.
            extraction = extract_prose_final_answer(
                row["response_text"], hit_token_limit=row["hit_token_limit"]
            )
            altered = {**row, "expected_answer": "999999"}
            self.assertEqual(
                extraction,
                extract_prose_final_answer(
                    altered["response_text"], hit_token_limit=altered["hit_token_limit"]
                ),
            )

    def test_export_contains_provenance_and_exact_checker(self):
        run = verify_recorded_run(RECORDED)
        hashes_before = copy.deepcopy(run["artifact_sha256"])
        # Export under an isolated root with the same source tree and notebook.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(
                ROOT / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
            )
            (root / "notebooks").mkdir()
            shutil.copyfile(
                ROOT / "notebooks/02_baseline_evaluation.ipynb",
                root / "notebooks/02_baseline_evaluation.ipynb",
            )
            archive = export_analysis(root, run, execution_mode="replay")
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertEqual(
                    len([name for name in bundle.namelist() if not name.endswith("/")]),
                    11,
                )
                metadata = json.loads(bundle.read("analysis_metadata.json"))
                self.assertFalse(metadata["inference_performed_this_execution"])
                self.assertEqual(metadata["execution_mode"], "replay")
                self.assertEqual(
                    json.loads(bundle.read("rescore_summary.json")), run["scores"]
                )
                for name in (*run["artifact_sha256"], "SHA256SUMS.json"):
                    self.assertEqual(
                        bundle.read(f"raw_run/{name}"), (RECORDED / name).read_bytes()
                    )
                restored_directory = root / "restored"
                bundle.extractall(restored_directory)
                restored = verify_recorded_run(restored_directory / "raw_run")
                self.assertEqual(restored["rows"], run["rows"])
                self.assertEqual(restored["scores"], run["scores"])
        self.assertEqual(
            verify_recorded_run(RECORDED)["artifact_sha256"], hashes_before
        )


class GenerationBoundaryTests(unittest.TestCase):
    def test_response_slice_excludes_exact_prompt(self):
        self.assertEqual(validate_generation_ids([1, 2, 3], [1, 2, 3, 4, 5], 2), [4, 5])

    def test_empty_misaligned_and_overlong_generation_fail(self):
        for prompt, sequence, budget in [
            ([], [1], 2),
            ([1, 2], [1, 9, 3], 2),
            ([1], [1], 2),
            ([1], [1, 2, 3], 1),
        ]:
            with (
                self.subTest(prompt=prompt, sequence=sequence),
                self.assertRaises(ValueError),
            ):
                validate_generation_ids(prompt, sequence, budget)

    def test_cpu_guard_precedes_model_loading(self):
        # A unit test of fail-fast behavior, not a substitute for CUDA inference.
        model_loader = MagicMock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "datasets": SimpleNamespace(load_dataset=MagicMock()),
            "transformers": SimpleNamespace(
                AutoModelForCausalLM=model_loader,
                AutoTokenizer=MagicMock(),
                GenerationConfig=MagicMock(),
            ),
        }
        with (
            patch.dict("sys.modules", modules),
            self.assertRaisesRegex(RuntimeError, "working CUDA"),
        ):
            generate_fresh_baseline({}, Path("unused-output"))
        model_loader.from_pretrained.assert_not_called()


if __name__ == "__main__":
    unittest.main()
