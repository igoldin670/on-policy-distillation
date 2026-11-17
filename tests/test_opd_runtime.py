"""Exercise the notebook's actual cache-setup code with temporary directories."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NotebookCacheTests(unittest.TestCase):
    def setUp(self):
        notebook = json.loads((ROOT / "notebooks/04_on_policy_distillation.ipynb").read_text())
        runtime = next(cell for cell in notebook["cells"] if cell["id"] == "opd-setup-runtime")
        source = "".join(runtime["source"])
        self.source = source[
            source.index("scratch_directory =") : source.index("# Public pinned downloads")
        ]
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def execute(self, environment, *, scratch_exists=True):
        scratch = self.root / "scratch"
        if scratch_exists:
            (scratch / "qauser").mkdir(parents=True)
        namespace = {
            "os": os,
            "PROJECT_ROOT": self.root / "project",
            "Path": lambda value: scratch if str(value) == "/scratch" else Path(value),
        }
        with patch.dict(os.environ, environment, clear=True):
            exec(compile(self.source, "notebook_cache_setup", "exec"), namespace)
            return dict(os.environ), namespace

    def test_amarel_uses_scratch(self):
        environment, namespace = self.execute({"USER": "qauser", "SLURM_JOB_ID": "test-only"})
        expected = self.root / "scratch/qauser/huggingface"
        self.assertTrue(namespace["IN_AMAREL"])
        self.assertEqual(environment["HF_HOME"], str(expected))
        self.assertEqual(environment["HF_HUB_CACHE"], str(expected / "hub"))
        self.assertEqual(environment["HF_DATASETS_CACHE"], str(expected / "datasets"))
        self.assertTrue(expected.is_dir())

    def test_local_inspection_uses_project_cache(self):
        environment, namespace = self.execute({"USER": "qauser"})
        self.assertFalse(namespace["IN_AMAREL"])
        self.assertEqual(
            environment["HF_HOME"], str(self.root / "project/.cache/source-verification")
        )

    def test_missing_scratch_keeps_portable_project_default(self):
        environment, namespace = self.execute(
            {"USER": "qauser", "SLURM_JOB_ID": "test-only"}, scratch_exists=False
        )
        self.assertFalse(namespace["IN_AMAREL"])
        self.assertIn("project/.cache", environment["HF_HOME"])

    def test_explicit_cache_overrides_are_preserved(self):
        expected = {
            "HF_HOME": str(self.root / "custom"),
            "HF_HUB_CACHE": str(self.root / "custom_hub"),
            "HF_DATASETS_CACHE": str(self.root / "custom_datasets"),
        }
        environment, _ = self.execute(expected | {"USER": "qauser", "SLURM_JOB_ID": "test-only"})
        self.assertEqual({key: environment[key] for key in expected}, expected)

    def test_missing_username_does_not_select_shared_scratch_root(self):
        _, namespace = self.execute({"SLURM_JOB_ID": "test-only"})
        self.assertFalse(namespace["IN_AMAREL"])


if __name__ == "__main__":
    unittest.main()
