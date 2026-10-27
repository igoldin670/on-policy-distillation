"""Check Colab restart handling without installing packages or starting a GPU."""

import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class ColabSetupTests(unittest.TestCase):
    def run_setup(self, installed_version, loaded_modules, *, mode="replay"):
        notebook = json.loads(
            (ROOT / "notebooks/02_baseline_evaluation.ipynb").read_text()
        )
        source = "".join(notebook["cells"][2]["source"])
        # Exercise the actual Colab dependency block, without the clone/bootstrap.
        block = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "IN_COLAB"
        )
        installer = Mock()
        namespace = {
            "IN_COLAB": True,
            "MODE": mode,
            "PROJECT_ROOT": ROOT,
            "sys": SimpleNamespace(modules=loaded_modules),
            "importlib": SimpleNamespace(
                metadata=SimpleNamespace(version=installed_version)
            ),
            "install_with_uv": installer,
        }
        return (
            compile(ast.Module(body=[block], type_ignores=[]), "setup", "exec"),
            namespace,
            installer,
        )

    def test_loaded_old_version_requires_restart_before_install(self):
        for package in ("numpy", "pandas", "matplotlib", "bottleneck"):
            with self.subTest(package=package):
                code, namespace, installer = self.run_setup(
                    lambda name: "new", {package: SimpleNamespace(__version__="old")}
                )
                with self.assertRaisesRegex(RuntimeError, "Restart the Colab kernel"):
                    exec(code, namespace)
                installer.assert_not_called()

    def test_installation_stops_before_plotting(self):
        code, namespace, installer = self.run_setup(lambda name: "old", {})
        with self.assertRaisesRegex(RuntimeError, "installed successfully"):
            exec(code, namespace)
        installer.assert_called_once()

    def test_colab_old_bottleneck_is_upgraded(self):
        versions = dict(
            line.split("==")
            for line in (ROOT / "requirements-notebook.txt").read_text().splitlines()
            if "==" in line
        )
        versions["bottleneck"] = "1.4.2"
        code, namespace, installer = self.run_setup(versions.__getitem__, {})
        with self.assertRaisesRegex(RuntimeError, "installed successfully"):
            exec(code, namespace)
        self.assertIn("bottleneck==1.5.0", installer.call_args.args[0])

    def test_generation_dependencies_install_before_model_imports(self):
        code, namespace, installer = self.run_setup(
            lambda name: "old", {}, mode="generate"
        )
        with self.assertRaisesRegex(RuntimeError, "installed successfully"):
            exec(code, namespace)
        self.assertIn("transformers==4.57.6", installer.call_args.args[0])
        self.assertFalse(
            any(item.startswith("torch") for item in installer.call_args.args[0])
        )

    def test_fresh_matching_kernel_continues(self):
        versions = dict(
            line.split("==")
            for line in (ROOT / "requirements-notebook.txt").read_text().splitlines()
            if "==" in line
        )
        versions["bottleneck"] = "1.5.0"
        code, namespace, installer = self.run_setup(
            versions.__getitem__,
            {
                name: SimpleNamespace(__version__=versions[name])
                for name in ("numpy", "pandas", "matplotlib")
            },
        )
        exec(code, namespace)
        installer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
