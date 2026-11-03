"""Clean-kernel REAL-DATA inspection of 04. Never run Qwen training on the CPU."""

from __future__ import annotations

import argparse
import ast
import copy
import html as html_tools
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def validate(root: Path, *, save_inspection=False):
    path = root / "notebooks/04_on_policy_distillation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if len({c.id for c in notebook.cells}) != len(notebook.cells):
        raise ValueError("Duplicate cell IDs.")
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', markdown))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=cell.id)
            if not any(line.lstrip().startswith("#") for line in cell.source.splitlines()):
                raise ValueError(f"Missing guidance comments: {cell.id}")
            cell.outputs, cell.execution_count = [], None
        else:
            if cell.source.count("```") % 2:
                raise ValueError(f"Unbalanced code fence: {cell.id}")
            if "\u2014" in cell.source:
                raise ValueError(f"Em dash in new prose: {cell.id}")
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if link.startswith("#"):
            if link[1:] not in anchors:
                raise ValueError(f"Missing navigation anchor: {link}")
        elif not link.startswith(("https:", "http:")) and not (path.parent / link).exists():
            raise ValueError(f"Missing relative image/link: {link}")

    # The source notebook's mode/output state and the VS Code Colab kernel stay
    # untouched. Only this private child kernel receives the inspection setting.
    environment = dict(os.environ)
    environment.update(OPD_TRAIN_MODE="inspect", MPLCONFIGDIR=str(root / ".cache/matplotlib"))
    environment.setdefault("HF_HOME", str(root / ".cache/source-verification"))
    with tempfile.TemporaryDirectory(prefix="opd04-kernel-") as temporary:
        spec = Path(temporary) / "python3"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "OPD CPU inspection",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
            **(
                {"transport": "ipc", "ip": str(Path(temporary) / "kernel")}
                if os.name != "nt"
                else {}
            ),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=600,
            startup_timeout=30,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=False,
        )
        print(
            "Executing all 04 cells in a clean CPU inspection kernel; no Qwen weights...",
            flush=True,
        )
        try:
            client.execute(env=environment)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if [c.execution_count for c in code_cells] != list(range(1, len(code_cells) + 1)):
        raise ValueError("Cells did not execute sequentially from a clean kernel.")
    stderr = []
    for cell in code_cells:
        for output in cell.outputs:
            if output.output_type == "error":
                raise ValueError(f"Error output in {cell.id}.")
            if output.output_type == "stream" and output.name == "stderr" and output.text.strip():
                stderr.append({"cell": cell.id, "text": output.text})
    if stderr:
        raise ValueError(f"Review unexpected stderr: {stderr}")
    destination = root / "results/notebook04_qa"
    destination.mkdir(parents=True, exist_ok=True)
    relocated = copy.deepcopy(notebook)
    for cell in relocated.cells:
        if cell.cell_type == "markdown":
            for link in re.findall(r"\]\(([^)]+)\)", cell.source):
                if not link.startswith(("http:", "https:", "#")):
                    cell.source = cell.source.replace(
                        f"]({link})", f"]({os.path.relpath(path.parent / link, destination)})"
                    )
    nbformat.write(relocated, destination / "notebook04_inspect.ipynb")
    html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(
        notebook, resources={"metadata": {"path": str(path.parent)}}
    )
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if not link.startswith(("http:", "https:", "#")):
            html = html.replace(
                f'href="{html_tools.escape(link, quote=True)}"',
                f'href="{html_tools.escape(os.path.relpath(path.parent / link, destination), quote=True)}"',
            )
    (destination / "notebook04.html").write_text(html)
    if save_inspection:
        notebook.metadata["validation"] = {
            "execution_mode": "local_cpu_inspection",
            "real_qwen_training_performed": False,
            "teacher_weights_loaded": False,
        }
        nbformat.write(notebook, path)
    report = {
        "execution_mode": "inspect",
        "clean_kernel": True,
        "code_cells_executed": len(code_cells),
        "errors": 0,
        "stderr_outputs": 0,
        "real_student_weights_loaded": False,
        "teacher_weights_loaded": False,
        "real_qwen_training_performed": False,
        "source_notebook_updated": save_inspection,
        "html_export": str((destination / "notebook04.html").relative_to(root)),
    }
    (destination / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-inspection",
        action="store_true",
        help="Save actual inspection outputs to 04; never pretend these are GPU results.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(Path(__file__).resolve().parents[1], save_inspection=args.save_inspection),
            indent=2,
        )
    )
