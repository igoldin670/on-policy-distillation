"""Execute the new-model notebook in a separate CPU inspection kernel."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def validate(root, sources):
    path = root / "notebooks/05_qwen38_restart.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    environment = dict(os.environ)
    environment.update(
        OPD_QWEN38_MODE="inspect",
        OPD_QWEN38_LOCAL_SOURCES=str(sources.resolve()),
        HF_HUB_OFFLINE="1",
        MPLCONFIGDIR=str(root / ".cache/matplotlib"),
    )
    with tempfile.TemporaryDirectory(prefix="qwen38-kernel-") as temporary:
        spec = Path(temporary) / "python3"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Qwen3.8 CPU inspection",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]),
            transport="ipc",
            ip=str(Path(temporary) / "kernel"),
        )
        try:
            NotebookClient(
                notebook,
                km=manager,
                timeout=180,
                allow_errors=False,
                resources={"metadata": {"path": str(path.parent)}},
            ).execute(env=environment)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    for cell in code:
        for output in cell.outputs:
            assert output.output_type != "error"
            if output.output_type == "stream" and output.name == "stderr":
                assert not output.text.strip(), output.text
    destination = root / "results/qwen38_upgrade_qa"
    destination.mkdir(parents=True, exist_ok=True)
    relocated = copy.deepcopy(notebook)
    for cell in relocated.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "assets/05_qwen38_restart/", "../../notebooks/assets/05_qwen38_restart/"
            )
    nbformat.write(relocated, destination / "notebook05_inspect.ipynb")
    html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(
        notebook, resources={"metadata": {"path": str(path.parent)}}
    )
    html = html.replace(
        'href="assets/05_qwen38_restart/', 'href="../../notebooks/assets/05_qwen38_restart/'
    )
    (destination / "notebook05.html").write_text(html)
    report = {
        "clean_kernel": True,
        "execution_mode": "inspect",
        "code_cells_executed": len(code),
        "errors": 0,
        "real_qwen_weights_loaded": False,
        "training_performed": False,
        "source_notebook_outputs_changed": False,
    }
    (destination / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(Path(__file__).resolve().parents[1], args.sources), indent=2))
