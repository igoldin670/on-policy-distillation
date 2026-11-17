"""Execute profile-aware notebooks in isolated CPU kernels and preserve source outputs."""

from __future__ import annotations

import argparse
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


def inspect(number, profile):
    root = Path(__file__).resolve().parents[1]
    path = next((root / "notebooks").glob(f"{number}_*.ipynb"))
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    environment = dict(
        os.environ,
        OPD_MODEL_PROFILE=profile,
        OPD_SFT_MODE="inspect",
        OPD_TRAIN_MODE="inspect",
        MPLCONFIGDIR=str(root / ".cache/matplotlib"),
    )
    environment.setdefault("HF_HOME", str(root / ".cache/source-verification"))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    with tempfile.TemporaryDirectory(prefix="gkd-inspection-") as temporary:
        spec = Path(temporary) / "python3"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "GKD CPU inspection",
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
                timeout=600,
                allow_errors=False,
                resources={"metadata": {"path": str(path.parent)}},
            ).execute(env=environment)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
    warnings = []
    for cell in cells:
        for result in cell.outputs:
            assert result.output_type != "error"
            if result.output_type == "stream" and result.name == "stderr" and result.text.strip():
                lines = result.text.strip().splitlines()
                if not all(
                    line.startswith(
                        (
                            "Using the latest cached version of the dataset since ",
                            "Found the latest cached dataset configuration ",
                            "Warning: You are sending unauthenticated requests",
                        )
                    )
                    for line in lines
                ):
                    raise ValueError(f"Unexpected stderr in {cell.id}: {result.text}")
                warnings.append({"cell": cell.id, "text": result.text})
    destination = root / "results/cpu_release_qa" / f"{number}_{profile}"
    destination.mkdir(parents=True, exist_ok=True)
    relocated = copy.deepcopy(notebook)
    for cell in relocated.cells:
        if cell.cell_type == "markdown":
            for link in re.findall(r"\]\(([^)]+)\)", cell.source):
                if not link.startswith(("https:", "http:", "#")):
                    cell.source = cell.source.replace(
                        f"]({link})", f"]({os.path.relpath(path.parent / link, destination)})"
                    )
    nbformat.write(relocated, destination / path.name)
    html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(
        notebook, resources={"metadata": {"path": str(path.parent)}}
    )
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if not link.startswith(("https:", "http:", "#")):
            html = html.replace(
                f'href="{html_tools.escape(link, quote=True)}"',
                f'href="{html_tools.escape(os.path.relpath(path.parent / link, destination), quote=True)}"',
            )
    (destination / "notebook.html").write_text(html)
    report = {
        "profile": profile,
        "notebook": path.name,
        "clean_kernel": True,
        "code_cells_executed": len(cells),
        "errors": 0,
        "reviewed_notices": warnings,
        "pretrained_weights_loaded": False,
        "source_outputs_changed": False,
    }
    (destination / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", choices=["03", "04"])
    parser.add_argument("--profile", choices=["legacy", "qwen38"], default="qwen38")
    args = parser.parse_args()
    print(json.dumps(inspect(args.number, args.profile), indent=2))
