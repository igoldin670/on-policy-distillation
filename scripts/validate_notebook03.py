"""Execute real-data CPU inspection, never the pretrained student's training path."""

from __future__ import annotations

import argparse
import copy
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


def validate(root: Path, *, save_inspection: bool = False):
    import ast

    path = root / "notebooks/03_sft_cold_start.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len({c.id for c in notebook.cells}) == len(notebook.cells)
    markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', markdown))
    for c in notebook.cells:
        if c.cell_type == "code":
            ast.parse(c.source)
            assert any(
                line.lstrip().startswith("#") for line in c.source.splitlines()
            ), c.id
            c.outputs, c.execution_count = [], None
        else:
            assert c.source.count("```") % 2 == 0
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if link.startswith("#"):
            assert link[1:] in anchors, link
        elif not link.startswith(("https:", "http:")):
            assert (path.parent / link).exists(), link
    # Set a private environment for the child kernel; do not change the VS Code kernel.
    os.environ["OPD_SFT_MODE"] = "inspect"
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str(root / ".cache/source-verification"))
    with tempfile.TemporaryDirectory(prefix="opd03-kernel-") as folder:
        spec = Path(folder) / "python3"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "SFT CPU QA",
                    "language": "python",
                }
            )
        )
        manager = KernelManager(
            kernel_name="python3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[folder]),
            **(
                {"transport": "ipc", "ip": str(Path(folder) / "kernel")}
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
            "Executing all cells in CPU INSPECT mode (real tokenizer/data, no student weights)...",
            flush=True,
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code_cells] == list(
        range(1, len(code_cells) + 1)
    )
    warnings = []
    for c in code_cells:
        for output in c.outputs:
            assert output.output_type != "error", c.id
            if (
                output.output_type == "stream"
                and output.name == "stderr"
                and output.text.strip()
            ):
                warnings.append({"cell": c.id, "text": output.text})
    # Keep logs explicit; unexpected warnings must be resolved, not suppressed here.
    if warnings:
        raise ValueError(f"Unexpected kernel stderr: {warnings}")
    if save_inspection:
        nbformat.write(notebook, path)
    destination = root / "results/notebook03_qa"
    destination.mkdir(parents=True, exist_ok=True)
    relocated = copy.deepcopy(notebook)
    for c in relocated.cells:
        if c.cell_type == "markdown":
            for link in re.findall(r"\]\(([^)]+)\)", c.source):
                if not link.startswith(("https:", "http:", "#")):
                    c.source = c.source.replace(
                        f"]({link})",
                        f"]({os.path.relpath(path.parent / link, destination)})",
                    )
    nbformat.write(relocated, destination / "notebook03_inspect.ipynb")
    html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(
        notebook, resources={"metadata": {"path": str(path.parent)}}
    )
    # Export is in results/, so repair ordinary relative links after image embedding.
    import html as html_tools

    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if not link.startswith(("https:", "http:", "#")):
            html = html.replace(
                f'href="{html_tools.escape(link, quote=True)}"',
                f'href="{html_tools.escape(os.path.relpath(path.parent / link, destination), quote=True)}"',
            )
    (destination / "notebook03.html").write_text(html)
    report = {
        "execution_mode": "inspect",
        "clean_kernel": True,
        "code_cells_executed": len(code_cells),
        "errors": 0,
        "stderr_outputs": 0,
        "student_weights_loaded": False,
        "real_student_training_performed": False,
        "source_notebook_updated": save_inspection,
        "html_export": str((destination / "notebook03.html").relative_to(root)),
    }
    (destination / "execution_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-inspection",
        action="store_true",
        help="Save genuinely executed inspection outputs to 03.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(
                Path(__file__).resolve().parents[1],
                save_inspection=args.save_inspection,
            ),
            indent=2,
        )
    )
