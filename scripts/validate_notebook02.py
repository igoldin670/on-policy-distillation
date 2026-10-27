"""Execute Notebook 02 sequentially in a clean CPU kernel and export HTML for QA.

Run from the project root after installing requirements-notebook.txt. This
validates replay only; it must never be reported as a new CUDA inference run.
"""

from __future__ import annotations

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


def validate_and_execute(root: Path, *, update_notebook: bool = False) -> dict:
    print("Checking notebook structure and relative assets...", flush=True)
    path = root / "notebooks/02_baseline_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ids = [cell.id for cell in notebook.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Notebook cell IDs must be unique.")
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', markdown))
    if not set(re.findall(r"\]\(#([^)]+)\)", markdown)) <= anchors:
        raise ValueError("A notebook navigation link has no target.")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=cell.id)
            cell.execution_count = None
            cell.outputs = []
        elif cell.source.count("```") % 2:
            raise ValueError(f"Unbalanced Markdown code fence: {cell.id}")
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if (
            not link.startswith(("http:", "https:", "#"))
            and not (path.parent / link).exists()
        ):
            raise ValueError(f"Missing relative asset or file: {link}")

    # A private temporary kernelspec points to this exact Python environment.
    # It neither changes the user's selected VS Code/Colab kernel nor installs globally.
    os.environ["OPD_NOTEBOOK_MODE"] = "replay"
    with tempfile.TemporaryDirectory(prefix="opd-kernels-") as folder:
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
                    "display_name": "Python 3",
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
            timeout=180,
            resources={"metadata": {"path": str(path.parent)}},
            allow_errors=False,
            startup_timeout=25,
        )
        try:
            print(
                "Starting an isolated kernel and executing replay cells...", flush=True
            )
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if [cell.execution_count for cell in code_cells] != list(
        range(1, len(code_cells) + 1)
    ):
        raise ValueError("Expected a sequential clean-kernel execution.")
    for cell in code_cells:
        for output in cell.outputs:
            if output.output_type == "error":
                raise ValueError(f"Execution error in {cell.id}")
            if (
                output.output_type == "stream"
                and output.name == "stderr"
                and output.text.strip()
            ):
                raise ValueError(f"Unexpected warning/log in {cell.id}: {output.text}")
    # Keep the author's saved CUDA outputs intact by default. CPU validation
    # produces a separate, genuinely executed replay notebook and HTML preview.
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    output_directory = root / "results/notebook02_qa"
    output_directory.mkdir(parents=True, exist_ok=True)
    replay_path = output_directory / "notebook02_replay.ipynb"
    replay_notebook = copy.deepcopy(notebook)
    for cell in replay_notebook.cells:
        if cell.cell_type == "markdown":
            for link in re.findall(r"\]\(([^)]+)\)", cell.source):
                if not link.startswith(("http:", "https:", "#")):
                    relocated = os.path.relpath(path.parent / link, output_directory)
                    cell.source = cell.source.replace(f"]({link})", f"]({relocated})")
    nbformat.write(replay_notebook, replay_path)
    if update_notebook:
        nbformat.write(notebook, path)
    print("Execution complete; exporting HTML...", flush=True)
    exporter = HTMLExporter(
        template_name="lab",
        embed_images=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    html, _ = exporter.from_notebook_node(
        notebook, resources={"metadata": {"path": str(path.parent)}}
    )
    # The preview lives two folders away from the notebook. Keep its asset/module
    # links working too, without changing the GitHub-relative links in the .ipynb.
    for link in re.findall(r"\]\(([^)]+)\)", markdown):
        if not link.startswith(("http:", "https:", "#")):
            preview_link = os.path.relpath(path.parent / link, output_directory)
            html = html.replace(
                f'href="{html_tools.escape(link, quote=True)}"',
                f'href="{html_tools.escape(preview_link, quote=True)}"',
            )
    html_path = output_directory / "notebook02.html"
    html_path.write_text(html, encoding="utf-8")
    png_outputs = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.outputs
    )
    if png_outputs != 4:
        raise ValueError("Expected four rendered figures.")
    report = {
        "execution_mode": "replay",
        "fresh_model_inference": False,
        "clean_kernel": True,
        "code_cells_executed": len(code_cells),
        "errors": 0,
        "stderr_outputs": 0,
        "rendered_figures": png_outputs,
        "html_export": str(html_path.relative_to(root)),
        "replay_notebook": str(replay_path.relative_to(root)),
        "source_notebook_updated": update_notebook,
    }
    (output_directory / "execution_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-notebook",
        action="store_true",
        help="Replace the source notebook outputs with CPU replay (default: preserve).",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_and_execute(
                Path(__file__).resolve().parents[1],
                update_notebook=args.update_notebook,
            ),
            indent=2,
        )
    )
