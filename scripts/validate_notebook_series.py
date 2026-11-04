"""Static and HTML QA for all four notebooks; this never executes model cells.

For equation rendering, install quickjs==1.19.4 in your isolated QA environment
and supply the official KaTeX bundle with --katex-js. GPU checks are separate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mistune
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
from nbconvert import HTMLExporter
from PIL import Image


def validate(root: Path, katex_js: Path | None = None):
    context = None
    if katex_js is not None:
        import quickjs

        context = quickjs.Context()
        context.eval(katex_js.read_text())
    markdown_parser = mistune.create_markdown(renderer="ast", plugins=["math"])
    transformer = TransformerManager()
    destination = root / "results/notebook_series_qa"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted((root / "notebooks").glob("0[1-4]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        anchors = set(re.findall(r'<a id="([^"]+)"></a>', markdown))
        math_count = 0
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(transformer.transform_cell(cell.source), filename=cell.id)
                assert any(line.lstrip().startswith("#") for line in cell.source.splitlines()), (
                    cell.id
                )
            else:
                assert "\u2014" not in cell.source, cell.id
                assert cell.source.count("```") % 2 == 0, cell.id
                tokens = list(markdown_parser(cell.source))
                while tokens:
                    token = tokens.pop()
                    tokens.extend(token.get("children", []))
                    if token["type"] in {"block_math", "inline_math"}:
                        math_count += 1
                        if context is not None:
                            expression = json.dumps(token["raw"])
                            display = str(token["type"] == "block_math").lower()
                            context.eval(
                                f"katex.renderToString({expression}, {{displayMode:{display}, throwOnError:true, strict:'error'}})"
                            )
        for link in re.findall(r"\]\(([^)]+)\)", markdown):
            if link.startswith("#"):
                assert link[1:] in anchors, (path.name, link)
            elif not link.startswith(("https:", "http:")):
                assert (path.parent / link).exists(), (path.name, link)
        html, _ = HTMLExporter(template_name="lab", embed_images=True).from_notebook_node(
            notebook, resources={"metadata": {"path": str(path.parent)}}
        )
        assert "attachment:" not in html, path.name
        (destination / f"{path.stem}.html").write_text(html)
        records.append(
            {
                "notebook": path.name,
                "schema_and_syntax": "passed",
                "guidance_comments": "passed",
                "relative_links": "passed",
                "html_export": "passed",
                "equations": math_count,
                "equation_rendering": "passed" if context else "not_run",
            }
        )
    image_count = 0
    for folder in (root / "assets", root / "notebooks/images", root / "notebooks/assets"):
        for path in folder.rglob("*"):
            if path.suffix == ".png":
                with Image.open(path) as image:
                    image.verify()
                image_count += 1
            elif path.suffix == ".svg":
                ET.parse(path)
                image_count += 1
    result = {
        "date": "2026-11-04",
        "scope": "Offline validation. No notebook model execution or live browser layout check.",
        "notebooks": records,
        "image_files_checked": image_count,
        "katex_version": context.eval("katex.version") if context else None,
    }
    (destination / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--katex-js", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(Path(__file__).resolve().parents[1], args.katex_js), indent=2))
