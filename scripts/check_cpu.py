"""Portable fresh-environment CPU checks. No pretrained model weights are loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_sources():
    import shutil

    from huggingface_hub import hf_hub_download

    from src.profiles import load_profile
    from src.utils import sha256_file, write_json

    profile = load_profile(ROOT)
    sources = ROOT / ".cache/qwen38-verification"
    for role in ("student", "teacher"):
        target = sources / role
        target.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        ):
            source = hf_hub_download(
                profile[f"{role}_model_id"], name, revision=profile[f"{role}_revision"], token=False
            )
            if Path(source).resolve() != (target / name).resolve():
                shutil.copyfile(source, target / name)
            hashes[name] = sha256_file(target / name)
        write_json(
            target / "download_manifest.json",
            {
                "model_id": profile[f"{role}_model_id"],
                "revision": profile[f"{role}_revision"],
                "sha256": hashes,
            },
        )
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Use already cached metadata, data and KaTeX."
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run static and unit checks without notebook kernels.",
    )
    parser.add_argument("--profile", choices=["legacy", "qwen38"], default="qwen38")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/source-verification"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.offline:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    sources = ROOT / ".cache/qwen38-verification"
    if args.profile == "qwen38":
        sources = sources if args.offline else prepare_sources()
        os.environ["OPD_QWEN38_LOCAL_SOURCES"] = str(sources)
    katex = ROOT / ".cache/katex.min.js"
    expected = "e8d885505949f3a5f4abdd5dd0d53696bd1371ad26ffbf4f310dcd77c8cdae89"
    if not katex.exists() and not args.offline:
        with urllib.request.urlopen(
            "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js", timeout=60
        ) as response:
            katex.write_bytes(response.read())
    if not katex.exists() or hashlib.sha256(katex.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the pinned official KaTeX bundle.")
    commands = [
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests", "notebooks"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "scripts", "tests", "notebooks"],
        [sys.executable, "scripts/validate_notebook_series.py", "--katex-js", str(katex)],
    ]
    if not args.unit_only:
        commands += [[sys.executable, "scripts/validate_notebook02.py"]]
        for profile in ("legacy", "qwen38") if args.profile == "qwen38" else ("legacy",):
            for number in ("03", "04"):
                commands.append(
                    [sys.executable, "scripts/inspect_notebook.py", number, "--profile", profile]
                )
        if args.profile == "qwen38":
            commands.append(
                [sys.executable, "scripts/validate_model_upgrade.py", "--sources", str(sources)]
            )
    records = []
    for command in commands:
        print("Running", " ".join(command), flush=True)
        result = subprocess.run(command, check=False)
        records.append({"command": command, "exit_code": result.returncode})
        if result.returncode:
            break
    destination = ROOT / "results/cpu_release_qa"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{args.profile}_checks.json").write_text(
        json.dumps(
            {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
                "working_tree_dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"]).strip()
                ),
                "checks": records,
                "pretrained_model_weights_loaded": False,
            },
            indent=2,
        )
        + "\n"
    )
    if records[-1]["exit_code"]:
        raise SystemExit(records[-1]["exit_code"])


if __name__ == "__main__":
    main()
