"""Collect checksummed experiment evidence using CPU-only file inspection.

The two positional arguments remain compatible with the original collector.
Model and optimizer tensors are verified at the source and stay on that server.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.opd import read_resume  # noqa: E402
from src.utils import verified_zip, verify_inventory, write_inventory, write_json  # noqa: E402


def collect(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise FileExistsError("Choose a fresh evidence destination.")
    if destination.is_relative_to(directory):
        raise ValueError("Put the evidence archive outside the source run directory.")
    if not (directory / "experiment.json").is_file():
        raise FileNotFoundError("Missing run experiment.json.")
    comparison = directory / "eos_comparison"
    is_comparison = comparison.is_dir()
    arms = [p for p in sorted(comparison.iterdir()) if p.is_dir()] if is_comparison else [directory]
    destination.parent.mkdir(parents=True, exist_ok=True)
    verified = {}
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "report"
        staged.mkdir()

        def copy_file(source):
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Expected a regular evidence file: {source}")
            relative = source.relative_to(directory)
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        for folder in dict.fromkeys([directory, *arms]):
            for source in sorted(folder.iterdir()):
                if source.is_file() and source.suffix in {".json", ".jsonl", ".png", ".svg"}:
                    copy_file(source)
        for arm_directory in arms:
            specification_path = arm_directory / "experiment.json"
            if not specification_path.exists():
                continue
            specification = json.loads(specification_path.read_text())
            arm_key = arm_directory.relative_to(directory).as_posix()
            verified[arm_key] = {}
            for metadata_path in sorted(arm_directory.glob("resume/*/step_*/resume.json")):
                snapshot = metadata_path.parent
                arm_name = snapshot.parent.name
                metadata = read_resume(snapshot, arm=arm_name, signature=specification["signature"])
                verified[arm_key][snapshot.relative_to(arm_directory).as_posix()] = metadata[
                    "completed_steps"
                ]
                for name in (
                    "resume.json",
                    "SHA256SUMS.json",
                    "student/config.json",
                    "student/generation_config.json",
                    "student/tokenizer_config.json",
                ):
                    source = snapshot / name
                    if source.is_file():
                        copy_file(source)
        controls = {}
        for checkpoint in sorted(directory.glob("checkpoints/*")):
            if checkpoint.is_dir() and (checkpoint / "SHA256SUMS.json").is_file():
                verify_inventory(checkpoint)
                controls[checkpoint.name] = True
                for name in ("SHA256SUMS.json", "config.json", "generation_config.json"):
                    if (checkpoint / name).is_file():
                        copy_file(checkpoint / name)
        write_json(
            staged / "collection.json",
            {
                "source_directory": str(directory),
                "layout": "eos_comparison" if is_comparison else "historical_single_run",
                "verified_snapshot_steps": verified,
                "verified_final_checkpoints": controls,
                "weights_in_archive": False,
                "scope": "Retained snapshot inventories were checked against source tensor files.",
            },
        )
        write_inventory(staged)
        verify_inventory(staged)
        staged.rename(destination)
    return verified_zip(destination, destination.with_suffix(".zip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(collect(args.directory, args.destination))


if __name__ == "__main__":
    main()
