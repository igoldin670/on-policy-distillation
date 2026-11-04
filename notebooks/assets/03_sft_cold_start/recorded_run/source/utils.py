"""Small, dependency-free artifact helpers shared by the experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    """Hash large checkpoints incrementally instead of reading gigabytes into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_member(name: str) -> Path:
    """Archive/inventory names must stay inside their own artifact directory."""
    member = PurePosixPath(name)
    if not name or "\\" in name or member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return Path(*member.parts)


def verify_inventory(directory: Path, expected: dict | None = None) -> dict:
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Artifact directory must not be a symbolic link.")
    inventory = json.loads((directory / "SHA256SUMS.json").read_text())
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Artifact inventory must be a nonempty mapping.")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Artifact inventory contains a symbolic link.")
        if path.is_file() and path != directory / "SHA256SUMS.json":
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(inventory):
        raise ValueError("Artifact files differ from the checksum inventory.")
    if expected is not None and inventory != expected:
        raise ValueError("Checkpoint fingerprints do not match the SFT report.")
    for name, digest in inventory.items():
        member = relative_member(name)
        if not isinstance(digest, str) or sha256_file(directory / member) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return inventory


def write_inventory(directory: Path) -> dict:
    directory = Path(directory)
    inventory = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "SHA256SUMS.json"
    }
    write_json(directory / "SHA256SUMS.json", inventory)
    return inventory


def artifact_directory(path: Path, cache: Path, *, max_bytes=8 * 2**30) -> Path:
    """Accept a folder or safely expand a ZIP once, streaming each file to disk.

    Archived Python files are evidence only. This function never imports them.
    A content-addressed cache avoids expanding the same multi-GB ZIP every run.
    """
    path, cache = Path(path), Path(cache)
    if path.is_dir():
        return path
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise FileNotFoundError(f"Expected an artifact folder or ZIP: {path}")
    destination = cache / sha256_file(path)
    if destination.is_dir():
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        if len(names) != len(set(names)):
            raise ValueError("Archive contains duplicate names.")
        if sum(entry.file_size for entry in members) > max_bytes:
            raise ValueError("Archive exceeds the uncompressed size limit.")
        for entry in members:
            relative_member(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Archive contains a symbolic link.")
        with tempfile.TemporaryDirectory(prefix="unpack-", dir=cache) as temporary:
            staging = Path(temporary)
            for entry in members:
                target = staging / relative_member(entry.filename)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest, length=1024 * 1024)
            staging.rename(destination)
    return destination
