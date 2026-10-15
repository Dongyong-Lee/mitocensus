#!/usr/bin/env python3
"""Verify a built MitoCensus release directory against SHA256SUMS.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    manifest_path = package / "SHA256SUMS.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]
    actual = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and path != manifest_path
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        relative
        for relative in set(expected) & set(actual)
        if actual[relative].stat().st_size != expected[relative]["bytes"]
        or sha256(actual[relative]) != expected[relative]["sha256"]
    )
    if missing or extra or mismatched:
        raise SystemExit(
            json.dumps(
                {"status": "FAIL", "missing": missing, "extra": extra, "mismatched": mismatched},
                indent=2,
            )
        )
    if any(path.stat().st_size == 0 for path in actual.values()):
        raise SystemExit("release contains an empty governed file")
    print(
        json.dumps(
            {"status": "PASS", "version": manifest["version"], "files": len(actual)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

