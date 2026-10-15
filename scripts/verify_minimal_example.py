#!/usr/bin/env python3
"""Re-run the minimal example and compare it with the frozen CPU reference."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    project = project_root()
    example = project / "examples/minimal"
    frozen = json.loads((example / "expected_manifest.json").read_text())
    file_errors: list[str] = []
    for relative, expected in frozen["files"].items():
        path = example / relative
        if not path.is_file():
            file_errors.append(f"missing frozen file: {relative}")
        elif sha256(path) != expected["sha256"]:
            file_errors.append(f"frozen hash mismatch: {relative}")
    if file_errors:
        raise SystemExit("\n".join(file_errors))

    with tempfile.TemporaryDirectory(prefix="mitocensus_verify_") as temporary:
        output = Path(temporary) / "output"
        command = [
            sys.executable,
            "-m",
            "mitocensus.cli",
            "infer-from-priors",
            "--raw",
            str(example / "input/raw_em.tif"),
            "--mitonet-probability",
            str(example / "input/mitonet_probability.tif"),
            "--omniem-probability",
            str(example / "input/omniem_probability.tif"),
            "--mitonet-instances",
            str(example / "input/mitonet_instances.tif"),
            "--policy-json",
            str(example / "g011_policy.json"),
            "--checkpoint",
            str(example / "refinement_C04.pt"),
            "--spacing",
            "30,16,16",
            "--output",
            str(output),
            "--cpu",
        ]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(project / "src")
        subprocess.run(command, cwd=project, env=environment, check=True)

        reference = example / "expected_output"
        probability = tifffile.imread(output / "refined_probability.tif")
        expected_probability = tifffile.imread(reference / "refined_probability.tif")
        if not np.allclose(probability, expected_probability, rtol=0.0, atol=1e-6):
            difference = float(np.max(np.abs(probability - expected_probability)))
            raise SystemExit(f"probability mismatch: maximum absolute difference {difference}")
        for name in ("refined_semantic.tif", "refined_instances.tif"):
            actual = tifffile.imread(output / name)
            expected = tifffile.imread(reference / name)
            if not np.array_equal(actual, expected):
                raise SystemExit(f"array mismatch: {name}")
        generated_manifest = json.loads((output / "manifest.json").read_text())
        expected_manifest = json.loads((reference / "manifest.json").read_text())
        stable_fields = (
            "schema_version",
            "checkpoint_sha256",
            "config",
            "shape_zyx",
            "spacing_nm_zyx",
            "normalization",
            "threshold",
            "reconstruction_policy",
        )
        for field in stable_fields:
            if generated_manifest[field] != expected_manifest[field]:
                raise SystemExit(f"manifest mismatch: {field}")

    print("MINIMAL_EXAMPLE_PASS")


if __name__ == "__main__":
    main()

