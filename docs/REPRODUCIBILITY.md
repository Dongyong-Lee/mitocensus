# Reproducing MitoCensus 0.4.0

## Fast path: verify the released executable examples

```bash
conda env create -f environment.yml
conda activate mitocensus
python -m pip install .
python -m pytest -q
python scripts/verify_minimal_example.py
python scripts/verify_biological_analysis.py
```

Expected outcomes are 44 passing tests with 134 passing parameterized subtests (16 core tests and 28 object-analysis
and visual-record tests),
`MINIMAL_EXAMPLE_PASS` and `BIOLOGICAL_ANALYSIS_PASS`. All tests use synthetic data and run without non-distributed
records.

The minimal-example verifier checks the frozen files against SHA-256, reruns CPU inference, compares probabilities
within absolute tolerance 1e-6, requires exact semantic and instance arrays, and compares stable provenance fields.

## Stored analysis records of the manuscript

The stored analysis records of the manuscript were rewritten with the measurement rules of version 0.3.1. The analysis
code of this release, re-run on the original inputs with the library versions pinned in `environment.yml` (NumPy
1.26.4, SciPy 1.11.4, scikit-image 0.22.0), reproduces every value, class, flag and identifier of each record: the
demonstration volumes (default checks; QC v2 with edge weights), the minimal example, 54 development audit records and
242 held-out and replication crop records; 10 development volume–source pairs, whose label maps are empty, are
declined. The outputs of the first version of the split-label screen, kept to document the corrected screen, come from
a development variant of the screen and are not part of this comparison.

Version 0.4.0 changes the defaults (section-extent check on, split screen as a review flag) but not the measurements.
With the settings of version 0.3.1 (`--min-axial-extent-nm none --split-screen-min-iou none`, or `--min-axial-extent-nm
none --min-axial-sections 3 --split-screen-action exclude` for the 0.3.1 "all checks"), it writes the same object tables
as version 0.3.1 for every held-out and development record of the manuscript.

## Library versions

Intensity-derived measurements (line-filter, matrix and boundary features) depend slightly on the numerical library
environment. Re-run with NumPy 2.3.5, SciPy 1.18.0 and scikit-image 0.25.2 (and newer Python and tifffile versions),
the development and held-out records of the manuscript keep every class, flag, count and identifier, principal-axis
directions are equal up to sign, and every other value differs by less than 0.02% relative or, for values near zero,
by less than 1e-7; which library causes the differences was not isolated. `environment.yml` pins the tested versions,
while `pyproject.toml` sets only lower bounds, so `python -m pip install .` can install newer versions. To reproduce a
stored record bit for bit, use the pinned versions.

## Verify an evidence bundle

```bash
python scripts/verify_release_bundle.py <release directory>
```

The verifier rejects missing, extra, empty or hash-mismatched governed files. It does not imply that third-party
datasets or weights are redistributed.

## Reproduce manuscript-level analyses

Full reruns require the canonical MitoEM 2.0, CellMap/OpenOrganelle, MitoNet and OmniEM/EM-DINO resources. Obtain
these from their official sources and follow their licenses. Large source data and third-party model files are
intentionally excluded.

The frozen one-way external procedure cannot be recreated as outcome-unseen after its references have been opened. A
rerun can reproduce the reported metrics, but it is a reproduction of a completed locked evaluation, not a new
independent validation.

## Container boundary

`Dockerfile` provides a Linux Python 3.11 package installation route. The local publication workstation did not have
a Docker engine, so the release does not claim an executed clean container build. Run the build in CI or another host
and archive the log before journal submission:

```bash
docker build -t mitocensus:0.4.0 .
docker run --rm mitocensus:0.4.0 --help
```

## Interpretation boundary

Do not infer biological replication from technical crops, sections, voxels or objects. Validate the entire pipeline
on representative labelled volumes from the intended acquisition before morphometry or biological use.
