# Reproducing MitoCensus 0.3.0

## Fast path: verify the released executable examples

```bash
conda env create -f environment.yml
conda activate mitocensus
python -m pip install .
python -m pytest -q
python scripts/verify_minimal_example.py
python scripts/verify_biological_analysis.py
```

Expected outcomes are 32 passing tests with 134 passing parameterized subtests (16 core and 16 object-analysis tests),
`MINIMAL_EXAMPLE_PASS` and `BIOLOGICAL_ANALYSIS_PASS`. All tests use synthetic data and run without non-distributed
records.

The minimal-example verifier checks the frozen files against SHA-256, reruns CPU inference, compares probabilities
within absolute tolerance 1e-6, requires exact semantic and instance arrays, and compares stable provenance fields.

## Stored analysis records of the manuscript

The analysis code of this release, re-run on the original inputs, reproduces every numeric value, class, flag and
identifier of each stored analysis record of the manuscript, apart from the superseded first split-screen outputs
(kept to document the corrected screen). The records comprise four demonstration volumes (default checks; QC v2 with
edge weights), the minimal example, 54 development audit records and 242 held-out and replication crop records; 10
development volume–source pairs, whose label maps are empty, are declined. The demonstration volumes, the minimal
example and 188 of the 296 audit records are reproduced with the library versions pinned in `environment.yml` (NumPy
1.26.4, SciPy 1.11.4, scikit-image 0.22.0); the other 108, all of them default-check development or crop records, are
reproduced with NumPy 2.3.5, SciPy 1.18.0 and scikit-image 0.25.2 (an environment that also differs in Python and
tifffile versions). Between the two library sets, classes, flags, counts and identifiers of these 108 records are
identical, principal-axis vectors are equal up to sign, and every other value differs by less than 0.08% relative
(largest 0.076%, line-filter coherence) or, for values near zero, by less than 1e-7.

## Library versions

Intensity-derived measurements (line-filter, matrix and boundary features) depend slightly on the numerical library
environment: between the two sets above, the records of the manuscript differ by less than 0.08%, or by less than 1e-7
for values near zero, with classes, flags and counts unchanged and principal-axis directions equal up to sign. The two
sets also differ in Python and tifffile versions; which library causes the differences was not isolated.
`environment.yml` pins the tested versions, while `pyproject.toml` sets only lower bounds, so `python -m pip install
.` can install newer versions. To reproduce a stored record bit for bit, use the library versions that reproduce it;
the development regression report records which set reproduces each record.

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
docker build -t mitocensus:0.3.0 .
docker run --rm mitocensus:0.3.0 --help
```

## Interpretation boundary

Do not infer biological replication from technical crops, sections, voxels or objects. Validate the entire pipeline
on representative labelled volumes from the intended acquisition before morphometry or biological use.
