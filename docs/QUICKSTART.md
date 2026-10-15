# MitoCensus quick start

MitoCensus measures and quality-controls every object of an aligned mitochondrial instance mask
(`analyze-biological`), compares groups by biological unit (`compare-biological`) and includes an optional refinement
module (`infer-from-priors`).

## 1. Create the tested environment

```bash
conda env create -f environment.yml
conda activate mitocensus
python -m pip install .
```

The exact frozen environment uses Python 3.11 and is recorded in `environment.yml`. The package also supplies a
Linux-compatible Dockerfile; the local publication audit could not execute a container build because no Docker engine
was installed on the test Mac.

## 2. Measure the objects of an instance mask

```bash
mitocensus analyze-biological \
  --raw sample_raw.tif \
  --instances sample_instances.tif \
  --spacing 30,16,16 \
  --sample-id specimen-01 \
  --group control \
  --output specimen-01_census \
  --min-axial-sections 3 --split-screen-min-iou 0.4 --edge-correction-weights
```

The instance mask can come from any segmenter. The last line switches on the section-support rule, the split-label
screen and edge weights; without it only the border and connectivity checks decide which labels enter whole-object
morphometry, and four object columns and five summary entries still record the options. This writes per-object
measurements and checks, sample-level summaries, close mitochondrion pairs, a manual-review label map and a visual
report for the 13 functions: amount; size and swelling; shape classes; line-filter score; matrix appearance; boundary
features; population heterogeneity; spatial organization; rare morphologies; mitochondria-on-a-string and
nanotunnel-like geometry; mitochondrion–mitochondrion pairs; cell-state-associated patterns; and, with
`compare-biological`, disease or treatment response by biological unit. The complete input, output and claim contract
is in [BIOLOGICAL_ANALYSIS.md](BIOLOGICAL_ANALYSIS.md).

## 3. Run the self-contained examples

```bash
python scripts/verify_minimal_example.py
python scripts/verify_biological_analysis.py
```

Expected terminal results: `MINIMAL_EXAMPLE_PASS` and `BIOLOGICAL_ANALYSIS_PASS`.

The minimal example is entirely synthetic and includes:

- `examples/minimal/input/raw_em.tif`;
- aligned MitoNet-like and OmniEM-like probability volumes;
- a MitoNet-like panoptic instance volume;
- the selected refinement checkpoint (`refinement_C04.pt`);
- the G011 reconstruction policy;
- frozen expected outputs and SHA-256 hashes.

The verifier runs CPU inference, checks the input and reference hashes, compares the probability output within an
absolute tolerance of 1e-6, requires exact semantic and instance arrays, and verifies stable provenance fields.

## 4. Optional refinement module

All volumes must be three-dimensional Z,Y,X TIFF or NPY arrays with identical geometry. Probability arrays must be
finite and lie in [0,1]. Spacing is provided as positive `Z,Y,X` nanometres. MitoCensus does not create upstream
priors; obtain and run MitoNet and OmniEM from their official distributions and align their probability outputs
before this step.

```bash
mitocensus infer-from-priors \
  --raw sample_raw.tif \
  --mitonet-probability sample_mitonet_probability.tif \
  --omniem-probability sample_omniem_probability.tif \
  --checkpoint refinement_C04.pt \
  --spacing 30,16,16 \
  --output sample_refined
```

The output directory contains:

- `refined_probability.tif`, float32;
- `refined_semantic.tif`, uint8 at the fixed default threshold 0.5;
- `manifest.json`, including hashes, geometry, normalization, threshold, model configuration and mean gate weight.

To reconstruct conservative 3-D instances, add aligned MitoNet panoptic instances and the released policy:

```bash
mitocensus infer-from-priors ... \
  --mitonet-instances sample_mitonet_instances.tif \
  --policy-json g011_policy.json
```

This adds `refined_instances.tif`. The command accepts either the standalone G011 policy or the full selected-model
contract. It validates the v4 policy fields and does not silently fall back to an older reconstruction route.

## 5. Interpret the result conservatively

Voxel overlap does not certify objects: check section support and split labels before reporting shape statistics,
and pool edge-weighted objects across crop positions rather than trusting one small crop. The refinement module met
the primary endpoint of the held-out kidney set but not that of the three-dataset set; validate the complete pipeline
on representative labelled volumes from the intended acquisition before morphometry or biological inference. Do not
treat crops, slices, voxels or objects from one acquisition as independent biological replicates.

MitoNet and OmniEM remain separately obtained upstream tools. Their source, weights and licenses are not
redistributed by this MIT-licensed package.
