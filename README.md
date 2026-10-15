<p align="center">
  <img src="docs/images/mitocensus_banner.png" alt="MitoCensus: object-level quality control and measurement of mitochondria in volume electron microscopy" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-1F5F99"></a>
  <img alt="Python 3.10 or later" src="https://img.shields.io/badge/python-3.10%2B-1F5F99">
  <img alt="Version 0.3.0" src="https://img.shields.io/badge/version-0.3.0-57606A">
  <a href="CITATION.cff"><img alt="Cite this software" src="https://img.shields.io/badge/cite-CITATION.cff-CC79A7"></a>
</p>

**MitoCensus turns a volume electron microscopy stack and any aligned mitochondrial instance mask into an
object-linked measurement record.** Every label is traced to its source sections and checked before it is measured.
Excluded labels stay visible, edge bias from a finite crop can be corrected, and cohort statistics are computed by
biological unit rather than by object.

## Why check objects

Voxel-overlap scores such as Dice do not show whether a segmentation delivers whole mitochondria. Labels confined to
one or two sections, labels cut by the crop border and one mitochondrion split across sections can all score well, and
they change size and shape statistics. MitoCensus checks every label before it enters whole-object morphometry.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/mitocensus_object_checks_dark.png">
    <img alt="The object checks on synthetic shapes in a crop of eight serial sections: a QC-passing object, a border-truncated object, a disconnected label, a flat label in fewer than three sections and a mitochondrion split across sections." src="docs/images/mitocensus_object_checks_light.png" width="820">
  </picture>
</p>

## How it works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/mitocensus_workflow_dark.png">
    <img alt="Workflow: an EM volume and an instance mask pass through the object checks into 13 measurement functions, an object record per sample and a cohort comparison; refinement and edge weights are optional." src="docs/images/mitocensus_workflow_light.png" width="820">
  </picture>
</p>

| Command | What it does |
|---|---|
| `mitocensus analyze-biological` | Checks and measures every label of one sample; writes the object record |
| `mitocensus compare-biological` | Compares groups by biological unit from several analysed samples |
| `mitocensus infer-from-priors` | Optional refinement module that combines aligned MitoNet and OmniEM probabilities |

## What the evidence shows

Technical results on public volumes (MitoEM 2.0 and CellMap). Mitochondria, sections, crops and voxels are nested
technical observations; none of these results is a biological-replicate claim.

- **Voxel overlap does not certify objects.** On the held-out CellMap kidney set, the refined masks had the highest
  pooled Dice (0.929, against 0.909 for OmniEM and 0.801 for MitoNet), yet MitoNet produced the most
  reference-matched objects (46, against 31 and 24). About half of the refined and OmniEM objects that passed a
  standard border check spanned only one or two sections.
- **The section-support rule and the split-label screen** (`--min-axial-sections`, `--split-screen-min-iou`) removed
  these labels on the held-out set. They cut elongated shape calls by about two-thirds or more while keeping nearly
  all reference-matched objects.
- **Edge weights** (`--edge-correction-weights`) brought mean mitochondrial volume, pooled over crop positions, to
  within 6% of the whole-volume value for mitochondria that fit the crop, in the five development volumes with at
  least 100 whole objects. A single small crop holds too few whole objects for a reliable estimate in most tissues.
- **The optional refinement module** had the highest semantic Dice of three models in seven of eight development
  volumes (used for model selection). It met the primary endpoint of the held-out kidney set but not that of the
  three-dataset set, and neither set met all prespecified criteria. Validate it on the acquisition of interest before
  use.

## Install

```bash
conda env create -f environment.yml
conda activate mitocensus
python -m pip install .
```

`environment.yml` pins the tested library versions. With other versions, intensity-derived measurements can differ
by less than 0.08%, or by less than 1e-7 for values near zero; see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Verify

```bash
python scripts/verify_minimal_example.py       # expected: MINIMAL_EXAMPLE_PASS
python scripts/verify_biological_analysis.py   # expected: BIOLOGICAL_ANALYSIS_PASS
python -m pytest -q
```

The minimal example is synthetic: 3-D inputs, the selected refinement checkpoint, the reconstruction policy, the
expected outputs and their SHA-256 hashes. It runs on CPU and needs neither upstream segmenter. The 32 tests (134
parameterized subtests) use synthetic data only. `SHA256SUMS.txt` lists the SHA-256 checksum of every other file of the
release; `shasum -a 256 -c SHA256SUMS.txt` checks them.

## Quick start

Measure one sample. The instance mask can come from any segmenter; for 2-D TEM fields give `Y,X` spacing.

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

The last line switches on the section-support rule, the split-label screen and edge weights. They are opt-in: without
them, only the border and connectivity checks decide which labels enter whole-object morphometry.

Compare groups by biological unit, with one manifest row per analysed field or volume
([template](examples/biological/cohort_manifest_template.csv)):

```bash
mitocensus compare-biological \
  --manifest cohort.csv \
  --control-group control \
  --output cohort_response
```

## Outputs

| File | Contents |
|---|---|
| `objects.csv` | One row per label: measured features, the checks it passed or failed and candidate flags |
| `sample_summary.json` | The 13 function summaries with their denominators and claim boundaries |
| `contacts.csv` | Close mitochondrion pairs and junction-like review candidates |
| `manual_review_instances.tif` | Flagged object IDs only, aligned to the input |
| `biological_report.png` | A compact visual report |
| `analysis_manifest.json` | SHA-256 hashes of inputs and outputs |
| `group_response.csv` (cohort) | Control-referenced deltas, Hedges g and biological-unit bootstrap intervals |

Excluded labels remain in the abundance outputs, and every trait summary reports its denominator of QC-passing
objects. The full input, output and claim contract is in [docs/BIOLOGICAL_ANALYSIS.md](docs/BIOLOGICAL_ANALYSIS.md).

## Interpreting the outputs

- Shape classes describe static geometry. They do not measure fission or fusion events.
- Line-filter, matrix and boundary features are intensity-based and depend on the acquisition. Compare them only
  within a documented fixation, staining and imaging protocol.
- Rare shapes, mitochondria-on-a-string or nanotunnel-like geometry, close pairs and stress-associated patterns are
  review candidates. Inspect the raw serial sections before reporting them.
- The outputs are exploratory image phenotypes. They do not measure molecular activity, metabolic function, a
  cell-death mechanism or a clinical state.

## Optional refinement module

`infer-from-priors` combines aligned MitoNet and OmniEM foreground probabilities with five adjacent raw sections,
through a learned spatial gate and a bounded residual correction, and can rebuild 3-D instances from MitoNet panoptic
objects with the released reconstruction policy. Obtain and run the upstream tools separately under their own
licenses. All eight MitoEM 2.0 development volumes were used for model selection, and the reconstruction policy was
designed after development failures. Held-out predictions were sealed before the references were read, but the four
CellMap acquisitions are not demonstrated independent donors or biological replicates. See the
[quick start](docs/QUICKSTART.md), the [model card](docs/MODEL_CARD.md) and the
[DOME-ML checklist](docs/DOME_ML_CHECKLIST.md).

## Documentation

- [Quick start](docs/QUICKSTART.md): step-by-step use of the three commands
- [Object analysis](docs/BIOLOGICAL_ANALYSIS.md): inputs, outputs, object checks and the interpretation contract
- [Reproducibility](docs/REPRODUCIBILITY.md): tested environments and expected numerical differences
- [Model card](docs/MODEL_CARD.md) and [DOME-ML checklist](docs/DOME_ML_CHECKLIST.md): the optional refinement module
- [Release notes](RELEASE_NOTES.md)

## Citing MitoCensus

A manuscript describing MitoCensus is in preparation. Until it is published, cite the software with the metadata in
[CITATION.cff](CITATION.cff); GitHub's **Cite this repository** button gives APA and BibTeX.

## License

MitoCensus source code is released under the [MIT license](LICENSE). Third-party code, checkpoints and datasets keep
their own licenses and are not redistributed.
