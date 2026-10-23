# MitoCensus release notes

## 0.4.0 (unreleased)

The section check is on by default, the split screen flags rather than excludes, and every analysis writes a visual
record. Measured values do not change; which labels enter whole-object morphometry by default does.

- **Section-extent check on by default (3-D).** `--min-axial-extent-nm` (default 90 nm) excludes labels whose extent
  across sections, distinct sections times section thickness, is below the threshold from whole-object morphometry
  (reason `axial_extent_below_minimum`, class `not_evaluable_thin_profile`); they stay in abundance summaries. The
  threshold is physical, so it applies to serial sections and to isotropic data alike. `--min-axial-sections` (the
  0.3.x count-based form) is still available; `none` turns the extent check off.
- **Split screen as a review flag by default.** `--split-screen-min-iou 0.4` is on by default with
  `--split-screen-action flag`: flagged labels stay QC-passing and are marked for review (`split_label_candidate`,
  `split_label_group`, `manual_review_recommended`). `--split-screen-action exclude` restores the 0.3.x exclusion.
  `sample_summary.json` lists the pairs the screen found (`quality_control_v2.split_label_links`: the label that ends,
  its end section, the label under it, that section and the footprint IoU).
- **Visual record.** `analyze-biological` writes `visual_record/index.html` (when matplotlib is installed and the
  report is on; `--no-visual-record` skips it): the check outcome, the labels excluded as too thin, QC-passing objects
  by shape class across volume quantiles, one compared pair per group of possible splits with its IoU, and object
  pages with the sections an object spans, its surface, its skeleton after pruning and the voxels counted by the
  line-filter, matrix and weak-boundary measures.
  The displayed layers come from the measurement code; `visual_record.json` records that the features recomputed from
  them equal `objects.csv`. `mitocensus visual-record` rewrites the record from a finished analysis.
- **PyTorch only for the refinement module.** Measurement, comparison and the visual record no longer import PyTorch;
  install it with `python -m pip install '.[refinement]'` for `infer-from-priors`.
- With `--min-axial-extent-nm none --split-screen-min-iou none` (0.3.1 defaults) or `--min-axial-extent-nm none
  --min-axial-sections 3 --split-screen-action exclude` (0.3.1 "all checks"), 0.4.0 writes the same `objects.csv` and
  `contacts.csv` as 0.3.1. Seven tests were added (44 in total).

## 0.3.1 (2026-10-22)

Measurement fixes found by a panel-by-panel review of the software paper's figures. Checks and options are unchanged;
measured values change.

- Pieces of a label smaller than `--min-voxels` are dropped before the checks, so a few stray voxels no longer make a
  whole object 'disconnected'. Dropped voxels are recorded per object (`ignored_fragment_voxels`).
- Skeletons are computed on cubic voxels at the finest voxel size; terminal branches shorter than the local diameter
  are treated as surface spurs; `branch_points` counts junctions of at least three real branches, and the skeleton
  length is measured along the skeleton. Before, the skeleton of anisotropic data was computed in voxel units, and
  spurs created branch-point voxels in compact or tube-shaped objects.
- The line-filter score and the matrix measures use interior voxels more than 48 nm from the boundary in the section
  plane (`line_filter_boundary_margin_nm`). Before, most counted line-filter voxels lay on the membrane band.
- Boundary support and the weak-boundary fraction use the in-section outline of each object. Before, faces along the
  section axis made up about half of the boundary voxels.
- The mitochondria-on-a-string candidate no longer requires branch and end points, because a string of beads is
  unbranched; it uses lobes, the neck ratio and elongation.
- Five tests on synthetic data cover these cases (37 tests in total).

## 0.3.0 (2026-10-15)

First public release.

### Object analysis

- `analyze-biological` checks and measures every label of an aligned mitochondrial instance mask from any segmenter,
  in one 2-D TEM field or one 3-D volume. It writes one row per label (`objects.csv`), close pairs (`contacts.csv`),
  the 13 function summaries with their denominators and claim boundaries (`sample_summary.json`), the calibration of
  the candidate flags (`reference_profile.json`), a review label map, a visual report and a hash manifest.
- `compare-biological` compares groups by biological unit: control-referenced endpoint deltas, Hedges g and
  biological-unit bootstrap intervals, with every sample recalibrated against a shared control profile.
- Labels touching the image or ROI border and labels of more than one connected component are excluded from
  whole-object morphometry and stay in the abundance outputs.
- Opt-in section-support rule `--min-axial-sections` and split-label screen `--split-screen-min-iou`
  (`--split-screen-max-gap-sections`), recorded per object and under `quality_control_v2` in the summary.
- Opt-in discrete Miles–Lantuéjoul edge-correction weights `--edge-correction-weights`, with weighted and unweighted
  mean sizes.
- Every object row carries `axial_sections` and `axial_extent_nm`.
- Empty numeric fields of object tables (for example no neck radius) are read as missing values, as they are when the
  analysis runs in memory.
- Reference profiles and analysis directories are read only under their MitoCensus schema; records of any other schema
  are rejected with an error that names the expected schema.

### Optional refinement module

- `infer-from-priors` runs the selected C04 checkpoint, which combines aligned MitoNet and OmniEM probabilities with
  five adjacent raw sections through a learned spatial gate and a bounded residual correction (752,701 parameters).
  Given MitoNet panoptic instances, it rebuilds 3-D instances with the deterministic G011 v4 reconstruction policy.
- The checkpoint ships as `examples/minimal/refinement_C04.pt` (SHA-256 `0a8bd8c7…`). Its weights and configuration
  are identical to those of the serialization cited by the frozen evaluation records (SHA-256 `c3dd763f…`).

### Reproducibility

- Exact conda environment (`environment.yml`) and a Linux-compatible Dockerfile.
- Synthetic 3-D minimal example with frozen CPU outputs and its verifier, and a verifier for the object analysis.
- 32 tests (16 core and 16 object analysis) with 134 parameterized subtests, all on synthetic data.
- `SHA256SUMS.txt` lists the SHA-256 checksum of every other file of the release.

### Scientific status

The refinement module showed a mean semantic Dice improvement of +0.0217 on eight selection-exposed development
volumes. The apparent central-field instance benefit contracted to +0.0018 in a 40-field sensitivity analysis.
Kidney-22 and replication-30 both failed their joint external contracts. The evidence does not support a
universal-superiority, stable-transport, biological-morphometry, clinical, prospective or causal claim for the module.

The object analysis provides measured image features and explicitly named exploratory candidates. It does not
validate fusion or fission rates, metabolic function, cell-death pathways or a clinical interpretation.

### Known limitations

- The complete development set was repeatedly selection-exposed.
- G011 was designed after observing development instance failures.
- Only 13 of 40 expanded development fields contained eligible interior reference objects.
- External data comprise four acquisitions without demonstrated donor independence.
- The refinement module requires separately obtained and aligned MitoNet and OmniEM predictions.
- The complete pretraining membership of EM-DINO/OmniEM is not locally auditable.
- A local Docker image build was not executed because the test Mac lacked a Docker engine.

### Release actions still required

Archive this release under a persistent DOI and obtain all-author approval before journal submission.
