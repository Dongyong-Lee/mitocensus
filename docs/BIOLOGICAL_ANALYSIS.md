# MitoCensus object analysis (`analyze-biological`, `compare-biological`)

MitoCensus converts an aligned mitochondrial instance mask and its raw TEM/vEM image into an object-linked
measurement record: one row per label, the checks that decide which labels count as whole objects, and transparent
exploratory features. It supports one 2-D TEM field (`Y,X`) or one 3-D volume (`Z,Y,X`) and accepts instance masks
from any segmenter.

The analysis does **not** diagnose stress, cell death or mitophagy, and a static image does not measure
fusion/fission events or rates. Candidate labels always retain `-like`, `-associated`, `pattern` or `candidate` in
machine-readable and human-readable outputs.

Install the report dependency with `python -m pip install '.[biology]'`. Both commands can run table/JSON analysis
without Matplotlib by adding `--no-report`.

`analyze-biological` accepts reference profiles of schema `mitocensus.biological-reference-profile.v1`, and
`compare-biological` accepts analysis directories of schema `mitocensus.biological-analysis.v1`; records of any other
schema are rejected with an error that names the expected schema.

## Implemented functions

| # | User-facing function | Returned measurements |
|---:|---|---|
| 1 | Mitochondrial amount | object count, QC-passing object count, mitochondrial area/volume fraction and ROI-normalized density |
| 2 | Size and swelling | physical size, equivalent diameter, roundness, swelling-like and megamitochondrion fractions |
| 3 | Shape classes (fusion/fission-associated morphology) | fragmented, intermediate, elongated (fusion-associated) and branched (hyperfused-like) composition; constriction candidates |
| 4 | Line-filter score | fraction of the interior above a local line-filter threshold, spacing, coherence and low-line-filter fraction |
| 5 | Matrix appearance | normalized electron density, heterogeneity, lucent fraction and pale/dense-like composition |
| 6 | Boundary features | weak-boundary fraction and a transparent four-component composite review score |
| 7 | Population heterogeneity | size CV, shape-class entropy, rare-state burden and review burden |
| 8 | Disease or treatment response | biological-unit endpoint table, group deltas, Hedges g and sample-level bootstrap intervals |
| 9 | Spatial organization | nearest-neighbour distance, complete-spatial-randomness ratio, clustering candidates and orientation coherence |
| 10 | Rare morphologies | giant, spheroid, ring/fenestrated candidates |
| 11 | Mitochondria-on-a-string and nanotunnel-like geometry | lobe/neck/skeleton-based candidates requiring raw-image review |
| 12 | Mitochondrion–mitochondrion pairs | close pairs, voxel-boundary gap approximation, interface measure and junction-like candidates |
| 13 | Cell-state-associated patterns | stress-associated review queue: composite, small-dense, swollen weak-boundary and double-membrane patterns |

## Analyze one sample

Use any aligned non-negative integer label map, for example the instance output of MitoNet, OmniEM or the optional
refinement module. A cell or tissue ROI is optional but recommended for abundance estimates.

```bash
mitocensus analyze-biological \
  --raw sample_raw.tif \
  --instances sample_instances.tif \
  --spacing 30,16,16 \
  --sample-id specimen-01 \
  --group control \
  --roi specimen-01_cell_roi.tif \
  --output specimen-01_census
```

For 2-D TEM, provide `Y,X` spacing:

```bash
mitocensus analyze-biological \
  --raw field_01.tif \
  --instances field_01_instances.tif \
  --spacing 8,8 \
  --sample-id animal-01 \
  --output animal-01_census
```

The output directory contains:

- `objects.csv`: one row per label with measured features, the checks it passed or failed and explicit exploratory
  candidate flags;
- `contacts.csv`: close mitochondrion pairs and junction-like review candidates (in `objects.csv`, `close_pair_count`
  is the number of close partners of each object; in the summary, `mitochondrion_pairs.close_pair_count` is the number
  of close pairs in the sample);
- `sample_summary.json`: all 13 user-facing function summaries and claim boundaries;
- `reference_profile.json`: the feature calibration used for candidate flags;
- `manual_review_instances.tif`: only flagged object IDs, aligned to the input;
- `biological_report.png`: a compact visual report;
- `analysis_manifest.json`: input/output SHA-256 hashes and the claim boundaries copied into the run provenance.

Without an external reference profile, candidate flags use the sample itself as an explicitly recorded exploratory
reference. For cross-condition work, use the cohort command below so every sample is recalibrated against a shared,
biological-unit-balanced control profile.

## Object checks

Whole-object morphology excludes labels touching the analysis-volume boundary, because their size, topology and
connectivity are truncated, and labels consisting of more than one connected component. Pieces of a label smaller
than `--min-voxels` are dropped before the checks (their number of voxels is kept in `ignored_fragment_voxels`), so a
few stray voxels do not make an otherwise whole object disconnected. Excluded labels remain in
abundance and foreground-fraction outputs; every trait summary reports its denominator of QC-passing objects, and
the object table records image-border, ROI-boundary and disconnected-label reasons separately. Fewer than 10
QC-passing reference objects trigger a machine-readable calibration warning.

### Measurement geometry

- Skeletons are computed on cubic voxels at the finest voxel size, so anisotropic serial-section data are not
  compressed along the section axis. A branch counts only when the skeleton extends beyond its junction by more than
  the local diameter (twice the larger of the 3-D and in-section inscribed radius); shorter branches are surface
  spurs and are ignored. `branch_points` counts junctions where at least three such branches meet, and
  `skeleton_length_um_proxy` is the length of the skeleton without spurs.
- The line-filter score and the matrix measures use interior voxels more than 48 nm from the object boundary in the
  section plane, so the dark membrane band at the boundary does not count as line structure.
- Boundary support and the weak-boundary fraction use the outline of each section; faces along the section axis,
  which carry no membrane contrast in serial sections, are left out.

### Optional section-support rule, split-label screen and edge weights (3-D)

Three opt-in options tighten whole-object morphometry. With the defaults, only the border and connectivity checks
above apply; the outputs still carry four object columns (`axial_sections`, `axial_extent_nm`,
`split_label_candidate`, `split_label_group`) and five summary entries (the four option settings under
`configuration`, and `quality_control_v2`) that record the options.

```bash
mitocensus analyze-biological ... \
  --min-axial-sections 3 \
  --split-screen-min-iou 0.4 --split-screen-max-gap-sections 3 \
  --edge-correction-weights
```

- `--min-axial-sections N` excludes objects present in fewer than `N` serial sections from whole-object morphometry
  (reason `axial_sections_below_minimum`, class `not_evaluable_thin_profile`). With little extent across sections,
  the minor axis of such objects is set by the voxel spacing, so their shape values are not interpretable.
- `--split-screen-min-iou T` flags labels whose first- or last-section footprint continues, within
  `--split-screen-max-gap-sections` sections, in another border-free single-component label with footprint
  IoU >= `T` (`split_label_candidate`, `split_label_group`; class `not_evaluable_split_label_candidate`). Labels are
  never merged or re-measured. The screen detects splits across sections, not side-by-side splits.
- `--edge-correction-weights` writes the discrete Miles–Lantuéjoul weight `prod_i L_i / (L_i - b_i - 1)`
  (`edge_correction_weight`) for border-free objects and adds weighted and unweighted mean sizes to
  `sample_summary.json` (`edge_correction`). The weighted mean of one small region with few objects is itself a noisy
  ratio estimate; pool weighted objects across regions, and remember that objects larger than the region can never
  be recovered.

Every object row also carries `axial_sections` and `axial_extent_nm`, and `sample_summary.json` records the settings
and flagged groups under `quality_control_v2`. Flagged objects stay in abundance summaries.

## Compare disease or treatment groups

Create a CSV with one row per analyzed field or volume and identify its independent biological unit explicitly:

```csv
sample_id,group,biological_unit,acquisition_batch,analysis_dir
control-01,control,mouse-01,batch-01,control-01_census
control-02,control,mouse-02,batch-01,control-02_census
treated-01,treated,mouse-11,batch-01,treated-01_census
treated-02,treated,mouse-12,batch-01,treated-02_census
```

A copy-ready version is provided at `examples/biological/cohort_manifest_template.csv`. Multiple technical fields may
share one `biological_unit`; each field still requires a unique `sample_id`.

Then run:

```bash
mitocensus compare-biological \
  --manifest cohort.csv \
  --control-group control \
  --output cohort_response
```

The command writes:

- `sample_endpoints.csv`: one row per biological unit; repeated fields or technical volumes sharing a
  `biological_unit` are averaged within that unit before any between-group effect size is calculated;
- `group_response.csv`: control-referenced endpoint deltas, Hedges g and biological-unit bootstrap intervals across
  all 13 feature families (an endpoint is omitted only when neither group has finite biological-unit values);
- `control_calibrated_objects.csv`: all per-object candidate flags recalculated against the shared
  biological-unit-balanced control profile;
- `control_calibrated_sample_summaries.json`: the corresponding 13-function summaries for every technical sample;
- `control_reference_profile.json`: sample-balanced control calibration;
- `disease_treatment_response.json`: the function-8 result;
- `cohort_response_report.png`: biological-unit dot plots;
- `cohort_manifest.json`: cohort-manifest and output hashes.

The cohort provenance also records SHA-256 hashes for every input sample's object table, contact table, sample
summary and reference profile.

No object-level P value is produced. A biological unit cannot occur in more than one group, and duplicate
`sample_id` values are rejected. If formal inference is required, its design must match the independent biological
unit, pairing, repeated measures, batch structure and prespecified endpoint family of the study.

`acquisition_batch` is optional but strongly recommended. When present it must be complete; if every batch contains
only one study group, the command stops because preparation/acquisition is completely confounded with biology.

A cohort cannot mix 2-D TEM fields and 3-D vEM volumes. Reference profiles also carry their dimensionality and
physical-size unit; incompatible profiles are rejected before analysis.

## Interpretation contract

- `fragmented_fission_associated` is a static shape class, not observed DRP1 activity or a fission event.
- `elongated_fusion_associated` and `branched_hyperfused_like` do not measure a fusion rate.
- `line_filter_fraction` is an intensity-based texture feature (Sato line filter), not a crista segmentation, until
  validated against a dedicated inner-membrane reference.
- Matrix and membrane density are acquisition dependent; compare only within a documented fixation, staining and
  imaging contract.
- Mitochondria-on-a-string, nanotunnel-like, junction-like, double-membrane and other cell-state-associated outputs
  are review queues. Inspect raw serial sections before reporting them.
- 3-D topology is required for strong ring, fenestration, connectivity, mitochondria-on-a-string and nanotunnel
  interpretations. A 2-D profile can be a sectioning artifact.
- Objects smaller than `--min-voxels` are excluded from both the retained object count and mitochondrial area/volume
  fraction; record this threshold in every study protocol.
- `--maximum-objects` is a speed-limited smoke-test option, not a study-analysis option. The summary records whether
  it truncated the input, in which case abundance and population endpoints are incomplete.
