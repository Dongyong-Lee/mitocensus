# MitoCensus release notes

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
