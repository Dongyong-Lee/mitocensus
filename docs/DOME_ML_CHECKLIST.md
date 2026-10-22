# DOME-ML reporting checklist for the MitoCensus refinement module

This checklist follows the Data, Optimization, Model, and Evaluation structure
recommended for supervised machine learning in biology. It records what is
complete locally and what remains an author or public-release action.

## Data

- **Task:** semantic and instance segmentation of mitochondria in 3D volume EM.
- **Development corpus:** eight MitoEM 2.0 volumes; all labels and derived
  caches are selection-exposed.
- **External corpus:** CellMap kidney-22 from one acquisition and replication-30
  from three additional acquisitions.
- **Independent reporting units:** complete development volume and external
  acquisition. Crops, sections, voxels, and objects are nested technical data.
- **Biological independence:** not established for the four external
  acquisitions; no biological-replicate claim.
- **Controlled access:** none.
- **External-use boundary:** no target raw image or statistic entered training,
  calibration, threshold selection, normalization fitting, or adaptation.
- **Provenance:** dataset identity, voxel spacing, exposure state, exclusions,
  and input hashes are recorded in frozen contracts and manifests.
- **Data availability:** MitoEM 2.0 DOI 10.5281/zenodo.17635006; CellMap
  collection DOI 10.25378/janelia.c.7456966.

## Optimization

- One hundred hypotheses were registered across eight causal families.
- Stage A used a common 96-pixel, two-epoch, one-seed mechanism screen.
- Stage B completed 26 candidates x 3 fixed seeds.
- Stage C trained five full models for 10 epochs under a common budget.
- C04 used AdamW, learning rate 0.00035, weight decay 0.0001, batch size 8,
  1,024 examples per epoch, CutMix, and seed 2026081902.
- Region loss was 0.5 BCE plus 0.5 soft Dice; boundary BCE, distance smooth-L1,
  and correction L1 weights were 0.20, 0.10, and 0.002.
- H096, a DINOv3-related imported execution, was audit-only and ineligible to
  define the final model.
- G011 was selected only after 60, 72, and 96 development policy searches; this
  post-outcome amendment is clearly marked selection-exposed.

## Model

- **Selected model:** C04, five-section CutMix-attention network.
- **Parameters:** 752,701.
- **Inputs:** five normalized raw sections, MitoNet probability, OmniEM
  probability, and anisotropy class derived only from voxel spacing.
- **Core:** residual encoder-decoder, learned spatial prior gate, bounded logit
  correction, attention, and region/boundary/distance outputs.
- **Identity exclusion:** no specimen identifier or external target label.
- **Semantic threshold:** 0.5.
- **Instance route:** deterministic G011 reconstruction anchored to MitoNet
  panoptic objects.
- **Checkpoint digest:**
  `0a8bd8c7bfd2f8adcf9d310852dea68c297597bdd708362c52e786a8598c496a`
  (shipped file); the frozen evaluation records cite the serialization
  `c3dd763f878de379ad3ec612f65d984d18ce55e698cbab909f420a732e851ae5`,
  which has identical weights and configuration.
- **Third-party boundary:** upstream code and weights are not redistributed.

## Evaluation

- **Development comparator:** better MitoNet or OmniEM specialist within each
  complete volume; instance comparator selected separately within volume.
- **External comparator:** identity fixed in each pre-outcome panel contract;
  per-crop best specialist is diagnostic only.
- **Endpoints:** complete-volume semantic Dice, physical boundary F1, instance
  F1 at IoU 0.75, worst-unit delta, foreground burden, count safety, and matched
  physical-trait relative errors.
- **Development result:** mean semantic Dice delta +0.021684; mean boundary-F1
  delta +0.030231; 7/8 volumes positive; worst semantic delta -0.0000416.
- **Spatial sensitivity:** 13/40 fields evaluable; volume-macro instance delta
  +0.001835; two of eight volume means positive; two of four traits improved.
- **Kidney-22:** joint FAIL despite pooled Dice delta +0.020260.
- **Replication-30:** joint FAIL; acquisition-macro Dice delta -0.018498,
  boundary-F1 delta -0.132390, and SUM159 delta -0.088919.
- **Combined external:** FAIL.
- **Statistics:** descriptive only; no P value, confidence interval, or
  biological-replicate inference.
- **Visual QA:** every one of 17 unique figure pages inspected at publication
  size and original resolution; absolute failures retained.

## Reproducibility and release

- Exact tested conda environment with Python 3.11. Between the two library sets
  that reproduce the stored records, intensity-derived measurements differ by
  less than 0.08%, or by less than 1e-7 for values near zero; see
  `docs/REPRODUCIBILITY.md`.
- PEP 517 package and `mitocensus` command-line entry point.
- 37 tests (16 core and 21 object analysis) and 134 parameterized subtests
  pass on synthetic data, without non-distributed records.
- CLI reads G011 from either a standalone policy or selected-model contract and
  invokes the v4 reconstruction implementation.
- Synthetic minimal example includes input, checkpoint, policy, expected
  output, stable-field comparison, and SHA-256 hashes.
- Linux-compatible Dockerfile is present; no local Docker engine was available,
  so an executed image build remains unverified.
- Model card, quick start, contracts, results summaries, and provenance
  manifests are included.
- Public repository: https://github.com/Dongyong-Lee/mitocensus, with author
  metadata in `CITATION.cff`. An archived DOI, RRID/bio.tools identifiers and
  all-author approval remain release actions and are not fabricated locally.
