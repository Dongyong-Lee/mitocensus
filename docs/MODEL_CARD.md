# MitoCensus refinement module: model card

## Model identity

- Selected semantic model: C04, `five_slice_cutmix_attention_full`.
- Trainable parameters: 752,701.
- Selected reconstruction policy: G011.
- Checkpoint SHA-256 (`examples/minimal/refinement_C04.pt`):
  `0a8bd8c7bfd2f8adcf9d310852dea68c297597bdd708362c52e786a8598c496a`.
  Its weights and configuration are identical to those of the serialization
  cited by the frozen evaluation records
  (`c3dd763f878de379ad3ec612f65d984d18ce55e698cbab909f420a732e851ae5`).
- Original-code license: MIT.
- Release state: version 0.3.0, public at https://github.com/Dongyong-Lee/mitocensus;
  a persistent archive identifier (DOI) is still required.

## Intended use

The optional refinement module of MitoCensus refines mitochondrial segmentation in 3D volume electron microscopy
when aligned raw sections, MitoNet foreground probabilities, and OmniEM
foreground probabilities are available. It is intended for method development,
technical benchmarking, and acquisition-specific evaluation. It is a refinement
layer that combines the two upstream priors, not a replacement license or redistribution channel for either upstream
segmenter.

The current evidence does not support direct use for biological morphometry,
clinical decisions, prospective inference, or claims of universal superiority.
Representative labelled volumes from the intended acquisition should be used
to validate the complete pipeline before substantive interpretation.

The MitoCensus object analysis (`analyze-biological`) measures and quality-controls
the objects of any aligned instance mask, including this module's output. Its
`-like`, `-associated`, `pattern` and `candidate` outputs are review and hypothesis-
generation aids, not validation of biological pathways or clinical states.

## Inputs and outputs

- Input geometry: one Z,Y,X raw EM volume and two foreground-probability volumes
  with identical geometry.
- Required metadata: positive Z,Y,X voxel spacing in nanometres.
- Semantic output: float32 foreground probability and uint8 mask at threshold
  0.5 by default.
- Optional instance output: 3D instance labels when an aligned MitoNet panoptic
  map and G011 policy are supplied.
- Provenance output: configuration, geometry, normalization, threshold, mean
  gate weight, and SHA-256 hashes for checkpoint, inputs, and outputs.
- Downstream object analysis: per-object physical morphology, line-filter
  score, matrix appearance, spatial organization, candidate review labels and
  biological-unit group comparisons. These features are not model outputs and
  inherit segmentation and acquisition error.

## Model mechanism

Five adjacent raw sections enter a residual encoder-decoder. A learned spatial
gate forms a local convex mixture of clipped MitoNet and OmniEM logits. A raw-
image-conditioned correction, bounded by `3 tanh(.)`, adjusts the mixture.
Region, boundary, and distance heads share decoder features. The correction
head starts at zero and the gate starts toward MitoNet, so optimization begins
from a specialist prior rather than unconstrained resegmentation.

G011 anchors instance reconstruction to eligible MitoNet panoptic objects. It
activates a contraction guard below a 0.95 candidate-to-primary foreground
ratio, permits a 24-nm primary extension, excludes objects below 500 native
voxels or touching the evaluation border, and limits zero-primary rescue to
four objects with at least 0.5 candidate overlap.

## Training and development exposure

C04 used AdamW, learning rate 0.00035, weight decay 0.0001, batch size 8,
1,024 examples per epoch, 10 epochs, CutMix, and seed 2026081902. One hundred
mechanism hypotheses were screened; 26 candidates were replicated over three
seeds; five full models were evaluated on eight native MitoEM 2.0 volumes.

Every MitoEM 2.0 volume was label-exposed during selection. G011 was designed
after all initial full models failed the instance gate. Its positive central-
field result is therefore post-outcome development evidence and has high
overfitting risk.

## Evaluation results

| Evaluation | Result | Decision |
|---|---|---|
| Eight development volumes | Mean semantic Dice delta +0.0217; physical boundary-F1 delta +0.0302; 7/8 positive | Development only |
| Central instance fields | Mean F1@0.75 delta +0.0704; 3/4 traits improved | Selection-exposed |
| Forty-field sensitivity | Volume-macro instance delta +0.0018; 13/40 evaluable; 2/4 traits improved | Narrowed morphometry claim |
| Kidney-22 | Pooled Dice +0.0203; pooled instance-F1 delta +0.0161; 9/22 crops improved; worst -0.0818 | FAIL |
| Replication-30 | Acquisition-macro Dice -0.0185; boundary -0.1324; SUM159 -0.0889 | FAIL |
| Combined external | Four acquisitions and 52 technical crops | FAIL |

The external result rejects a claim of stable transport. A positive pooled
kidney metric does not override failed instance, consistency, and worst-unit
gates.

## Independent-unit and fairness boundary

Complete development volumes and external acquisitions are the reporting
units. Patches, fields, crops, sections, voxels, and mitochondria are nested
technical observations. The four external acquisitions are not demonstrated
independent donors or biological replicates. No inferential P value or
confidence interval is claimed.

MitoNet and OmniEM were separately frozen comparators. The complete pretraining
membership of EM-DINO/OmniEM was not locally auditable, so overlap with public
EM imagery cannot be excluded. H096, the strongest DINOv3-related screen, was
retained as audit-only and did not define C04.

## Known failure modes

- Near-zero absolute performance in the MitoEM 2.0 Jurkat development volume.
- Negative acquisition-level transfer in SUM159.
- Boundary deterioration in the three-acquisition external panel.
- Semantic improvements that contract, split, merge, or miscount instances.
- Sparse eligible interior objects after physical-size and border exclusions.
- Potential distribution shift in intensity, anisotropy, preparation, organelle
  prevalence, upstream-prior calibration, and acquisition modality.

## Safety and use recommendations

1. Generate both upstream priors with fixed, documented versions.
2. Verify ZYX alignment and physical spacing before inference.
3. Inspect raw/reference/error overlays on representative labelled volumes.
4. Report complete-volume or acquisition-level behavior and worst units.
5. Revalidate instance reconstruction and trait error separately from semantic
   Dice.
6. Abstain from substantive biological interpretation when the acquisition-
   specific validation fails.

## Reproducibility

The release includes an exact conda environment, PEP 517 package, CLI,
Linux-compatible Dockerfile, 32 tests (16 core and 16 object analysis) with 134
parameterized subtests, all on synthetic data, synthetic minimal inputs, frozen
CPU outputs, dedicated core and biological-analysis verifiers, machine-readable
contracts, and SHA-256 manifests. A Docker engine was unavailable on the tested
Mac, so a container build is not claimed as locally executed.

## License and redistribution

MitoCensus source code is MIT licensed. Third-party code, checkpoints,
and datasets are not relicensed or redistributed. Users must obtain MitoNet,
OmniEM, MitoEM 2.0, and CellMap materials from their official sources and obey
their terms.
