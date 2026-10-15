# Cohort-comparison template

`cohort_manifest_template.csv` is a copy-ready input for
`mitocensus compare-biological`.

- `sample_id` identifies one analyzed TEM field or vEM volume and must be
  unique.
- `group` is the study arm used in the descriptive comparison.
- `biological_unit` is the independent animal, patient, organoid, cell culture
  replicate or other prespecified specimen. Multiple technical fields may use
  the same value; MitoCensus averages them before group comparison.
- `acquisition_batch` is strongly recommended. A batch column that is
  completely confounded with group causes a hard stop.
- `analysis_dir` points to the output of `mitocensus analyze-biological` and is
  resolved relative to the CSV file unless it is absolute.

Run:

```bash
mitocensus compare-biological \
  --manifest cohort_manifest_template.csv \
  --control-group control \
  --output cohort_response
```

Do not assign fields, sections, mitochondria or voxels as independent units.
The output is descriptive and exploratory; design-specific inferential models
remain the investigator's responsibility.
