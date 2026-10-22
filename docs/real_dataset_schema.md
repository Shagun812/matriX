# Real calibration and validation dataset schema

This manifest describes real, provenance-bearing measurements. It does not
contain or represent simulator `Case` objects. Files named below are paths
relative to the manifest unless absolute paths are supplied.

```yaml
schema_version: 1
dataset_id: lab-study-2026-01
samples:
  - sample_id: specimen-001
    material: tio2
    ground_truth: H1
    ground_truth_confidence: 0.95
    composition: measurements/specimen-001-composition.csv
    xrd: measurements/specimen-001-xrd.csv
    optical: measurements/specimen-001-optical.csv
    metadata: measurements/specimen-001-metadata.yaml
    xrd_reference: references/anatase-rutile-profile.csv
    repeat:
      composition: measurements/specimen-001-composition-repeat.csv
      xrd: measurements/specimen-001-xrd-repeat.csv
      optical: measurements/specimen-001-optical-repeat.csv
    provenance:
      source: laboratory or study identifier
      measurement_ids:
        composition: unique-measurement-id
        xrd: unique-measurement-id
        optical: unique-measurement-id
```

`ground_truth` is one of `H1`, `H2`, `H3`, `H4`, `UNRESOLVED`, or `UNKNOWN`.
Missing ground truth is normalized to `UNKNOWN`; the
`ground_truth_confidence` key is required and may be `null` when unknown.
Only labeled H1-H4 records contribute to supervised likelihood estimation.
`UNRESOLVED` and `UNKNOWN` are retained in evidence exports but excluded from
H1-H4 supervised calibration. Calibration requires at least one valid labeled
sample in each H1-H4 class.

All records require composition, XRD, optical, and metadata paths. Repeats are
optional per measurement. `provenance.source` and unique measurement IDs are
required. Sample metadata requirements and measurement CSV columns are
documented in the README real-data diagnosis section. The metadata YAML may
include `xrd_reference` provenance matching the `xrd_reference` path. Reference
profiles may be shared and are not treated as per-sample raw measurement
leakage.

The validator rejects duplicate sample IDs, duplicate measurement IDs,
duplicate raw measurement content (SHA-256), unsupported labels/materials,
missing files/fields, invalid units/uncertainties, and malformed arrays. It
does not repair input data.

Calibration splits are distinct datasets. Held-out validation fails if any
sample ID, raw measurement hash, or measurement ID overlaps the calibration
dataset. Dataset and model hashes, per-sample leakage signatures, class counts,
feature bin definitions, likelihoods, smoothing, prior selection, and decision
threshold are stored in the real model artifact.

The supplied `tests/fixtures/real_data/` inputs are synthetic software-test
fixtures. They are not labeled experimental examples and must not be used for
scientific calibration or reported as experimental validation.
