# Real calibration and held-out data

This folder is a structure/template only. No experimental measurements,
ground-truth labels, measurement IDs, or provenance have been supplied.
Consequently, these manifests are intentionally not ready to validate or
calibrate.

Add genuine measurement files under `measurements/` and edit the manifests:

- `calibration.yaml` must contain real labeled records covering H1, H2, H3,
  and H4.
- `held-out.yaml` must contain separate real records. Do not reuse sample IDs,
  raw measurement files, or measurement IDs from calibration.

Each sample needs composition, XRD, optical, and metadata files, plus the
measurement provenance described in
[`../../docs/real_dataset_schema.md`](../../docs/real_dataset_schema.md).
The manifest paths are relative to this folder.

The manifest examples are comments, not data. Replace every angle-bracket
placeholder with verified source information; do not invent IDs, labels,
uncertainties, metadata, or measurements. These templates are not expected to
pass dataset validation while `samples` is empty.
