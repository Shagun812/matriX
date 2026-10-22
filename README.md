# matriX — Demo Run and Real-Data Validation Guide

matriX is a materials-characterization diagnostic system for TiO2 evidence. It evaluates whether a sample can be diagnosed from composition, XRD, and optical measurements using an evidence-first workflow, with a real-data calibration path and a synthetic demo-data path for local validation and presentations.

This repository includes:

- a real-data validation and calibration workflow
- a demo dataset that follows the real manifest schema
- a one-click script to run a complete demo end to end
- a command-line diagnosis tool that reads measurement files and produces a report

The project is intended for demo runs and local validation. The synthetic demo files are not experimental data and must not be presented as scientific measurements.

## 1. Project overview

The repository supports two common workflows:

1. Demo workflow
   - uses the generated synthetic sample files under `data/real/`
   - validates the YAML manifests
   - trains a demo calibration model
   - diagnoses all demo samples
   - writes reports under `results/`

2. Real-data workflow
   - accepts user-supplied measurement files
   - validates structure and provenance
   - calibrates a model on labeled H1-H4 samples
   - evaluates a held-out dataset
   - diagnoses a new sample with the calibrated model

## 2. Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_real_demo.sh
├── data/
│   └── real/
│       ├── README.md
│       ├── calibration.yaml
│       ├── held-out.yaml
│       ├── demo-calibration/
│       │   └── real_model.json
│       ├── measurements/
│       │   ├── mock-cal-001-composition.csv
│       │   ├── mock-cal-001-xrd.csv
│       │   ├── mock-cal-001-optical.csv
│       │   ├── mock-cal-001-metadata.yaml
│       │   ├── mock-cal-001-composition-repeat.csv
│       │   ├── mock-cal-001-xrd-repeat.csv
│       │   ├── mock-cal-001-optical-repeat.csv
│       │   └── ...
│       └── references/
│           └── anatase-rutile-profile.csv
├── results/
│   ├── mock-cal-001-diagnosis/
│   ├── mock-cal-002-diagnosis/
│   ├── mock-cal-003-diagnosis/
│   ├── mock-cal-004-diagnosis/
│   └── matrix-real-demo-full/
├── matrix/
│   ├── __init__.py
│   ├── __main__.py
│   ├── real_cli.py
│   ├── real_data.py
│   ├── real_dataset.py
│   ├── real_calibration.py
│   ├── real_calibration_cli.py
│   └── ...
├── tests/
│   └── ...
└── .venv/
```

## 3. Setup

### 3.1 Create and activate a virtual environment

```bash
cd /path/to/matriX
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

If you prefer a smaller environment for the demo, the project also supports direct package installation from `requirements.txt`.

## 4. Quick start: run the full demo

The easiest way to run the project is the included script:

```bash
cd /path/to/matriX
chmod +x run_real_demo.sh
./run_real_demo.sh
```

This script will:

- validate both calibration and held-out YAML manifests
- create the calibration model if it does not already exist
- run the diagnosis tool for all four demo samples
- save results under `results/`

The script is designed to give you a one-click demo without having to remember the individual commands.

## 5. Manual demo workflow

If you want to run the same steps manually, use the following sequence.

### 5.1 Validate datasets

```bash
python -m matrix real-dataset-validate --dataset data/real/calibration.yaml
python -m matrix real-dataset-validate --dataset data/real/held-out.yaml
```

Both should return `valid: true`.

### 5.2 Calibrate the demo model

```bash
python -m matrix real-calibrate \
  --dataset data/real/calibration.yaml \
  --out data/real/demo-calibration \
  --decision-threshold 0.7
```

This writes the model artifact to:

- `data/real/demo-calibration/real_model.json`

### 5.3 Run diagnosis on a demo sample

```bash
python -m matrix diagnose \
  --composition data/real/measurements/mock-cal-001-composition.csv \
  --xrd data/real/measurements/mock-cal-001-xrd.csv \
  --optical data/real/measurements/mock-cal-001-optical.csv \
  --metadata data/real/measurements/mock-cal-001-metadata.yaml \
  --xrd-reference data/real/references/anatase-rutile-profile.csv \
  --repeat-composition data/real/measurements/mock-cal-001-composition-repeat.csv \
  --repeat-xrd data/real/measurements/mock-cal-001-xrd-repeat.csv \
  --repeat-optical data/real/measurements/mock-cal-001-optical-repeat.csv \
  --model data/real/demo-calibration/real_model.json \
  --out results/matrix-real-demo-full
```

This runs all relevant checks:

- C1 composition check
- C2 XRD displacement fit
- C3 relative phase fraction estimation
- C4 optical Tauc and alternate-baseline analysis
- C5 independent-repeat comparison

## 6. What the demo data includes

The generated demo YAML files follow the project schema for real-data manifests and include:

- `sample_id`
- `material`
- `ground_truth`
- `ground_truth_confidence`
- measurement CSV paths
- repeat measurement paths
- provenance metadata and measurement IDs
- `xrd_reference` path

The actual demo files are intentionally synthetic and are designed to satisfy the schema and support local demonstration runs.

## 7. Data folders and files

### 7.1 Calibration manifest

- `data/real/calibration.yaml`

Contains four labeled samples covering H1, H2, H3, and H4.

### 7.2 Held-out manifest

- `data/real/held-out.yaml`

Contains a separate set of sample records intended for validation.

### 7.3 Measurements

- `data/real/measurements/`

This folder contains all raw CSV and YAML files used by demo validation and diagnosis.

### 7.4 References

- `data/real/references/anatase-rutile-profile.csv`

This is the reference file used by the XRD fitting logic for C2/C3.

### 7.5 Demo model output

- `data/real/demo-calibration/real_model.json`

This is the calibration artifact produced by `real-calibrate`.

## 8. Understanding the CLI outputs

### 8.1 Validation output

```bash
python -m matrix real-dataset-validate --dataset data/real/calibration.yaml
```

This checks that the manifest is structurally valid and includes all required measurement and provenance fields.

Expected result:

- `valid: true`
- no `errors`
- a valid dataset hash and sample count

### 8.2 Diagnosis output

```bash
python -m matrix diagnose ...
```

The CLI writes a report directory containing:

- `diagnosis.json`
- `evidence.json`
- `report.md`

The report shows:

- the sample ID
- the material
- the status of each check
- the diagnosis result if a calibrated model is supplied
- the provenance and evidence used

### 8.3 Check status values

The project uses the following behavior:

- `AVAILABLE`: the check ran successfully
- `UNAVAILABLE`: the check could not run because required input was missing or invalid

For example, C2 and C3 require a valid XRD reference profile; C5 requires repeat measurements.

If a check is unavailable, it is not a crash. It is an expected part of the evidence-first workflow.

## 9. Running validation without a full demo

You can validate a dataset on its own:

```bash
python -m matrix real-dataset-validate --dataset data/real/held-out.yaml
```

You can generate or regenerate a calibration model:

```bash
python -m matrix real-calibrate \
  --dataset data/real/calibration.yaml \
  --out data/real/demo-calibration \
  --decision-threshold 0.7
```

You can also validate a held-out dataset against the trained model:

```bash
python -m matrix real-validate \
  --model data/real/demo-calibration/real_model.json \
  --dataset data/real/held-out.yaml \
  --out results/held-out-validation
```

## 10. Notes on data quality and scope

This repository includes a demo dataset that satisfies the schema and runs locally, but it is not scientific research data.

Important:

- the data under `data/real/` is synthetic and intended for demonstration
- it is not a real laboratory calibration dataset
- it must not be used as evidence for scientific claims
- it exists to exercise the validation, calibration, and diagnosis pipeline

## 11. Troubleshooting

### 11.1 `ModuleNotFoundError`

Install the requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

### 11.2 Dataset validation fails

Check that the file exists and the YAML is valid:

```bash
python -m matrix real-dataset-validate --dataset data/real/calibration.yaml
```

Common issues:

- missing measurement file
- missing metadata fields
- duplicate measurement IDs
- malformed CSV columns
- invalid YAML structure

### 11.3 Some checks show `UNAVAILABLE`

This is expected when required files are missing.

Typical causes:

- no `xrd_reference` supplied for C2/C3
- no repeat measurements supplied for C5
- missing provenance metadata or path fields

### 11.4 Demo script fails

Check that the repo is at the project root and that the virtual environment is active:

```bash
source .venv/bin/activate
./run_real_demo.sh
```

## 12. Recommended demo run order

For a simple presentation or internal demo, use this sequence:

```bash
source .venv/bin/activate
python -m matrix real-dataset-validate --dataset data/real/calibration.yaml
python -m matrix real-dataset-validate --dataset data/real/held-out.yaml
python -m matrix real-calibrate --dataset data/real/calibration.yaml --out data/real/demo-calibration --decision-threshold 0.7
python -m matrix diagnose \
  --composition data/real/measurements/mock-cal-001-composition.csv \
  --xrd data/real/measurements/mock-cal-001-xrd.csv \
  --optical data/real/measurements/mock-cal-001-optical.csv \
  --metadata data/real/measurements/mock-cal-001-metadata.yaml \
  --xrd-reference data/real/references/anatase-rutile-profile.csv \
  --repeat-composition data/real/measurements/mock-cal-001-composition-repeat.csv \
  --repeat-xrd data/real/measurements/mock-cal-001-xrd-repeat.csv \
  --repeat-optical data/real/measurements/mock-cal-001-optical-repeat.csv \
  --model data/real/demo-calibration/real_model.json \
  --out results/matrix-real-demo-full
```

## 13. Example result

The demo run should produce a diagnosis such as:

```text
matriX TiO2 evidence report
Sample: mock-cal-001
Diagnosis: H1 (real-calibrated model)
C1: AVAILABLE
C2: AVAILABLE
C3: AVAILABLE
C4: AVAILABLE
C5: AVAILABLE
```

This is exactly the expected behavior for the synthetic demo fixture.

## 14. Summary

This repository is ready for a local demo run. The included script and generated data allow you to validate the schema, calibrate a model, and run a working diagnosis end to end.

If you want a quick live demo, run:

```bash
./run_real_demo.sh
```

This will produce a complete demonstration using the synthetic TiO2 samples under `data/real/`.
