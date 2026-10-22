#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="$ROOT/data/real/demo-calibration"
MODEL_PATH="$MODEL_DIR/real_model.json"

mkdir -p "$ROOT/results"

echo "[1/4] Validating real datasets..."
python -m matrix real-dataset-validate --dataset data/real/calibration.yaml
python -m matrix real-dataset-validate --dataset data/real/held-out.yaml

echo "[2/4] Calibrating real model..."
if [[ ! -f "$MODEL_PATH" ]]; then
  python -m matrix real-calibrate \
    --dataset data/real/calibration.yaml \
    --out "$MODEL_DIR" \
    --decision-threshold 0.7
fi

echo "[3/4] Running full diagnostic checks on demo samples..."
for sample in \
  mock-cal-001 \
  mock-cal-002 \
  mock-cal-003 \
  mock-cal-004; do
  OUT_DIR="$ROOT/results/${sample}-diagnosis"
  python -m matrix diagnose \
    --composition "$ROOT/data/real/measurements/${sample}-composition.csv" \
    --xrd "$ROOT/data/real/measurements/${sample}-xrd.csv" \
    --optical "$ROOT/data/real/measurements/${sample}-optical.csv" \
    --metadata "$ROOT/data/real/measurements/${sample}-metadata.yaml" \
    --xrd-reference "$ROOT/data/real/references/anatase-rutile-profile.csv" \
    --repeat-composition "$ROOT/data/real/measurements/${sample}-composition-repeat.csv" \
    --repeat-xrd "$ROOT/data/real/measurements/${sample}-xrd-repeat.csv" \
    --repeat-optical "$ROOT/data/real/measurements/${sample}-optical-repeat.csv" \
    --model "$MODEL_PATH" \
    --out "$OUT_DIR"
  echo "   -> $OUT_DIR"
done

echo "[4/4] Demo complete."
printf "\nModel: %s\n" "$MODEL_PATH"
printf "Reports written to: %s/results\n" "$ROOT"
