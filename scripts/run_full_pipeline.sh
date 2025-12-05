#!/usr/bin/env bash
# Run baseline, influence, subset experiments, and plots end-to-end.
# Usage: (activate your venv) then: bash run_full_pipeline.sh

set -euo pipefail

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-data}
RESULTS_CSV=${RESULTS_CSV:-subset_results.csv}

BASELINE_EPOCHS=${BASELINE_EPOCHS:-5}
BASELINE_BS=${BASELINE_BS:-32}
BASELINE_LR=${BASELINE_LR:-1e-4}

INFL_BATCH=${INFL_BATCH:-8}
INFL_WORKERS=${INFL_WORKERS:-2}
INFL_EVAL_LIMIT=${INFL_EVAL_LIMIT:-200}

SUBSET_EPOCHS=${SUBSET_EPOCHS:-5}
SUBSET_BS=${SUBSET_BS:-32}
SUBSET_LR=${SUBSET_LR:-1e-4}
REMOVE_PERCENTS=${REMOVE_PERCENTS:-"5 15 35 60"}
SELECT_PERCENTS=${SELECT_PERCENTS:-"2 5 10 20 40 80"}

echo "Using python: $($PYTHON -V 2>&1)"
echo "Data dir: $DATA_DIR"
echo "Results CSV: $RESULTS_CSV (existing file will be appended)"

echo "=== 1) Baseline training ==="
$PYTHON baseline_nih_chestxray.py \
  --data_dir "$DATA_DIR" \
  --epochs "$BASELINE_EPOCHS" \
  --batch_size "$BASELINE_BS" \
  --lr "$BASELINE_LR"

echo "=== 2) Influence computation (TracIn-style) ==="
$PYTHON tracin_influence.py \
  --data_dir "$DATA_DIR" \
  --checkpoints best_model.pth \
  --batch_size "$INFL_BATCH" \
  --num_workers "$INFL_WORKERS" \
  --eval_limit "$INFL_EVAL_LIMIT" \
  --output_csv influence_rankings.csv

echo "=== 3A) Subset experiments: remove top-k% ==="
for pct in $REMOVE_PERCENTS; do
  echo "-- remove top ${pct}% --"
  $PYTHON subset_experiments.py \
    --data_dir "$DATA_DIR" \
    --ranking_csv influence_rankings.csv \
    --mode remove \
    --percent "$pct" \
    --run_train \
    --epochs "$SUBSET_EPOCHS" \
    --batch_size "$SUBSET_BS" \
    --lr "$SUBSET_LR" \
    --results_csv "$RESULTS_CSV"
done

echo "=== 3B) Subset experiments: keep top-k% ==="
for pct in $SELECT_PERCENTS; do
  echo "-- keep top ${pct}% --"
  $PYTHON subset_experiments.py \
    --data_dir "$DATA_DIR" \
    --ranking_csv influence_rankings.csv \
    --mode select \
    --percent "$pct" \
    --run_train \
    --epochs "$SUBSET_EPOCHS" \
    --batch_size "$SUBSET_BS" \
    --lr "$SUBSET_LR" \
    --results_csv "$RESULTS_CSV"
done

echo "=== 4) Plot figures ==="
$PYTHON - <<'PY'
import plot_templates as p

print("Plotting influence histogram -> influence_hist.png")
p.plot_influence_histogram("influence_rankings.csv")

print("Plotting performance curve (AUROC) -> performance_curve.png")
p.plot_performance_curve("subset_results.csv", metric_col="macro_auroc", label="Macro AUROC")

print("Plotting performance curve (Macro F1) -> performance_curve_f1.png")
p.plot_performance_curve("subset_results.csv", metric_col="macro_f1", label="Macro F1", output_path="performance_curve_f1.png")
PY

echo "Pipeline complete."
