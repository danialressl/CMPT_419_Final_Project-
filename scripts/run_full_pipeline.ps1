# Run baseline, influence, subset experiments, and plots end-to-end.
# Usage: (activate your venv) then: powershell -ExecutionPolicy Bypass -File .\run_full_pipeline.ps1

$PYTHON = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$DATA_DIR = if ($env:DATA_DIR) { $env:DATA_DIR } else { "data" }
$RESULTS_CSV = if ($env:RESULTS_CSV) { $env:RESULTS_CSV } else { "subset_results.csv" }

$BASELINE_EPOCHS = if ($env:BASELINE_EPOCHS) { $env:BASELINE_EPOCHS } else { "5" }
$BASELINE_BS = if ($env:BASELINE_BS) { $env:BASELINE_BS } else { "32" }
$BASELINE_LR = if ($env:BASELINE_LR) { $env:BASELINE_LR } else { "1e-4" }

$INFL_BATCH = if ($env:INFL_BATCH) { $env:INFL_BATCH } else { "8" }
$INFL_WORKERS = if ($env:INFL_WORKERS) { $env:INFL_WORKERS } else { "2" }
$INFL_EVAL_LIMIT = if ($env:INFL_EVAL_LIMIT) { $env:INFL_EVAL_LIMIT } else { "200" }

$SUBSET_EPOCHS = if ($env:SUBSET_EPOCHS) { $env:SUBSET_EPOCHS } else { "5" }
$SUBSET_BS = if ($env:SUBSET_BS) { $env:SUBSET_BS } else { "32" }
$SUBSET_LR = if ($env:SUBSET_LR) { $env:SUBSET_LR } else { "1e-4" }
$REMOVE_PERCENTS = if ($env:REMOVE_PERCENTS) { $env:REMOVE_PERCENTS.Split(" ") } else { @("5","15","35","60") }
$SELECT_PERCENTS = if ($env:SELECT_PERCENTS) { $env:SELECT_PERCENTS.Split(" ") } else { @("2","5","10","20","40","80") }

Write-Host "Using python: $(& $PYTHON -V)"
Write-Host "Data dir: $DATA_DIR"
Write-Host "Results CSV: $RESULTS_CSV (existing file will be appended)"

Write-Host "=== 1) Baseline training ==="
& $PYTHON baseline_nih_chestxray.py --data_dir $DATA_DIR --epochs $BASELINE_EPOCHS --batch_size $BASELINE_BS --lr $BASELINE_LR

Write-Host "=== 2) Influence computation (TracIn-style) ==="
& $PYTHON tracin_influence.py --data_dir $DATA_DIR --checkpoints best_model.pth --batch_size $INFL_BATCH --num_workers $INFL_WORKERS --eval_limit $INFL_EVAL_LIMIT --output_csv influence_rankings.csv

Write-Host "=== 3A) Subset experiments: remove top-k% ==="
foreach ($pct in $REMOVE_PERCENTS) {
  Write-Host "-- remove top $pct% --"
  & $PYTHON subset_experiments.py --data_dir $DATA_DIR --ranking_csv influence_rankings.csv --mode remove --percent $pct --run_train --epochs $SUBSET_EPOCHS --batch_size $SUBSET_BS --lr $SUBSET_LR --results_csv $RESULTS_CSV
}

Write-Host "=== 3B) Subset experiments: keep top-k% ==="
foreach ($pct in $SELECT_PERCENTS) {
  Write-Host "-- keep top $pct% --"
  & $PYTHON subset_experiments.py --data_dir $DATA_DIR --ranking_csv influence_rankings.csv --mode select --percent $pct --run_train --epochs $SUBSET_EPOCHS --batch_size $SUBSET_BS --lr $SUBSET_LR --results_csv $RESULTS_CSV
}

Write-Host "=== 4) Plot figures ==="
$pyCode = @'
import plot_templates as p

print("Plotting influence histogram -> influence_hist.png")
p.plot_influence_histogram("influence_rankings.csv")

print("Plotting performance curve (AUROC) -> performance_curve.png")
p.plot_performance_curve("subset_results.csv", metric_col="macro_auroc", label="Macro AUROC")

print("Plotting performance curve (Macro F1) -> performance_curve_f1.png")
p.plot_performance_curve("subset_results.csv", metric_col="macro_f1", label="Macro F1", output_path="performance_curve_f1.png")
'@
& $PYTHON -c "$pyCode"

Write-Host "Pipeline complete."
