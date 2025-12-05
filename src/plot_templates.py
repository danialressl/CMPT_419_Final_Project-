"""
Templates for plotting influence scores and subset experiment results.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_influence_histogram(ranking_csv: str, output_path: str = "influence_hist.png") -> None:
    df = pd.read_csv(ranking_csv)
    plt.figure(figsize=(7, 4))
    plt.hist(df["influence_score"], bins=50, color="#1f77b4", alpha=0.8)
    plt.xlabel("Influence score")
    plt.ylabel("Count")
    plt.title("Influence score distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved histogram to {output_path}")


def plot_performance_curve(
    results_csv: str,
    output_path: str = "performance_curve.png",
    metric_col: str = "macro_auroc",
    label: str = "Macro AUROC",
    baseline_value: Optional[float] = None,
    baseline_label: str = "Baseline",
) -> None:
    """
    Expects a CSV with columns: percent, macro_auroc, macro_f1 (optional), mode (optional).
    If a 'mode' column is present, a separate plot is produced for each value
    (e.g., select vs. remove). A horizontal baseline is drawn when baseline_value is given.
    """
    df = pd.read_csv(results_csv)
    grouped = [("overall", df)] if "mode" not in df.columns else df.groupby("mode")

    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"

    for mode_name, subset in grouped:
        subset_sorted = subset.sort_values("percent")
        plt.figure(figsize=(7, 4))
        plt.plot(subset_sorted["percent"], subset_sorted[metric_col], marker="o", label=label)
        if "macro_f1" in subset_sorted.columns and metric_col != "macro_f1":
            plt.plot(
                subset_sorted["percent"],
                subset_sorted["macro_f1"],
                marker="s",
                label="Macro F1",
                linestyle="--",
            )
        if baseline_value is not None:
            plt.axhline(
                y=baseline_value,
                color="#444444",
                linestyle="--",
                linewidth=1.2,
                label=f"{baseline_label} ({baseline_value:.3f})",
            )
        plt.xlabel("Percent of training data kept")
        plt.ylabel("Score")
        title_mode = "" if mode_name == "overall" else f" ({mode_name})"
        plt.title(f"Subset performance{title_mode}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out_path = output_path if mode_name == "overall" else f"{root}_{mode_name}{ext}"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved performance plot for '{mode_name}' to {out_path}")


if __name__ == "__main__":
    # Example usage (fill in your CSV paths):
    # plot_influence_histogram("influence_rankings.csv")
    # plot_performance_curve("subset_results.csv", baseline_value=0.78)
    pass
