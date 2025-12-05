import argparse
import os
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import baseline_nih_chestxray as base


def build_subset_list(
    original_list: str,
    ranking_csv: str,
    mode: str,
    percent: float,
    output_path: str,
) -> Tuple[List[str], List[str]]:
    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in (0, 100]")
    with open(original_list, "r") as f:
        original_files = [l.strip() for l in f if l.strip()]

    df = pd.read_csv(ranking_csv)
    if "filename" not in df.columns or "influence_score" not in df.columns:
        raise ValueError("ranking_csv must contain columns: filename, influence_score")

    # Align to files present in the original list
    df = df[df["filename"].isin(original_files)]
    total = len(df)
    if total == 0:
        raise ValueError("No overlapping filenames between ranking and original list.")

    keep_count = max(1, int(total * percent / 100.0))

    if mode == "select":
        df_sorted = df.sort_values("influence_score", ascending=False)
        kept = df_sorted.head(keep_count)["filename"].tolist()
    elif mode == "remove":
        df_sorted = df.sort_values("influence_score", ascending=False)
        kept = df_sorted.tail(total - keep_count)["filename"].tolist()
    else:
        raise ValueError("mode must be 'select' or 'remove'")

    removed = [f for f in original_files if f not in kept]

    with open(output_path, "w") as f:
        for fname in kept:
            f.write(fname + "\n")

    print(f"Subset list written to {output_path} (kept {len(kept)} of {total})")
    return kept, removed


def train_on_subset(
    data_dir: str,
    train_list_path: str,
    test_list_path: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    output_model: str,
    percent: float,
    mode: str,
    results_csv: str,
    backbone: str,
    loss_type: str,
    focal_gamma: float,
) -> None:
    train_loader, test_loader = base.build_dataloaders(
        data_dir,
        batch_size,
        num_workers=num_workers,
        train_list_path=train_list_path,
        test_list_path=test_list_path,
    )

    model = base.build_model_with_backbone(backbone)
    train_set = train_loader.dataset
    label_mat = np.stack([lbl for _, lbl in train_set.samples])
    pos_counts = label_mat.sum(axis=0)
    neg_counts = len(train_set.samples) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(base.DEVICE)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=2.0, max=8.0)

    if loss_type == "focal":
        criterion = base.FocalLossWithLogits(gamma=focal_gamma, pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_macro_auroc = -float("inf")
    best_macro_f1 = float("-inf")
    best_ece = float("inf")
    for epoch in tqdm(range(epochs), desc="Subset train epochs"):
        train_loss = base.train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        macro_auroc, macro_f1, _, ece = base.evaluate(model, test_loader)
        print(
            f"[subset] Epoch {epoch + 1}/{epochs} "
            f"- Loss {train_loss:.4f} "
            f"- Macro AUROC {macro_auroc:.4f} "
            f"- Macro F1 {macro_f1:.4f} "
            f"- ECE {ece:.4f}"
        )
        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            best_macro_f1 = macro_f1
            best_ece = ece
            torch.save(model.state_dict(), output_model)
            print(f"Saved subset best model to {output_model} (macro AUROC {best_macro_auroc:.4f})")
        scheduler.step()

    row = {
        "mode": mode,
        "percent": percent,
        "macro_auroc": best_macro_auroc,
        "macro_f1": best_macro_f1,
        "ece": best_ece,
    }
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_csv, index=False)
    print(f"Appended subset results to {results_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset removal/selection experiments using influence scores")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root")
    parser.add_argument("--ranking_csv", type=str, required=True, help="Influence ranking CSV")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["select", "remove"],
        required=True,
        help="select = keep top k%%; remove = drop top k%% (keep the rest)",
    )
    parser.add_argument("--percent", type=float, required=True, help="Percentage for selection/removal")
    # Suggested sweep levels: remove -> 5, 15, 35, 60; select -> 2, 5, 10, 20, 40, 80
    parser.add_argument(
        "--original_train_list",
        type=str,
        default=None,
        help="Original train list file (defaults to train_val_list.txt)",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Optional custom test list (defaults to test_list.txt)",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        default="data/subset_list.txt",
        help="Path to write the generated subset list",
    )
    parser.add_argument("--run_train", action="store_true", help="If set, train on the subset after building it")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for subset training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for subset training")
    parser.add_argument("--lr", type=float, default=1e-4, help="LR for subset training")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=base.BACKBONE_CHOICES,
        help="Backbone architecture to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="focal",
        choices=["bce", "focal"],
        help="Loss type: standard BCE or focal loss",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for focal loss")
    parser.add_argument(
        "--output_model",
        type=str,
        default="subset_best_model.pth",
        help="Where to save the best subset-trained model",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="subset_results.csv",
        help="CSV file to store subset experiment results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_train_list = (
        args.original_train_list
        if args.original_train_list is not None
        else os.path.join(args.data_dir, "train_val_list.txt")
    )

    kept, removed = build_subset_list(
        original_train_list,
        args.ranking_csv,
        args.mode,
        args.percent,
        args.output_list,
    )

    print(f"Kept {len(kept)} samples; removed {len(removed)} samples.")

    if args.run_train:
        train_on_subset(
            args.data_dir,
            args.output_list,
            args.test_list,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.num_workers,
            args.output_model,
            args.percent,
            args.mode,
            args.results_csv,
            args.backbone,
            args.loss,
            args.focal_gamma,
        )


if __name__ == "__main__":
    main()
