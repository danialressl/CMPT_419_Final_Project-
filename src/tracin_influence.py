import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_nih_chestxray import (
    NIHChestXray,
    build_model,
    build_dataloaders,
    DEVICE,
)


def flatten_grads(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return torch.tensor([], device=DEVICE)
    return torch.cat(grads)


def compute_batch_grad(
    model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, criterion: nn.Module
) -> torch.Tensor:
    outputs = model(images)
    loss = criterion(outputs, targets)
    loss.backward()
    grads = flatten_grads(model)
    model.zero_grad(set_to_none=True)
    return grads


def aggregate_eval_grad(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    eval_limit: Optional[int],
) -> torch.Tensor:
    model.eval()
    grad_sum = None
    batches = 0
    with torch.enable_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Eval gradients", leave=False)):
            if len(batch) == 3:
                images, targets, _ = batch
            else:
                images, targets = batch
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            grad = compute_batch_grad(model, images, targets, criterion)
            grad_sum = grad if grad_sum is None else grad_sum + grad
            batches += 1
            if eval_limit is not None and batches >= eval_limit:
                break
    if grad_sum is None:
        raise RuntimeError("No gradients computed for evaluation set")
    return grad_sum / batches


def compute_influence_scores(
    checkpoints: List[str],
    train_loader: DataLoader,
    eval_loader: DataLoader,
    criterion: nn.Module,
    eval_limit: Optional[int],
) -> Dict[str, float]:
    influence: Dict[str, float] = {}
    model = build_model()

    for ckpt in checkpoints:
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.zero_grad(set_to_none=True)

        eval_grad = aggregate_eval_grad(model, eval_loader, criterion, eval_limit=eval_limit)

        model.train()
        for batch in tqdm(train_loader, desc=f"Train influence {os.path.basename(ckpt)}", leave=False):
            if len(batch) == 3:
                images, targets, filenames = batch
            else:
                images, targets = batch
                filenames = [f"idx_{i}" for i in range(len(targets))]
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            grad_train = compute_batch_grad(model, images, targets, criterion)
            score = torch.dot(grad_train, eval_grad).item()
            per_sample = score / max(len(filenames), 1)
            for fname in filenames:
                influence[fname] = influence.get(fname, 0.0) + per_sample

    return influence


def save_rankings(influence: Dict[str, float], output_csv: str) -> None:
    df = pd.DataFrame(
        {"filename": list(influence.keys()), "influence_score": list(influence.values())}
    )
    df = df.sort_values("influence_score", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"Saved influence rankings to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TracIn-style influence scores")
    parser.add_argument("--data_dir", type=str, default="data", help="Dataset root")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="List of checkpoint paths to use for influence computation (default: best_model.pth if present)",
    )
    parser.add_argument(
        "--train_list", type=str, default=None, help="Optional custom train list (defaults to train_val_list.txt)"
    )
    parser.add_argument(
        "--eval_list", type=str, default=None, help="Optional eval/test list (defaults to test_list.txt)"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for influence computation")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_csv", type=str, default="influence_rankings.csv", help="Output CSV file")
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=None,
        help="Optional limit on eval batches for faster approximation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoints is None:
        default_ckpt = "best_model.pth"
        if os.path.exists(default_ckpt):
            args.checkpoints = [default_ckpt]
            print(f"--checkpoints not provided; using {default_ckpt}")
        else:
            raise SystemExit("Error: --checkpoints is required and no best_model.pth found.")

    train_loader, eval_loader = build_dataloaders(
        args.data_dir,
        args.batch_size,
        num_workers=args.num_workers,
        train_list_path=args.train_list,
        test_list_path=args.eval_list,
        return_filenames=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    influence = compute_influence_scores(
        args.checkpoints,
        train_loader,
        eval_loader,
        criterion,
        eval_limit=args.eval_limit,
    )
    save_rankings(influence, args.output_csv)


if __name__ == "__main__":
    main()
