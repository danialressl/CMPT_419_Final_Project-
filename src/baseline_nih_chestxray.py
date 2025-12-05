import argparse
import os
import random
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
LABEL_SMOOTHING = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE_CHOICES = ["densenet121", "resnet50"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(preference: str) -> str:
    """Set global DEVICE based on user preference."""
    global DEVICE
    if preference == "cuda":
        if torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            DEVICE = "cpu"
    elif preference == "cpu":
        DEVICE = "cpu"
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE


class NIHChestXray(Dataset):
    """Dataset for NIH ChestXray14 with multi-label targets."""

    def __init__(
        self,
        img_dir: str,
        csv_path: str,
        file_list_path: str,
        transform: transforms.Compose,
        return_filename: bool = False,
    ) -> None:
        self.img_dir = img_dir
        self.transform = transform
        self.return_filename = return_filename

        df = pd.read_csv(csv_path)
        if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
            raise ValueError("CSV is missing required columns: Image Index, Finding Labels")
        df = df.set_index("Image Index")

        with open(file_list_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]

        self.samples: List[Tuple[str, np.ndarray]] = []
        for fname in filenames:
            if fname not in df.index:
                raise ValueError(f"{fname} not found in {csv_path}")
            labels_str = df.loc[fname, "Finding Labels"]
            label_vec = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            if isinstance(labels_str, str):
                for label in labels_str.split("|"):
                    if label in CLASS_MAP:
                        label_vec[CLASS_MAP[label]] = 1.0
            self.samples.append((fname, label_vec))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, label_vec = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img = self.transform(img)
        labels = torch.from_numpy(label_vec)
        if self.return_filename:
            return img, labels, fname
        return img, labels


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, test_transform


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    train_list_path: Optional[str] = None,
    test_list_path: Optional[str] = None,
    return_filenames: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    img_dir = os.path.join(data_dir, "images")
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    train_list = train_list_path or os.path.join(data_dir, "train_val_list.txt")
    test_list = test_list_path or os.path.join(data_dir, "test_list.txt")

    train_transform, test_transform = get_transforms()

    train_dataset = NIHChestXray(
        img_dir, csv_path, train_list, train_transform, return_filename=return_filenames
    )
    test_dataset = NIHChestXray(
        img_dir, csv_path, test_list, test_transform, return_filename=return_filenames
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def build_model() -> nn.Module:
    return build_model_with_backbone("densenet121")


def build_model_with_backbone(backbone: str = "densenet121") -> nn.Module:
    backbone = backbone.lower()
    if backbone == "densenet121":
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
            model = models.densenet121(weights=weights)
        except AttributeError:
            model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(CLASS_NAMES))
    elif backbone == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from {BACKBONE_CHOICES}.")
    return model.to(DEVICE)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    epoch_idx: int,
) -> float:
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc=f"Train {epoch_idx + 1}", leave=False)
    for batch in progress:
        if len(batch) == 3:
            images, targets, _ = batch
        else:
            images, targets = batch
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        smoothed_targets = (1.0 - LABEL_SMOOTHING) * targets + LABEL_SMOOTHING * 0.5
        loss = criterion(outputs, smoothed_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        modulating = (1 - pt).pow(self.gamma)
        loss = modulating * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= start) & (y_prob < end if i < n_bins - 1 else y_prob <= end)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        weight = mask.mean()
        ece += np.abs(conf - acc) * weight
    return float(ece)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float, List[float], float]:
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            if len(batch) == 3:
                images, targets, _ = batch
            else:
                images, targets = batch
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()

    per_class_auroc = []
    for i in range(len(CLASS_NAMES)):
        try:
            score = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            score = np.nan
        per_class_auroc.append(score)

    macro_auroc = float(np.nanmean(per_class_auroc))

    # per-class threshold tuning
    best_thresholds = []
    f1_scores = []

    threshold_grid = torch.linspace(0.05, 0.95, steps=19)

    for i in range(NUM_CLASSES):
        best_f1 = 0
        best_t = 0.5
        for t in threshold_grid:
            t_val = float(t.item())
            preds = (y_prob[:, i] >= t_val).astype(int)
            try:
                f1 = f1_score(y_true[:, i], preds, zero_division=0)
            except Exception:
                f1 = 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t_val
        best_thresholds.append(best_t)
        f1_scores.append(best_f1)

    macro_f1 = float(np.mean(f1_scores))

    ece = expected_calibration_error(y_true.flatten(), y_prob.flatten())
    return macro_auroc, macro_f1, per_class_auroc, ece


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline NIH ChestXray14 classifier")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_list", type=str, default=None, help="Optional custom train list file")
    parser.add_argument("--test_list", type=str, default=None, help="Optional custom test list file")
    parser.add_argument("--val_list", type=str, default=None, help="Optional validation list for threshold tuning")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=BACKBONE_CHOICES,
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (auto selects CUDA if available)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_device(args.device)
    set_seed(42)

    print(f"Using device: {DEVICE}")
    print(
        "CUDA available: {} | torch CUDA build: {} | GPU count: {}".format(
            torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count()
        )
    )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_loader, test_loader = build_dataloaders(
        args.data_dir,
        args.batch_size,
        num_workers=args.num_workers,
        train_list_path=args.train_list,
        test_list_path=args.test_list,
    )
    model = build_model_with_backbone(args.backbone)

    # Optional validation loader for threshold tuning / checkpointing
    val_loader = None
    if args.val_list:
        _, test_transform = get_transforms()
        img_dir = os.path.join(args.data_dir, "images")
        csv_path = os.path.join(args.data_dir, "Data_Entry_2017.csv")
        val_dataset = NIHChestXray(img_dir, csv_path, args.val_list, test_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # compute pos_weight for class-balanced BCE loss
    train_set = train_loader.dataset
    label_mat = np.stack([lbl for _, lbl in train_set.samples])
    pos_counts = label_mat.sum(axis=0)
    neg_counts = len(train_set.samples) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)
    pos_weight_tensor = torch.clamp(pos_weight_tensor, min=2.0, max=8.0)

    if args.loss == "focal":
        criterion = FocalLossWithLogits(gamma=args.focal_gamma, pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    eval_loader = val_loader if val_loader is not None else test_loader

    best_macro_auroc = -float("inf")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

        macro_auroc, macro_f1, per_class_auroc, ece = evaluate(model, eval_loader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Macro AUROC: {macro_auroc:.4f} "
            f"- Macro F1: {macro_f1:.4f} "
            f"- ECE: {ece:.4f}"
        )

        if macro_auroc > best_macro_auroc:
            best_macro_auroc = macro_auroc
            torch.save(model.state_dict(), "best_model.pth")
            print("New best model saved with Macro AUROC {:.4f}".format(best_macro_auroc))

        scheduler.step()

    print(f"Best Macro AUROC achieved: {best_macro_auroc:.4f}")
    macro_auroc, macro_f1, per_class_auroc, ece = evaluate(model, test_loader)
    df = pd.DataFrame({"class": CLASS_NAMES, "auroc": per_class_auroc})
    df.to_csv("baseline_per_class_auroc.csv", index=False)
    print("Saved per-class AUROC to baseline_per_class_auroc.csv")


if __name__ == "__main__":
    main()
