#!/usr/bin/env python3
"""Fine-tune a pretrained dual-head baseline on a source-only 7:2:1 split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_TO_LABEL = {
    "dvd_no": (0, 0),
    "eso_no": (1, 0),
    "eso_A": (1, 1),
    "eso_V": (1, 2),
    "exo_no": (2, 0),
    "exo_A": (2, 1),
    "exo_V": (2, 2),
}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_REFERENCE = Path("model_sources")
INCEPTION_PRETRAINED = Path("/opt/data/CI_GNN/codex-home/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-only stratified 7:2:1 training/validation/test."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("Source_9gaze_composed"))
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--model", choices=("resnet50", "inception_v3", "vit", "densenet121", "vgg16", "swin_b"), default="resnet50")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet50_source_721_seed42"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-factor", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/model only.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scan_dataset(root: Path) -> dict[str, list[Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    result: dict[str, list[Path]] = {}
    unknown = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in CLASS_TO_LABEL)
    if unknown:
        raise ValueError(f"Unknown class directories under {root}: {unknown}")
    for class_name in sorted(CLASS_TO_LABEL):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if paths:
            result[class_name] = paths
    if not result:
        raise ValueError(f"No supported images found under {root}")
    return result


def stratified_source_split(
    grouped: dict[str, list[Path]], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]], list[tuple[Path, str]]]:
    if val_ratio <= 0.0 or test_ratio <= 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio and --test-ratio must be positive and sum to less than 1.")
    rng = random.Random(seed)
    train: list[tuple[Path, str]] = []
    val: list[tuple[Path, str]] = []
    test: list[tuple[Path, str]] = []
    for class_name, original_paths in sorted(grouped.items()):
        paths = list(original_paths)
        rng.shuffle(paths)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        n_test = max(1, int(round(len(paths) * test_ratio)))
        if n_val + n_test >= len(paths):
            raise ValueError(f"Class {class_name} needs more source images for train/val/test.")
        val.extend((path, class_name) for path in paths[:n_val])
        test.extend((path, class_name) for path in paths[n_val:n_val + n_test])
        train.extend((path, class_name) for path in paths[n_val + n_test:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def flatten_grouped(grouped: dict[str, list[Path]]) -> list[tuple[Path, str]]:
    return [
        (path, class_name)
        for class_name, paths in sorted(grouped.items())
        for path in paths
    ]


class EyeDataset(Dataset):
    def __init__(
        self, samples: list[tuple[Path, str]], size: tuple[int, int], augment: bool
    ) -> None:
        self.samples = samples
        self.height, self.width = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, class_name = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.width, self.height), Image.Resampling.BILINEAR
            )
            if self.augment:
                image = self._augment(image)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        tensor = (tensor - 0.5) / 0.5
        return tensor, torch.tensor(CLASS_TO_LABEL[class_name], dtype=torch.long)

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
        return image


class InceptionTwoHead(nn.Module):
    def __init__(self, backbone: nn.Module, features: int = 2048) -> None:
        super().__init__()
        self.backbone = backbone
        self.head1 = nn.Linear(features, 3)
        self.head2 = nn.Linear(features, 3)

    def forward(self, inputs: torch.Tensor):
        features = self.backbone(inputs)
        return self.head1(features), self.head2(features)


def load_model(model_name: str, reference_dir: Path, pretrained: Path) -> nn.Module:
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")
    if model_name == "inception_v3":
        from torchvision.models import inception_v3
        backbone = inception_v3(weights=None, aux_logits=False)
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("AuxLogits.") and not k.startswith("fc.")}
        result = backbone.load_state_dict(state, strict=False)
        if set(result.missing_keys) != {"fc.weight", "fc.bias"} or result.unexpected_keys:
            raise RuntimeError(f"InceptionV3 pretrained mismatch: {result}")
        backbone.fc = nn.Identity()
        return InceptionTwoHead(backbone)
    if model_name in {"densenet121", "vgg16", "swin_b"}:
        from torchvision.models import densenet121, swin_b, vgg16
        specs = {
            "densenet121": (densenet121, "classifier", 1024),
            "vgg16": (vgg16, "classifier.6", 4096),
            "swin_b": (swin_b, "head", 1024),
        }
        constructor, head_name, features = specs[model_name]
        backbone = constructor(weights=None)
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        if model_name == "densenet121":
            pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
            for key in list(state):
                match = pattern.match(key)
                if match:
                    state[match.group(1) + match.group(2)] = state.pop(key)
        prefixes = (head_name + ".weight", head_name + ".bias")
        state = {k: v for k, v in state.items() if k not in prefixes}
        result = backbone.load_state_dict(state, strict=False)
        if set(result.missing_keys) != set(prefixes) or result.unexpected_keys:
            raise RuntimeError(f"{model_name} pretrained mismatch: {result}")
        if model_name == "densenet121":
            backbone.classifier = nn.Identity()
        elif model_name == "vgg16":
            backbone.classifier[6] = nn.Identity()
        else:
            backbone.head = nn.Identity()
        return InceptionTwoHead(backbone, features)
    if model_name == "vit":
        sys.path.insert(0, str(reference_dir))
        try:
            from vit_model import vit_base_patch16_224_in21k
        finally:
            sys.path.pop(0)
        model = vit_base_patch16_224_in21k(num_classes=3, has_logits=False)
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        for key in ("head.weight", "head.bias", "pre_logits.fc.weight", "pre_logits.fc.bias"):
            state.pop(key, None)
        old = state.get("pos_embed")
        if old is not None and old.shape != model.pos_embed.shape:
            cls_token, patches = old[:, :1], old[:, 1:]
            side = int(math.sqrt(patches.shape[1]))
            patches = patches.reshape(1, side, side, patches.shape[2]).permute(0, 3, 1, 2)
            patches = torch.nn.functional.interpolate(patches, size=(12, 36), mode="bilinear", align_corners=False)
            state["pos_embed"] = torch.cat((cls_token, patches.permute(0, 2, 3, 1).reshape(1, 432, -1)), dim=1)
        result = model.load_state_dict(state, strict=False)
        expected = {"head1.weight", "head1.bias", "head2.weight", "head2.bias"}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError(f"ViT pretrained mismatch: {result}")
        return model
    model_root = reference_dir / "model"
    sys.path.insert(0, str(model_root))
    try:
        from Resnet.model import resnet50
    finally:
        sys.path.pop(0)
    model = resnet50(num_classes=3, include_top=True)
    state = torch.load(pretrained, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def combined_class(labels: torch.Tensor) -> list[str]:
    reverse = {value: key for key, value in CLASS_TO_LABEL.items()}
    return [reverse.get(tuple(row.tolist()), f"{row[0].item()}_{row[1].item()}") for row in labels]


def joint_class_predictions(output1: torch.Tensor, output2: torch.Tensor) -> list[str]:
    classes = sorted(name for name in CLASS_TO_LABEL if name != "eso_A")
    pairs = torch.tensor([CLASS_TO_LABEL[name] for name in classes], device=output1.device)
    scores = torch.log_softmax(output1, 1)[:, pairs[:, 0]] + torch.log_softmax(output2, 1)[:, pairs[:, 1]]
    return [classes[i] for i in scores.argmax(1).cpu().tolist()]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    loss_sum = 0.0
    labels_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    joint_predictions: list[str] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output1, output2 = model(images)
            loss = loss_fn(output1, labels[:, 0]) + loss_fn(output2, labels[:, 1])
            if training:
                loss.backward()
                optimizer.step()
        predictions = torch.stack((output1.argmax(1), output2.argmax(1)), dim=1)
        joint_predictions.extend(joint_class_predictions(output1, output2))
        loss_sum += loss.item() * images.size(0)
        labels_all.append(labels.cpu())
        predictions_all.append(predictions.cpu())

    labels_tensor = torch.cat(labels_all)
    predictions_tensor = torch.cat(predictions_all)
    exact = (labels_tensor == predictions_tensor).all(dim=1).float().mean().item()
    joint_labels = combined_class(labels_tensor)
    valid_classes = sorted(name for name in CLASS_TO_LABEL if name != "eso_A")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        joint_labels, joint_predictions, labels=valid_classes, average="weighted", zero_division=0
    )
    return {
        "loss": loss_sum / len(loader.dataset),
        "exact_accuracy": exact,
        "head1_accuracy": (labels_tensor[:, 0] == predictions_tensor[:, 0]).float().mean().item(),
        "head2_accuracy": (labels_tensor[:, 1] == predictions_tensor[:, 1]).float().mean().item(),
        "joint_accuracy": accuracy_score(joint_labels, joint_predictions),
        "joint_weighted_precision": float(weighted_p),
        "joint_weighted_recall": float(weighted_r),
        "joint_weighted_f1": float(weighted_f1),
        "labels": labels_tensor,
        "predictions": predictions_tensor,
        "joint_predictions": joint_predictions,
    }


def save_split_manifest(
    output_dir: Path,
    train: list[tuple[Path, str]],
    val: list[tuple[Path, str]],
    test: list[tuple[Path, str]],
) -> None:
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "class", "head1", "head2", "path"))
        for split, samples in (("train", train), ("val", val), ("test", test)):
            for path, class_name in samples:
                writer.writerow((split, class_name, *CLASS_TO_LABEL[class_name], str(path.resolve())))


def public_metrics(metrics: dict) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if isinstance(value, float)}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    source = scan_dataset(args.source_dir)
    train_samples, val_samples, test_samples = stratified_source_split(source, args.val_ratio, args.test_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(args.output_dir, train_samples, val_samples, test_samples)
    counts = {
        "source_train": dict(sorted(_counts(train_samples).items())),
        "source_val": dict(sorted(_counts(val_samples).items())),
        "test": dict(sorted(_counts(test_samples).items())),
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2, ensure_ascii=False))

    if args.pretrained is not None:
        pretrained = args.pretrained
    elif args.model == "vit":
        pretrained = args.reference_dir / "vit_base_patch16_224_in21k.pth"
    elif args.model == "inception_v3":
        pretrained = INCEPTION_PRETRAINED
    else:
        pretrained = args.reference_dir / "weights" / "ResNet.pkl"
    model = load_model(args.model, args.reference_dir, pretrained)
    if args.dry_run:
        print(f"Dry run successful; loaded pretrained weights: {pretrained}")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {args.device}, but CUDA is unavailable. Use --device cpu or a CUDA-compatible environment."
        )
    device = torch.device(args.device)
    model.to(device)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        EyeDataset(train_samples, (args.height, args.width), not args.no_augment),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        EyeDataset(val_samples, (args.height, args.width), False),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        EyeDataset(test_samples, (args.height, args.width), False),
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2)
        * (1 - args.min_lr_factor)
        + args.min_lr_factor,
    )
    history: list[dict] = []
    best_accuracy = -1.0
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in public_metrics(train_metrics).items()},
            **{f"val_{k}": v for k, v in public_metrics(val_metrics).items()},
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["exact_accuracy"] > best_accuracy:
            best_accuracy = val_metrics["exact_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics": public_metrics(val_metrics),
                    "class_to_label": CLASS_TO_LABEL,
                    "args": vars(args),
                },
                best_path,
            )
        scheduler.step()

    # This checkpoint is produced locally above and includes argparse Path
    # values.  PyTorch 2.6+ defaults to weights_only=True, which rejects Path.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, device)
    label_names = sorted(name for name in CLASS_TO_LABEL if name != "eso_A")
    y_true = combined_class(test_metrics["labels"])
    y_pred = test_metrics["joint_predictions"]
    results = {
        "best_source_val_epoch": checkpoint["epoch"],
        "best_source_val": checkpoint["val_metrics"],
        "source_test": public_metrics(test_metrics),
        "test_confusion_matrix_labels": label_names,
        "test_confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_names).tolist(),
        "test_classification_report": classification_report(
            y_true, y_pred, labels=label_names, output_dict=True, zero_division=0
        ),
        "test_six_class_accuracy": accuracy_score(y_true, y_pred),
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "test_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


def _counts(samples: list[tuple[Path, str]]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for _, class_name in samples:
        counts[class_name] += 1
    return dict(counts)


if __name__ == "__main__":
    main()
