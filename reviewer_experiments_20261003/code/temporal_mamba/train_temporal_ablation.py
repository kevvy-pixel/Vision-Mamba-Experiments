#!/usr/bin/env python3
"""Validate clinical-order Temporal Mamba using frozen nine-gaze features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import densenet121

from pure_torch_mamba import MambaResidualBlock


TYPE_TO_ID = {"dvd": 0, "eso": 1, "exo": 2}
PATTERN_TO_ID = {"no": 0, "A": 1, "V": 2}
GAZE_ORDER = ((1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0))
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed seed used only to create the train/val/test split.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=("mean", "mamba"), required=True)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scan(root: Path):
    samples = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gaze_type, pattern = class_dir.name.split("_", 1)
        type_id, pattern_id = TYPE_TO_ID[gaze_type], PATTERN_TO_ID[pattern]
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED:
                samples.append((path.resolve(), type_id, pattern_id, class_dir.name))
    if not samples:
        raise ValueError(f"No images under {root}")
    return samples


def stratified_split(samples, seed: int):
    groups = {}
    for sample in samples:
        groups.setdefault(sample[3], []).append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for _, group in sorted(groups.items()):
        rng.shuffle(group)
        n_val, n_test = max(1, round(len(group) * 0.2)), max(1, round(len(group) * 0.1))
        val.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])
        train.extend(group[n_val + n_test:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class NineGazeImages(Dataset):
    def __init__(self, samples, size: int = 224):
        self.samples, self.size = samples, size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, type_id, pattern_id, _ = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            xs, ys = (0, width // 3, 2 * width // 3, width), (0, height // 3, 2 * height // 3, height)
            frames = []
            for row, col in GAZE_ORDER:
                frame = image.crop((xs[col], ys[row], xs[col + 1], ys[row + 1]))
                frame = ImageOps.contain(frame, (self.size, self.size), Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
                canvas.paste(frame, ((self.size - frame.width) // 2, (self.size - frame.height) // 2))
                array = np.asarray(canvas, dtype=np.float32) / 255.0
                frames.append(torch.from_numpy(array.transpose(2, 0, 1)))
        video = torch.stack(frames)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (video - mean) / std, type_id, pattern_id, str(path)


def load_backbone(pretrained: Path, device: torch.device):
    model = densenet121(weights=None)
    state = torch.load(pretrained, map_location="cpu", weights_only=True)
    import re
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    for key in list(state):
        match = pattern.match(key)
        if match:
            state[match.group(1) + match.group(2)] = state.pop(key)
    result = model.load_state_dict(state, strict=True)
    model.classifier = nn.Identity()
    model.eval().requires_grad_(False).to(device)
    return model


@torch.inference_mode()
def extract_features(samples, args, device):
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(NineGazeImages(samples), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers)
    backbone = load_backbone(args.pretrained, device)
    features, type_ids, pattern_ids, paths = [], [], [], []
    for batch_index, (video, type_id, pattern_id, batch_paths) in enumerate(loader, 1):
        batch, time, channels, height, width = video.shape
        frame_features = []
        for frame_index in range(time):
            frame = video[:, frame_index].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                encoded = backbone(frame)
            frame_features.append(encoded.float().cpu())
        features.append(torch.stack(frame_features, dim=1))
        type_ids.append(type_id)
        pattern_ids.append(pattern_id)
        paths.extend(batch_paths)
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(json.dumps({"feature_batches": batch_index, "total": len(loader)}), flush=True)
    payload = {
        "features": torch.cat(features),
        "type_ids": torch.cat(type_ids),
        "pattern_ids": torch.cat(pattern_ids),
        "paths": paths,
        "gaze_order": GAZE_ORDER,
        "backbone": "torchvision DenseNet-121 ImageNet weights",
    }
    torch.save(payload, args.cache)
    return payload


class MeanTemporalClassifier(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.project = nn.Linear(feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.type_head, self.pattern_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3)

    def forward(self, features):
        hidden = self.dropout(self.norm(self.project(features).mean(dim=1)))
        return self.type_head(hidden), self.pattern_head(hidden)


class TemporalMambaClassifier(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, d_state: int, depth: int, dropout: float):
        super().__init__()
        self.project = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, 9, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)])
        self.norm, self.dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.type_head, self.pattern_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3)

    def forward(self, features):
        hidden = self.project(features) + self.position
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.dropout(self.norm(hidden).mean(dim=1))
        return self.type_head(pooled), self.pattern_head(pooled)


def metrics(type_logits, pattern_logits, type_ids, pattern_ids):
    type_pred, pattern_pred = type_logits.argmax(1), pattern_logits.argmax(1)
    exact = ((type_pred == type_ids) & (pattern_pred == pattern_ids)).float().mean().item()
    truth = [f"{a.item()}_{b.item()}" for a, b in zip(type_ids, pattern_ids)]
    pred = [f"{a.item()}_{b.item()}" for a, b in zip(type_pred, pattern_pred)]
    return {
        "exact_accuracy": exact,
        "type_accuracy": (type_pred == type_ids).float().mean().item(),
        "pattern_accuracy": (pattern_pred == pattern_ids).float().mean().item(),
        "joint_accuracy": accuracy_score(truth, pred),
        "joint_weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
    }


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total, all_t, all_p, all_tl, all_pl = 0.0, [], [], [], []
    for features, type_ids, pattern_ids in loader:
        features, type_ids, pattern_ids = features.to(device), type_ids.to(device), pattern_ids.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            type_logits, pattern_logits = model(features)
            loss = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(pattern_logits, pattern_ids)
            if training:
                loss.backward()
                optimizer.step()
        loss_total += loss.item() * features.shape[0]
        all_t.append(type_logits.detach().cpu()); all_p.append(pattern_logits.detach().cpu())
        all_tl.append(type_ids.cpu()); all_pl.append(pattern_ids.cpu())
    result = metrics(torch.cat(all_t), torch.cat(all_p), torch.cat(all_tl), torch.cat(all_pl))
    result["loss"] = loss_total / len(loader.dataset)
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    samples = scan(args.data_dir)
    train_samples, val_samples, test_samples = stratified_split(samples, args.split_seed)
    ordered = train_samples + val_samples + test_samples
    counts = {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)}
    print(json.dumps({"split_counts": counts}), flush=True)
    if args.cache.is_file():
        payload = torch.load(args.cache, map_location="cpu", weights_only=False)
        if payload["paths"] != [str(s[0]) for s in ordered]:
            raise RuntimeError("Feature cache paths do not match the seed-42 split")
    else:
        payload = extract_features(ordered, args, device)
    if args.extract_only:
        return

    features, type_ids, pattern_ids = payload["features"], payload["type_ids"], payload["pattern_ids"]
    n_train, n_val = counts["train"], counts["val"]
    datasets = {
        "train": TensorDataset(features[:n_train], type_ids[:n_train], pattern_ids[:n_train]),
        "val": TensorDataset(features[n_train:n_train+n_val], type_ids[n_train:n_train+n_val], pattern_ids[n_train:n_train+n_val]),
        "test": TensorDataset(features[n_train+n_val:], type_ids[n_train+n_val:], pattern_ids[n_train+n_val:]),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train", generator=generator if name == "train" else None)
        for name, ds in datasets.items()
    }
    feature_dim = features.shape[-1]
    model = (MeanTemporalClassifier(feature_dim, args.d_model, args.dropout) if args.model == "mean" else
             TemporalMambaClassifier(feature_dim, args.d_model, args.d_state, args.depth, args.dropout)).to(device)
    if args.dry_run:
        batch = next(iter(loaders["train"]))[0][:2].to(device)
        outputs = model(batch)
        print(json.dumps({"dry_run": True, "input": list(batch.shape), "outputs": [list(x.shape) for x in outputs]}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("split", "class", "path"))
        for split, group in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
            writer.writerows((split, sample[3], str(sample[0])) for sample in group)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best = [], -1.0
    best_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], device)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
        if val_metrics["exact_accuracy"] > best:
            best = val_metrics["exact_accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "args": vars(args)}, best_path)
        scheduler.step()
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, loaders["test"], device)
    result = {"model": args.model, "best_epoch": checkpoint["epoch"], "best_val": checkpoint["val"], "test": test_metrics}
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

