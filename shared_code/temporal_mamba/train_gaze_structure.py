#!/usr/bin/env python3
"""Run ordered and structural nine-gaze Pure-PyTorch Mamba ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pure_torch_mamba import MambaResidualBlock
from train_temporal_ablation import MeanTemporalClassifier, run_epoch, scan, seed_all, stratified_split


# The shared cache is stored in clinical order.
CLINICAL_GAZES = (5, 2, 3, 6, 9, 8, 7, 4, 1)
CLINICAL_INDEX = {gaze: index for index, gaze in enumerate(CLINICAL_GAZES)}
PERIPHERAL_ANGLES = {gaze: index * math.pi / 4 for index, gaze in enumerate((2, 3, 6, 9, 8, 7, 4, 1))}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("mean", "clinical", "random", "reverse", "row_major", "closed_loop", "circular_pe", "bidirectional", "center_ring"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fixed_random_order(seed: int) -> tuple[int, ...]:
    gazes = list(CLINICAL_GAZES)
    random.Random(seed).shuffle(gazes)
    return tuple(gazes)


def gaze_encoding(gazes: tuple[int, ...]) -> torch.Tensor:
    values = []
    for gaze in gazes:
        if gaze == 5:
            values.append((0.0, 0.0, 1.0))
        else:
            angle = PERIPHERAL_ANGLES[gaze]
            values.append((math.sin(angle), math.cos(angle), 0.0))
    return torch.tensor(values, dtype=torch.float32)


class SequenceEncoder(nn.Module):
    def __init__(self, feature_dim, d_model, d_state, depth, dropout, gazes, circular_pe=False):
        super().__init__()
        self.gazes = tuple(gazes)
        indices = [CLINICAL_INDEX[gaze] for gaze in gazes]
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long), persistent=True)
        self.project = nn.Linear(feature_dim, d_model)
        self.circular_pe = circular_pe
        if circular_pe:
            self.register_buffer("gaze_code", gaze_encoding(self.gazes), persistent=True)
            self.position_project = nn.Linear(3, d_model, bias=False)
        else:
            self.position = nn.Parameter(torch.zeros(1, len(gazes), d_model))
            nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features):
        selected = features.index_select(1, self.indices)
        hidden = self.project(selected)
        if self.circular_pe:
            hidden = hidden + self.position_project(self.gaze_code).unsqueeze(0)
        else:
            hidden = hidden + self.position
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden).mean(dim=1)


class SingleSequenceClassifier(nn.Module):
    def __init__(self, feature_dim, args, gazes, circular_pe=False):
        super().__init__()
        self.encoder = SequenceEncoder(feature_dim, args.d_model, args.d_state, args.depth, args.dropout, gazes, circular_pe)
        self.dropout = nn.Dropout(args.dropout)
        self.type_head = nn.Linear(args.d_model, 3)
        self.pattern_head = nn.Linear(args.d_model, 3)

    def forward(self, features):
        pooled = self.dropout(self.encoder(features))
        return self.type_head(pooled), self.pattern_head(pooled)


class BidirectionalClassifier(nn.Module):
    def __init__(self, feature_dim, args, include_center=True):
        super().__init__()
        if include_center:
            forward = (5, 2, 3, 6, 9, 8, 7, 4, 1, 2)
            backward = (5, 2, 1, 4, 7, 8, 9, 6, 3, 2)
        else:
            forward = (2, 3, 6, 9, 8, 7, 4, 1, 2)
            backward = (2, 1, 4, 7, 8, 9, 6, 3, 2)
        self.forward_encoder = SequenceEncoder(feature_dim, args.d_model, args.d_state, args.depth, args.dropout, forward, True)
        self.backward_encoder = SequenceEncoder(feature_dim, args.d_model, args.d_state, args.depth, args.dropout, backward, True)
        self.dropout = nn.Dropout(args.dropout)
        self.type_head = nn.Linear(args.d_model, 3)
        self.pattern_head = nn.Linear(args.d_model, 3)

    def fused_ring(self, features):
        return 0.5 * (self.forward_encoder(features) + self.backward_encoder(features))

    def forward(self, features):
        pooled = self.dropout(self.fused_ring(features))
        return self.type_head(pooled), self.pattern_head(pooled)


class CenterRingClassifier(nn.Module):
    def __init__(self, feature_dim, args):
        super().__init__()
        self.ring = BidirectionalClassifier(feature_dim, args, include_center=False)
        self.center_project = nn.Sequential(
            nn.Linear(feature_dim, args.d_model), nn.LayerNorm(args.d_model), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * args.d_model, args.d_model), nn.LayerNorm(args.d_model), nn.GELU(), nn.Dropout(args.dropout)
        )
        self.type_head = nn.Linear(args.d_model, 3)
        self.pattern_head = nn.Linear(args.d_model, 3)

    def forward(self, features):
        center = self.center_project(features[:, CLINICAL_INDEX[5]])
        ring = self.ring.fused_ring(features)
        pooled = self.fuse(torch.cat((center, ring), dim=-1))
        return self.type_head(pooled), self.pattern_head(pooled)


def build_model(feature_dim, args):
    if args.variant == "mean":
        model = MeanTemporalClassifier(feature_dim, args.d_model, args.dropout)
        return model, {"sequence": "unordered mean over identical nine features", "position": "none"}
    if args.variant == "clinical":
        return SingleSequenceClassifier(feature_dim, args, CLINICAL_GAZES), {
            "sequence": CLINICAL_GAZES, "position": "learned linear"
        }
    if args.variant == "random":
        gazes = fixed_random_order(args.seed)
        return SingleSequenceClassifier(feature_dim, args, gazes), {"sequence": gazes, "position": "learned linear"}
    if args.variant == "reverse":
        gazes = tuple(reversed(CLINICAL_GAZES))
        return SingleSequenceClassifier(feature_dim, args, gazes), {"sequence": gazes, "position": "learned linear"}
    if args.variant == "row_major":
        gazes = tuple(range(1, 10))
        return SingleSequenceClassifier(feature_dim, args, gazes), {"sequence": gazes, "position": "learned linear"}
    if args.variant == "closed_loop":
        gazes = CLINICAL_GAZES + (2,)
        return SingleSequenceClassifier(feature_dim, args, gazes), {"sequence": gazes, "position": "learned linear"}
    if args.variant == "circular_pe":
        gazes = CLINICAL_GAZES + (2,)
        return SingleSequenceClassifier(feature_dim, args, gazes, True), {"sequence": gazes, "position": "sin/cos ring + center flag"}
    if args.variant == "bidirectional":
        model = BidirectionalClassifier(feature_dim, args, include_center=True)
        return model, {
            "forward": (5, 2, 3, 6, 9, 8, 7, 4, 1, 2),
            "backward": (5, 2, 1, 4, 7, 8, 9, 6, 3, 2),
            "position": "sin/cos ring + center flag",
            "fusion": "mean of independent direction encoders",
        }
    model = CenterRingClassifier(feature_dim, args)
    return model, {
        "center": (5,),
        "ring_forward": (2, 3, 6, 9, 8, 7, 4, 1, 2),
        "ring_backward": (2, 1, 4, 7, 8, 9, 6, 3, 2),
        "position": "sin/cos ring",
        "fusion": "center MLP concatenated with bidirectional circular ring",
    }


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    samples = scan(args.data_dir)
    train_samples, val_samples, test_samples = stratified_split(samples, args.seed)
    ordered = train_samples + val_samples + test_samples
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    expected_paths = [str(sample[0]) for sample in ordered]
    if len(set(payload["paths"])) != len(payload["paths"]):
        raise RuntimeError("Feature cache contains duplicate paths")
    cache_index = {path: index for index, path in enumerate(payload["paths"])}
    try:
        reorder = torch.tensor([cache_index[path] for path in expected_paths], dtype=torch.long)
    except KeyError as exc:
        raise RuntimeError(f"Feature cache is missing dataset path: {exc.args[0]}") from exc
    # Reuse the single feature cache for every seed by reordering it to each
    # seed-specific stratified split. No visual features are re-extracted.
    features = payload["features"].index_select(0, reorder)
    type_ids = payload["type_ids"].index_select(0, reorder)
    pattern_ids = payload["pattern_ids"].index_select(0, reorder)
    n_train, n_val = len(train_samples), len(val_samples)
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
    model, design = build_model(features.shape[-1], args)
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "variant": args.variant,
        "design": design,
        "seed": args.seed,
        "epochs": args.epochs,
        "d_model": args.d_model,
        "d_state": args.d_state,
        "depth": args.depth,
        "feature_shape": list(features.shape),
        "split": {"train": n_train, "val": n_val, "test": len(test_samples)},
        "trainable_parameters": parameter_count,
        "pure_pytorch": True,
    }
    print(json.dumps(config), flush=True)
    if args.dry_run:
        batch = next(iter(loaders["train"]))[0][:2].to(device)
        outputs = model(batch)
        print(json.dumps({"dry_run": True, "outputs": [list(output.shape) for output in outputs]}), flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
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
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics, "config": config}, best_path)
        scheduler.step()
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, loaders["test"], device)
    result = {
        "variant": args.variant,
        "best_epoch": checkpoint["epoch"],
        "best_val": checkpoint["val"],
        "test": test_metrics,
        "config": config,
    }
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
