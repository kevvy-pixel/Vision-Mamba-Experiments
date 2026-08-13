#!/usr/bin/env python3
"""End-to-end lightweight shared CNN + Clinical Mamba + attention ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pure_torch_mamba import MambaResidualBlock
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


VALID_CLASSES = (
    ("dvd_no", 0, 0), ("eso_no", 1, 0), ("eso_V", 1, 2),
    ("exo_A", 2, 1), ("exo_no", 2, 0), ("exo_V", 2, 2),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=("cnn_mean", "cnn_mamba_mean", "cnn_mamba_attention"), required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def conv_stage(in_ch, out_ch):
    groups = min(16, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.GroupNorm(groups, out_ch), nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_ch), nn.GELU(),
    )


class LightweightCNN(nn.Module):
    """A small shared gaze encoder; approximately 0.7M parameters."""
    def __init__(self, d_model=192):
        super().__init__()
        self.features = nn.Sequential(
            conv_stage(3, 32), conv_stage(32, 64),
            conv_stage(64, 128), conv_stage(128, 192),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(192, d_model)

    def forward(self, x):
        return self.project(self.pool(self.features(x)).flatten(1))


class GatedAttentionPool(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        hidden = max(32, d_model // 2)
        self.v = nn.Linear(d_model, hidden)
        self.u = nn.Linear(d_model, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        logits = self.score(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        return self.dropout(torch.sum(x * weights.unsqueeze(-1), dim=1)), weights


class LightweightClinicalModel(nn.Module):
    def __init__(self, model_name, d_model, d_state, depth, dropout):
        super().__init__()
        self.model_name = model_name
        self.cnn = LightweightCNN(d_model)
        self.position = nn.Parameter(torch.zeros(1, 9, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)]
            if model_name != "cnn_mean" else []
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = GatedAttentionPool(d_model, dropout) if model_name == "cnn_mamba_attention" else None
        self.type_head = nn.Linear(d_model, 3)
        self.pattern_head = nn.Linear(d_model, 3)

    def forward(self, video, return_attention=False):
        b, t, c, h, w = video.shape
        hidden = self.cnn(video.reshape(b * t, c, h, w)).reshape(b, t, -1)
        if self.model_name != "cnn_mean":
            hidden = hidden + self.position
            for block in self.blocks:
                hidden = block(hidden)
        hidden = self.norm(hidden)
        if self.attention is None:
            pooled, weights = self.dropout(hidden.mean(dim=1)), None
        else:
            pooled, weights = self.attention(hidden)
        output = (self.type_head(pooled), self.pattern_head(pooled))
        return (*output, weights) if return_attention else output


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, tls, pls, type_logits_all, pattern_logits_all, weights_all = 0.0, [], [], [], [], []
    for video, type_ids, pattern_ids, _ in loader:
        video, type_ids, pattern_ids = video.to(device), type_ids.to(device), pattern_ids.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            type_logits, pattern_logits, weights = model(video, return_attention=True)
            loss = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(pattern_logits, pattern_ids)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * video.shape[0]
        tls.append(type_ids.cpu()); pls.append(pattern_ids.cpu())
        type_logits_all.append(type_logits.detach().float().cpu())
        pattern_logits_all.append(pattern_logits.detach().float().cpu())
        if weights is not None:
            weights_all.append(weights.detach().float().cpu())
    type_ids, pattern_ids = torch.cat(tls), torch.cat(pls)
    type_logits, pattern_logits = torch.cat(type_logits_all), torch.cat(pattern_logits_all)
    type_pred, pattern_pred = type_logits.argmax(1), pattern_logits.argmax(1)
    truth = [f"{a.item()}_{b.item()}" for a, b in zip(type_ids, pattern_ids)]
    pred = [f"{a.item()}_{b.item()}" for a, b in zip(type_pred, pattern_pred)]
    result = {
        "loss": loss_sum / len(loader.dataset),
        "exact_accuracy": ((type_pred == type_ids) & (pattern_pred == pattern_ids)).float().mean().item(),
        "type_accuracy": (type_pred == type_ids).float().mean().item(),
        "pattern_accuracy": (pattern_pred == pattern_ids).float().mean().item(),
        "joint_weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
        "joint_macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
    }
    if collect:
        pairs = torch.tensor([[x[1], x[2]] for x in VALID_CLASSES])
        scores = torch.log_softmax(type_logits, 1)[:, pairs[:, 0]] + torch.log_softmax(pattern_logits, 1)[:, pairs[:, 1]]
        joint_pred = scores.argmax(1).tolist()
        reverse = {(x[1], x[2]): i for i, x in enumerate(VALID_CLASSES)}
        joint_true = [reverse[(int(a), int(b))] for a, b in zip(type_ids, pattern_ids)]
        names = [x[0] for x in VALID_CLASSES]
        result.update({
            "six_class_labels": names,
            "six_class_confusion_matrix": confusion_matrix(joint_true, joint_pred, labels=range(6)).tolist(),
            "six_class_report": classification_report(joint_true, joint_pred, target_names=names, output_dict=True, zero_division=0),
            "type_confusion_matrix": confusion_matrix(type_ids, type_pred, labels=range(3)).tolist(),
            "pattern_confusion_matrix": confusion_matrix(pattern_ids, pattern_pred, labels=range(3)).tolist(),
        })
        if weights_all:
            weights = torch.cat(weights_all)
            result["attention_mean_clinical_order"] = weights.mean(0).tolist()
            result["attention_std_clinical_order"] = weights.std(0, unbiased=False).tolist()
    return result


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples = scan(args.data_dir); train, val, test = stratified_split(samples, args.seed)
    loaders = {
        "train": DataLoader(NineGazeImages(train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed)),
        "val": DataLoader(NineGazeImages(val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(NineGazeImages(test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    model = LightweightClinicalModel(args.model, args.d_model, args.d_state, args.depth, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    counts = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "cnn": sum(p.numel() for p in model.cnn.parameters())}
    if args.dry_run:
        video, type_ids, pattern_ids, _ = next(iter(loaders["train"])); video = video.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(video); loss = F.cross_entropy(outputs[0], type_ids.to(device)) + F.cross_entropy(outputs[1], pattern_ids.to(device))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        print(json.dumps({"dry_run": True, "model": args.model, "video": list(video.shape), "loss": float(loss), "parameters": counts, "cuda_max_memory_mb": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else 0}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(("split", "class", "path"))
        for split, group in (("train", train), ("val", val), ("test", test)):
            writer.writerows((split, x[3], str(x[0])) for x in group)
    config = vars(args).copy()
    config.update({"data_dir": str(args.data_dir), "output_dir": str(args.output_dir), "parameters": counts, "input": "nine 224x224 clinical-order crops", "gaze_order": "5-2-3-6-9-8-7-4-1", "attention": "gated attention pooling over nine Mamba states" if args.model.endswith("attention") else "none"})
    (args.output_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history, best, best_path = [], -1.0, args.output_dir / "best.pt"
    for index in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, args.amp, optimizer, scaler)
        val_metrics = run_epoch(model, loaders["val"], device, args.amp)
        row = {"epoch": index, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(row); print(json.dumps(row), flush=True)
        if val_metrics["exact_accuracy"] > best:
            best = val_metrics["exact_accuracy"]
            torch.save({"model": model.state_dict(), "epoch": index, "val": val_metrics, "config": config}, best_path)
        scheduler.step()
    checkpoint = torch.load(best_path, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, loaders["test"], device, args.amp, collect=True)
    result = {"best_epoch": checkpoint["epoch"], "best_val": checkpoint["val"], "test": test_metrics, "config": config}
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.output_dir / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
