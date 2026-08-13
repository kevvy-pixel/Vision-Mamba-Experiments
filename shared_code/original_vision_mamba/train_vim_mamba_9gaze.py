#!/usr/bin/env python3
"""Train a dual-head Vim + temporal Mamba classifier on composed 9-gaze images.

Each input is a 3x3 clinical gaze grid.  It is converted into a pseudo-video in
the order: primary, up, up-right, right, down-right, down, down-left, left,
up-left.  Vim encodes each frame and Mamba models the resulting 9-token video.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


TYPE_TO_ID = {"dvd": 0, "eso": 1, "exo": 2}
PATTERN_TO_ID = {"no": 0, "A": 1, "V": 2}
ID_TO_TYPE = {value: key for key, value in TYPE_TO_ID.items()}
ID_TO_PATTERN = {value: key for key, value in PATTERN_TO_ID.items()}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# (row, column) positions in a conventional 3x3 clinical gaze chart.
GAZE_ORDER = (
    (1, 1),  # primary position
    (0, 1),  # up
    (0, 2),  # up-right
    (1, 2),  # right
    (2, 2),  # down-right
    (2, 1),  # down
    (2, 0),  # down-left
    (1, 0),  # left
    (0, 0),  # up-left
)
GAZE_NAMES = ("primary", "up", "up_right", "right", "down_right", "down", "down_left", "left", "up_left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Source_9gaze_composed"))
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Held-out source-domain test fraction.")
    parser.add_argument("--vim-dir", type=Path, default=Path("Vim"))
    parser.add_argument("--pretrained", type=Path, default=None, help="Official Vim checkpoint (.pth).")
    parser.add_argument("--resume", type=Path, default=None, help="Resume model weights from a training checkpoint.")
    parser.add_argument("--start-epoch", type=int, default=0, help="Last completed epoch when resuming.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vim_base_mamba_source_721_seed42"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--temporal-depth", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--type-loss-weight", type=float, default=1.0)
    parser.add_argument("--pattern-loss-weight", type=float, default=1.0)
    parser.add_argument("--frame-chunk-size", type=int, default=0, help="Encode this many frames at once; 0 encodes B*9 together.")
    parser.add_argument("--freeze-vim", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check data/model and run one forward pass.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_class_name(name: str) -> tuple[int, int]:
    try:
        gaze_type, pattern = name.split("_", maxsplit=1)
        return TYPE_TO_ID[gaze_type], PATTERN_TO_ID[pattern]
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"Invalid class directory {name!r}; expected <dvd|eso|exo>_<no|A|V>."
        ) from exc


def scan_dataset(root: Path) -> list[tuple[Path, int, int, str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset does not exist: {root}")
    samples: list[tuple[Path, int, int, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        type_id, pattern_id = parse_class_name(class_dir.name)
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                samples.append((path, type_id, pattern_id, class_dir.name))
    if not samples:
        raise ValueError(f"No supported images found under {root}")
    return samples


def stratified_split(samples: list[tuple[Path, int, int, str]], val_ratio: float, test_ratio: float, seed: int):
    if val_ratio <= 0.0 or test_ratio <= 0.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio and --test-ratio must be positive and sum to less than 1")
    groups: dict[str, list[tuple[Path, int, int, str]]] = {}
    for sample in samples:
        groups.setdefault(sample[3], []).append(sample)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for name, group in sorted(groups.items()):
        rng.shuffle(group)
        if len(group) < 3:
            raise ValueError(f"Class {name} needs at least three images for train/val/test splitting")
        n_val = max(1, round(len(group) * val_ratio))
        n_test = max(1, round(len(group) * test_ratio))
        if n_val + n_test >= len(group):
            raise ValueError(f"Class {name} is too small for the requested split")
        val.extend(group[:n_val])
        test.extend(group[n_val:n_val + n_test])
        train.extend(group[n_val + n_test:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class NineGazeDataset(Dataset):
    def __init__(self, samples, image_size: int = 224, augment: bool = False) -> None:
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def split_grid(image: Image.Image) -> list[Image.Image]:
        width, height = image.size
        if width < 3 or height < 3:
            raise ValueError(f"Image is too small to be a 3x3 grid: {image.size}")
        x_edges = (0, width // 3, 2 * width // 3, width)
        y_edges = (0, height // 3, 2 * height // 3, height)
        grid = {
            (row, col): image.crop((x_edges[col], y_edges[row], x_edges[col + 1], y_edges[row + 1]))
            for row in range(3) for col in range(3)
        }
        return [grid[position] for position in GAZE_ORDER]

    def _letterbox(self, frame: Image.Image) -> Image.Image:
        frame = ImageOps.contain(frame, (self.image_size, self.image_size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        canvas.paste(frame, ((self.image_size - frame.width) // 2, (self.image_size - frame.height) // 2))
        return canvas

    def __getitem__(self, index: int):
        path, type_id, pattern_id, _ = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
            # Identical photometric transform for all nine frames preserves time coherence.
            if self.augment:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.9, 1.1))
            frames = self.split_grid(image)
        arrays = []
        for frame in frames:
            array = np.asarray(self._letterbox(frame), dtype=np.float32) / 255.0
            arrays.append(torch.from_numpy(array.transpose(2, 0, 1)))
        video = torch.stack(arrays)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (video - mean) / std, torch.tensor(type_id), torch.tensor(pattern_id), str(path)


def import_vim(vim_dir: Path):
    source_dir = (vim_dir / "vim").resolve()
    if not (source_dir / "models_mamba.py").is_file():
        raise FileNotFoundError(f"Official Vim source not found under {vim_dir}")
    sys.path.insert(0, str(source_dir))
    from models_mamba import VisionMamba
    return VisionMamba


class VimMambaDualHead(nn.Module):
    def __init__(self, vim_dir: Path, image_size: int, temporal_depth: int, d_state: int, dropout: float, freeze_vim: bool) -> None:
        super().__init__()
        VisionMamba = import_vim(vim_dir)
        from mamba_ssm.modules.mamba_simple import Mamba
        self.spatial = VisionMamba(
            img_size=image_size, patch_size=16, stride=16, embed_dim=768, depth=24,
            d_state=16, num_classes=0, rms_norm=True, residual_in_fp32=True,
            fused_add_norm=True, final_pool_type="mean", if_abs_pos_embed=True,
            if_rope=False, if_rope_residual=False, bimamba_type="v2",
            if_cls_token=True, if_divide_out=True, use_middle_cls_token=True,
        )
        if freeze_vim:
            self.spatial.requires_grad_(False)
        self.temporal_pos = nn.Parameter(torch.zeros(1, len(GAZE_ORDER), 768))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.temporal = nn.ModuleList([Mamba(d_model=768, d_state=d_state, d_conv=3, expand=2, layer_idx=i) for i in range(temporal_depth)])
        self.temporal_norms = nn.ModuleList([nn.LayerNorm(768) for _ in range(temporal_depth)])
        self.final_norm = nn.LayerNorm(768)
        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(768, 3)
        self.pattern_head = nn.Linear(768, 3)
        self.frame_chunk_size = 0

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.frame_chunk_size <= 0:
            return self.spatial(frames, return_features=True)
        return torch.cat([
            self.spatial(chunk, return_features=True)
            for chunk in frames.split(self.frame_chunk_size)
        ], dim=0)

    def forward(self, video: torch.Tensor):
        batch, time, channels, height, width = video.shape
        tokens = self.encode_frames(video.reshape(batch * time, channels, height, width)).reshape(batch, time, -1)
        hidden = tokens + self.temporal_pos[:, :time]
        for norm, mamba in zip(self.temporal_norms, self.temporal):
            hidden = hidden + mamba(norm(hidden))
        pooled = self.dropout(self.final_norm(hidden).mean(dim=1))
        return self.type_head(pooled), self.pattern_head(pooled)


def load_pretrained(model: VimMambaDualHead, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", payload))
    cleaned = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("spatial.")
        if not key.startswith("head."):
            cleaned[key] = value
    result = model.spatial.load_state_dict(cleaned, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Vim checkpoint mismatch: {result}")
    print("Loaded Vim checkpoint with exact key match", flush=True)


def make_loader(samples, args, training: bool):
    return DataLoader(
        NineGazeDataset(samples, args.image_size, training and not args.no_augment),
        batch_size=args.batch_size, shuffle=training, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"), drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def run_epoch(model, loader, device, criterion, optimizer, scaler, args):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    type_true, type_pred, pattern_true, pattern_pred = [], [], [], []
    for video, type_label, pattern_label, _ in loader:
        video = video.to(device, non_blocking=True)
        type_label = type_label.to(device, non_blocking=True)
        pattern_label = pattern_label.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            type_logits, pattern_logits = model(video)
            loss = args.type_loss_weight * criterion(type_logits, type_label) + args.pattern_loss_weight * criterion(pattern_logits, pattern_label)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * video.size(0)
        type_true.extend(type_label.cpu().tolist())
        pattern_true.extend(pattern_label.cpu().tolist())
        type_pred.extend(type_logits.argmax(1).cpu().tolist())
        pattern_pred.extend(pattern_logits.argmax(1).cpu().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "type_accuracy": accuracy_score(type_true, type_pred),
        "type_macro_f1": f1_score(type_true, type_pred, average="macro", zero_division=0),
        "pattern_accuracy": accuracy_score(pattern_true, pattern_pred),
        "pattern_macro_f1": f1_score(pattern_true, pattern_pred, average="macro", zero_division=0),
        "mean_accuracy": (accuracy_score(type_true, type_pred) + accuracy_score(pattern_true, pattern_pred)) / 2,
        "type_confusion_matrix": confusion_matrix(type_true, type_pred, labels=[0, 1, 2]).tolist(),
        "pattern_confusion_matrix": confusion_matrix(pattern_true, pattern_pred, labels=[0, 1, 2]).tolist(),
    }


def write_manifest(output_dir: Path, partitions) -> None:
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "path", "type", "pattern", "source_class"))
        for split, samples in partitions:
            for path, type_id, pattern_id, class_name in samples:
                writer.writerow((split, path.resolve(), ID_TO_TYPE[type_id], ID_TO_PATTERN[pattern_id], class_name))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; pass --device cpu for a data-only smoke test")
    device = torch.device(args.device)
    all_samples = scan_dataset(args.data_dir)
    train_samples, val_samples, test_samples = stratified_split(all_samples, args.val_ratio, args.test_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir, (("train", train_samples), ("val", val_samples), ("test", test_samples)))
    summary = {
        "gaze_order": list(GAZE_NAMES),
        "type_classes": TYPE_TO_ID,
        "pattern_classes": PATTERN_TO_ID,
        "split_ratios": {"train": 1.0 - args.val_ratio - args.test_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "counts": {"all": Counter(sample[3] for sample in all_samples), "train": Counter(sample[3] for sample in train_samples), "val": Counter(sample[3] for sample in val_samples), "test": Counter(sample[3] for sample in test_samples)},
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    model = VimMambaDualHead(args.vim_dir, args.image_size, args.temporal_depth, args.d_state, args.dropout, args.freeze_vim)
    model.frame_chunk_size = args.frame_chunk_size
    if args.pretrained:
        load_pretrained(model, args.pretrained)
    model.to(device)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model"])
        print(f"Resumed model from {args.resume} at epoch {args.start_epoch}", flush=True)
    train_loader = make_loader(train_samples, args, True)
    val_loader = make_loader(val_samples, args, False)
    if args.dry_run:
        video, _, _, paths = next(iter(val_loader))
        with torch.no_grad():
            outputs = model(video.to(device))
        print(f"dry-run ok: video={tuple(video.shape)}, type_logits={tuple(outputs[0].shape)}, pattern_logits={tuple(outputs[1].shape)}, sample={paths[0]}")
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.resume and args.start_epoch > 0:
        resumed_lr = args.lr * (1.0 + np.cos(np.pi * args.start_epoch / args.epochs)) / 2.0
        for group in optimizer.param_groups:
            group["lr"] = resumed_lr
        scheduler.last_epoch = args.start_epoch
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    history_path = args.output_dir / "history.json"
    history: list[dict[str, Any]] = [item for item in json.loads(history_path.read_text()) if int(item["epoch"]) <= args.start_epoch] if args.resume and history_path.is_file() else []
    best_score = float(resume_payload.get("metrics", {}).get("mean_accuracy", -1.0)) if resume_payload else -1.0
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler, args)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, criterion, None, scaler, args)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(record)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} type_acc={val_metrics['type_accuracy']:.4f} pattern_acc={val_metrics['pattern_accuracy']:.4f}", flush=True)
        if val_metrics["mean_accuracy"] > best_score:
            best_score = val_metrics["mean_accuracy"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "metrics": val_metrics}, args.output_dir / "best.pt")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loader = make_loader(test_samples, args, False)
    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, device, criterion, None, scaler, args)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
