#!/usr/bin/env python3
"""End-to-end DenseNet-121 + clinical-order Pure-PyTorch Temporal Mamba."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision.models import densenet121

from pure_torch_mamba import MambaResidualBlock
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


VALID_CLASSES = (
    ("dvd_no", 0, 0), ("eso_no", 1, 0), ("eso_V", 1, 2),
    ("exo_A", 2, 1), ("exo_no", 2, 0), ("exo_V", 2, 2),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--temporal-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_densenet(path: Path):
    model = densenet121(weights=None, memory_efficient=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    for key in list(state):
        match = pattern.match(key)
        if match:
            state[match.group(1) + match.group(2)] = state.pop(key)
    model.load_state_dict(state, strict=True)
    model.classifier = nn.Identity()
    return model


class EndToEndDenseNetMamba(nn.Module):
    def __init__(self, pretrained, d_model, d_state, depth, dropout):
        super().__init__()
        self.backbone = load_densenet(pretrained)
        self.project = nn.Linear(1024, d_model)
        self.position = nn.Parameter(torch.zeros(1, 9, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)])
        self.norm, self.dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.type_head, self.pattern_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3)

    def forward(self, video):
        batch, time, channels, height, width = video.shape
        frames = video.reshape(batch * time, channels, height, width)
        if self.training and torch.is_grad_enabled():
            spatial = checkpoint(self.backbone, frames, use_reentrant=False)
        else:
            spatial = self.backbone(frames)
        hidden = self.project(spatial.reshape(batch, time, -1)) + self.position
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.dropout(self.norm(hidden).mean(dim=1))
        return self.type_head(pooled), self.pattern_head(pooled)


def epoch(model, loader, device, amp, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None; model.train(training)
    loss_sum, all_type, all_pattern, all_tl, all_pl = 0.0, [], [], [], [],
    paths = []
    for video, type_ids, pattern_ids, batch_paths in loader:
        video, type_ids, pattern_ids = video.to(device), type_ids.to(device), pattern_ids.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            type_logits, pattern_logits = model(video)
            loss = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(pattern_logits, pattern_ids)
        if training:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum += loss.item() * video.shape[0]
        all_type.append(type_logits.detach().float().cpu()); all_pattern.append(pattern_logits.detach().float().cpu())
        all_tl.append(type_ids.cpu()); all_pl.append(pattern_ids.cpu()); paths.extend(batch_paths)
    type_logits, pattern_logits = torch.cat(all_type), torch.cat(all_pattern)
    type_ids, pattern_ids = torch.cat(all_tl), torch.cat(all_pl)
    type_pred, pattern_pred = type_logits.argmax(1), pattern_logits.argmax(1)
    exact = ((type_pred == type_ids) & (pattern_pred == pattern_ids)).float().mean().item()
    truth = [f"{a.item()}_{b.item()}" for a, b in zip(type_ids, pattern_ids)]
    pred = [f"{a.item()}_{b.item()}" for a, b in zip(type_pred, pattern_pred)]
    result = {
        "loss": loss_sum / len(loader.dataset), "exact_accuracy": exact,
        "type_accuracy": (type_pred == type_ids).float().mean().item(),
        "pattern_accuracy": (pattern_pred == pattern_ids).float().mean().item(),
        "joint_weighted_f1": f1_score(truth, pred, average="weighted", zero_division=0),
    }
    if collect:
        pairs = torch.tensor([[x[1], x[2]] for x in VALID_CLASSES])
        scores = torch.log_softmax(type_logits, 1)[:, pairs[:, 0]] + torch.log_softmax(pattern_logits, 1)[:, pairs[:, 1]]
        joint_pred = scores.argmax(1).tolist(); reverse = {(x[1], x[2]): i for i, x in enumerate(VALID_CLASSES)}
        joint_true = [reverse[(int(a), int(b))] for a, b in zip(type_ids, pattern_ids)]
        names = [x[0] for x in VALID_CLASSES]
        result.update({
            "six_class_labels": names,
            "six_class_confusion_matrix": confusion_matrix(joint_true, joint_pred, labels=range(6)).tolist(),
            "six_class_report": classification_report(joint_true, joint_pred, target_names=names, output_dict=True, zero_division=0),
            "type_confusion_matrix": confusion_matrix(type_ids, type_pred, labels=range(3)).tolist(),
            "pattern_confusion_matrix": confusion_matrix(pattern_ids, pattern_pred, labels=range(3)).tolist(),
        })
    return result


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples = scan(args.data_dir); train, val, test = stratified_split(samples, args.split_seed)
    loaders = {
        "train": DataLoader(NineGazeImages(train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed)),
        "val": DataLoader(NineGazeImages(val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
        "test": DataLoader(NineGazeImages(test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
    }
    model = EndToEndDenseNetMamba(args.pretrained, args.d_model, args.d_state, args.depth, args.dropout).to(device)
    temporal = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": temporal, "lr": args.temporal_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if args.dry_run:
        video, type_ids, pattern_ids, _ = next(iter(loaders["train"])); video = video.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(video); loss = F.cross_entropy(outputs[0], type_ids.to(device)) + F.cross_entropy(outputs[1], pattern_ids.to(device))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        print(json.dumps({"dry_run": True, "video": list(video.shape), "loss": float(loss), "cuda_max_memory_mb": torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else 0}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(("split","class","path"))
        for split, group in (("train",train),("val",val),("test",test)): writer.writerows((split,x[3],str(x[0])) for x in group)
    config = {"seed":args.seed,"split_seed":args.split_seed,"epochs":args.epochs,"batch_size":args.batch_size,"num_workers":args.num_workers,"amp":args.amp,"backbone_lr":args.backbone_lr,"temporal_lr":args.temporal_lr,"weight_decay":args.weight_decay,"d_model":args.d_model,"d_state":args.d_state,"depth":args.depth,"dropout":args.dropout,"input":"nine 224x224 clinical-order crops","joint_fine_tuning":"all DenseNet and Temporal Mamba parameters"}
    (args.output_dir / "model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    history=[];best=-1.0;best_path=args.output_dir/"best.pt"
    for index in range(1,args.epochs+1):
        train_metrics=epoch(model,loaders["train"],device,args.amp,optimizer,scaler)
        val_metrics=epoch(model,loaders["val"],device,args.amp)
        row={"epoch":index,"backbone_lr":optimizer.param_groups[0]["lr"],"temporal_lr":optimizer.param_groups[1]["lr"],"train":train_metrics,"val":val_metrics}
        history.append(row);print(json.dumps(row),flush=True)
        if val_metrics["exact_accuracy"]>best:
            best=val_metrics["exact_accuracy"];torch.save({"model":model.state_dict(),"epoch":index,"val":val_metrics,"config":config},best_path)
        scheduler.step()
    checkpoint_payload=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(checkpoint_payload["model"])
    test_metrics=epoch(model,loaders["test"],device,args.amp,collect=True)
    result={"best_epoch":checkpoint_payload["epoch"],"best_val":checkpoint_payload["val"],"test":test_metrics,"config":config}
    (args.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (args.output_dir/"test_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
