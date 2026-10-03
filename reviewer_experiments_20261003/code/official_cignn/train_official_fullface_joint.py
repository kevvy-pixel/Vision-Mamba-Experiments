from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


CLASSES = ["dvd_no", "eso_no", "eso_V", "exo_A", "exo_no", "exo_V"]
# Official labels: type 0=ESO, 1=EXO, 2=vertical; pattern 0=A, 1=V, 2=no pattern.
LEGAL = [(2, 2), (0, 2), (0, 1), (1, 0), (1, 2), (1, 1)]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def index_nodes(node_dir: Path) -> list[Path]:
    return sorted(node_dir.glob("*.npy"))


def unique_match(subject: str, files: list[Path]) -> Path | None:
    matches = [p for p in files if subject in p.stem]
    return matches[0] if len(matches) == 1 else None


def load_dataset(data_root: Path, source_root: Path):
    graph_root = source_root / "GNN" / "data"
    ex_nodes = index_nodes(graph_root / "ex_9_data" / "node")
    av_nodes = index_nodes(graph_root / "av_data" / "node")
    rows = []
    excluded = []
    for cls in CLASSES:
        for subject_dir in sorted((data_root / cls).iterdir()):
            if not subject_dir.is_dir():
                continue
            images = sorted(subject_dir.glob("*.jpg"))
            ex_path = unique_match(subject_dir.name, ex_nodes)
            av_path = unique_match(subject_dir.name, av_nodes)
            if len(images) != 9 or ex_path is None or av_path is None:
                excluded.append(
                    {"class": cls, "subject": subject_dir.name, "images": len(images),
                     "has_ex": ex_path is not None, "has_av": av_path is not None}
                )
                continue
            ex = np.load(ex_path).astype(np.float32)
            av = np.load(av_path).astype(np.float32)
            if ex.shape != (9, 5) or av.shape != (9, 9):
                excluded.append(
                    {"class": cls, "subject": subject_dir.name, "images": len(images),
                     "has_ex": True, "has_av": True, "ex_shape": list(ex.shape), "av_shape": list(av.shape)}
                )
                continue
            joint = CLASSES.index(cls)
            type_y, pattern_y = LEGAL[joint]
            ex_label = int(float((graph_root / "ex_9_data" / "label" / f"{ex_path.stem}.txt").read_text().strip()))
            av_label = int(float((graph_root / "av_data" / "label" / f"{av_path.stem}.txt").read_text().strip()))
            if (ex_label, av_label) != (type_y, pattern_y):
                excluded.append(
                    {"class": cls, "subject": subject_dir.name, "images": len(images),
                     "has_ex": True, "has_av": True, "reason": "label_disagreement",
                     "official_type": ex_label, "official_pattern": av_label,
                     "folder_type": type_y, "folder_pattern": pattern_y}
                )
                continue
            rows.append(
                {"class": cls, "subject": subject_dir.name, "joint": joint,
                 "type": type_y, "pattern": pattern_y, "ex_path": str(ex_path), "av_path": str(av_path),
                 "ex": ex, "av": av}
            )
    return rows, excluded


def make_split(joint: np.ndarray, seed: int) -> np.ndarray:
    idx = np.arange(len(joint))
    train, temporary = train_test_split(
        idx, test_size=0.30, random_state=seed, stratify=joint
    )
    val, test = train_test_split(
        temporary, test_size=0.50, random_state=seed, stratify=joint[temporary]
    )
    split = np.full(len(joint), "", dtype="<U5")
    split[train], split[val], split[test] = "train", "val", "test"
    return split


def load_official_classes(source_root: Path):
    for name in ["thop", "torchsummary", "torchinfo"]:
        module = types.ModuleType(name)
        module.profile = lambda model, inputs=(): (0, sum(p.numel() for p in model.parameters()))
        module.summary = lambda *args, **kwargs: None
        sys.modules.setdefault(name, module)
    gnn_root = source_root / "GNN"
    sys.path.insert(0, str(gnn_root))
    from models.GCN import GNN_single
    from models.NormClassifier import LinearClsHead
    from utils import WarmupMultiStepLR
    return GNN_single, LinearClsHead, WarmupMultiStepLR


class OfficialTask(nn.Module):
    def __init__(self, gnn_cls, head_cls, num_head: int):
        super().__init__()
        self.gnn = gnn_cls(in_channels=9, hidden_channels=(2048, 2048, 2048), dropout=None)
        self.head = head_cls(
            num_classes=3, feat_dim=2048, use_effect=True, num_head=num_head,
            tau=8.0, alpha=0.5, gamma=0.0625
        )

    def forward(self, x):
        features = self.gnn(x)
        logits, _ = self.head(
            features, torch.zeros(len(x), dtype=torch.long, device=x.device), None
        )
        return logits


def loaders(x, y, split, batch_size, seed):
    result = {}
    for phase in ["train", "val", "test"]:
        mask = split == phase
        generator = torch.Generator().manual_seed(seed) if phase == "train" else None
        result[phase] = DataLoader(
            TensorDataset(torch.from_numpy(x[mask]), torch.from_numpy(y[mask])),
            batch_size=batch_size, shuffle=phase == "train", generator=generator, num_workers=0
        )
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, logits = [], [], []
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss_sum += nn.functional.cross_entropy(output, y).item() * len(y)
        ys.extend(y.cpu().tolist())
        preds.extend(output.argmax(1).cpu().tolist())
        logits.append(output.cpu())
    metrics = {
        "loss": loss_sum / len(ys),
        "accuracy": float(accuracy_score(ys, preds)),
        "weighted_f1": float(f1_score(ys, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(ys, preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(ys, preds, labels=[0, 1, 2]).tolist(),
    }
    return metrics, torch.cat(logits), np.asarray(ys), np.asarray(preds)


def train_task(name, x, y, split, gnn_cls, head_cls, scheduler_cls, output_dir, args, device):
    batch_size = args.pattern_batch_size if name == "pattern" else args.type_batch_size
    data = loaders(x, y, split, batch_size, args.seed)
    model = OfficialTask(gnn_cls, head_cls, 8 if name == "pattern" else 4).to(device)
    lr = args.pattern_lr if name == "pattern" else args.type_lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    scheduler = scheduler_cls(
        optimizer, milestones=[120, 160], gamma=0.01, warmup_epochs=10
    )
    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in data["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        scheduler.step()
        val, _, _, _ = evaluate(model, data["val"], device)
        record = {
            "epoch": epoch, "train_loss": total_loss / len(data["train"].dataset),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"val_{key}": value for key, value in val.items()}
        }
        history.append(record)
        if val["accuracy"] > best_acc + 1e-12:
            best_acc = val["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, output_dir / f"best_{name}.pt")
        if epoch == 1 or epoch % 25 == 0:
            print(f"{name} epoch={epoch:04d} val_acc={val['accuracy']:.4f} best={best_acc:.4f}", flush=True)
    checkpoint = torch.load(output_dir / f"best_{name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics, test_logits, test_truth, test_pred = evaluate(model, data["test"], device)
    (output_dir / f"history_{name}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, checkpoint, test_metrics, test_logits, test_truth, test_pred


def six_class_metrics(truth, pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=range(6), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "weighted_f1": float(f1_score(truth, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=range(6)).tolist(),
        "per_class": {
            CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]),
                         "f1": float(f1[i]), "support": int(support[i])}
            for i in range(6)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--type-batch-size", type=int, default=128)
    parser.add_argument("--pattern-batch-size", type=int, default=32)
    parser.add_argument("--type-lr", type=float, default=0.05)
    parser.add_argument("--pattern-lr", type=float, default=0.01)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, excluded = load_dataset(args.data_root, args.source_root)
    ex = np.stack([row.pop("ex") for row in rows])
    av = np.stack([row.pop("av") for row in rows])
    joint = np.asarray([row["joint"] for row in rows], dtype=np.int64)
    type_y = np.asarray([row["type"] for row in rows], dtype=np.int64)
    pattern_y = np.asarray([row["pattern"] for row in rows], dtype=np.int64)
    split = make_split(joint, args.seed)
    for row, phase in zip(rows, split):
        row["split"] = phase

    with (args.output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "excluded.json").write_text(json.dumps(excluded, indent=2, ensure_ascii=False), encoding="utf-8")
    prepared = {
        "matched_subjects": len(rows), "excluded_subjects": len(excluded),
        "split_counts": dict(Counter(split)), "class_counts": dict(Counter(row["class"] for row in rows)),
        "ex_shape": list(ex.shape), "av_shape": list(av.shape)
    }
    print(json.dumps(prepared, indent=2, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return

    gnn_cls, head_cls, scheduler_cls = load_official_classes(args.source_root)
    device = torch.device("cuda")
    type_model, type_ck, type_metrics, type_logits, type_truth, type_pred = train_task(
        "type", ex, type_y, split, gnn_cls, head_cls, scheduler_cls, args.output_dir, args, device
    )
    type_model.cpu()
    torch.cuda.empty_cache()
    pattern_model, pattern_ck, pattern_metrics, pattern_logits, pattern_truth, pattern_pred = train_task(
        "pattern", av, pattern_y, split, gnn_cls, head_cls, scheduler_cls, args.output_dir, args, device
    )

    test_mask = split == "test"
    joint_truth = joint[test_mask]
    exact_match = (type_pred == type_truth) & (pattern_pred == pattern_truth)
    type_logp = type_logits.log_softmax(1)
    pattern_logp = pattern_logits.log_softmax(1)
    legal_scores = torch.stack(
        [type_logp[:, type_index] + pattern_logp[:, pattern_index]
         for type_index, pattern_index in LEGAL], dim=1
    )
    legal_pred = legal_scores.argmax(1).numpy()
    result = {
        "method_name": "Official-feature CI-GNN joint re-evaluation",
        "data_protocol": "E:/data full-face nine-view subjects; official precomputed nodes",
        "test_samples": int(test_mask.sum()),
        "type_task": type_metrics,
        "pattern_task": pattern_metrics,
        "joint_exact_match_accuracy": float(exact_match.mean()),
        "joint_exact_match_correct": int(exact_match.sum()),
        "six_subtype_legal_decode": six_class_metrics(joint_truth, legal_pred),
        "best_type_epoch": int(type_ck["epoch"]),
        "best_pattern_epoch": int(pattern_ck["epoch"]),
        "prepared": prepared,
        "comparison_warning": "Different input dataset and split from GazeMamba; descriptive comparison only."
    }
    (args.output_dir / "test_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "true_class", "type_true", "type_pred", "pattern_true", "pattern_pred",
                         "both_correct", "legal_six_pred"])
        test_rows = [row for row in rows if row["split"] == "test"]
        for row, tt, tp, pt, pp, both, six_pred in zip(
            test_rows, type_truth, type_pred, pattern_truth, pattern_pred, exact_match, legal_pred
        ):
            writer.writerow([row["subject"], row["class"], tt, tp, pt, pp, int(both), CLASSES[int(six_pred)]])
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
