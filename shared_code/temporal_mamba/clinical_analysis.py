#!/usr/bin/env python3
"""Generate six-class, type, and pattern clinical error analyses."""

from __future__ import annotations

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from train_gaze_structure import build_model
from train_temporal_ablation import TemporalMambaClassifier, scan, seed_all, stratified_split


TYPE_NAMES = ("dvd", "eso", "exo")
PATTERN_NAMES = ("no", "A", "V")
VALID_CLASSES = (
    ("dvd_no", 0, 0),
    ("eso_no", 1, 0),
    ("eso_V", 1, 2),
    ("exo_A", 2, 1),
    ("exo_no", 2, 0),
    ("exo_V", 2, 2),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def write_matrix(path: Path, labels, matrix):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("true/pred", *labels))
        for label, row in zip(labels, matrix):
            writer.writerow((label, *row))


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples = scan(args.data_dir)
    _, _, test_samples = stratified_split(samples, args.seed)
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    cache_index = {path: index for index, path in enumerate(payload["paths"])}
    indices = torch.tensor([cache_index[str(sample[0])] for sample in test_samples])
    features = payload["features"].index_select(0, indices)
    true_type = payload["type_ids"].index_select(0, indices)
    true_pattern = payload["pattern_ids"].index_select(0, indices)
    loader = DataLoader(TensorDataset(features, true_type, true_pattern), batch_size=64)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    model_args = Namespace(
        variant=args.variant,
        seed=args.seed,
        d_model=int(config.get("d_model", 192)),
        d_state=int(config.get("d_state", 16)),
        depth=int(config.get("depth", 2)),
        dropout=0.1,
    )
    if args.variant == "clinical" and not any(key.startswith("encoder.") for key in checkpoint["model"]):
        # The original seed-42 clinical checkpoint predates the unified ablation
        # runner but has the same architecture and hyperparameters.
        model = TemporalMambaClassifier(features.shape[-1], model_args.d_model, model_args.d_state, model_args.depth, model_args.dropout)
        design = {"sequence": (5, 2, 3, 6, 9, 8, 7, 4, 1), "position": "learned linear"}
    else:
        model, design = build_model(features.shape[-1], model_args)
    model.load_state_dict(checkpoint["model"]); model.to(device).eval()
    type_logits_all, pattern_logits_all = [], []
    with torch.inference_mode():
        for batch_features, _, _ in loader:
            type_logits, pattern_logits = model(batch_features.to(device))
            type_logits_all.append(type_logits.cpu()); pattern_logits_all.append(pattern_logits.cpu())
    type_logits, pattern_logits = torch.cat(type_logits_all), torch.cat(pattern_logits_all)
    pred_type, pred_pattern = type_logits.argmax(1), pattern_logits.argmax(1)

    class_names = [item[0] for item in VALID_CLASSES]
    pairs = torch.tensor([[item[1], item[2]] for item in VALID_CLASSES])
    joint_scores = torch.log_softmax(type_logits, 1)[:, pairs[:, 0]] + torch.log_softmax(pattern_logits, 1)[:, pairs[:, 1]]
    pred_joint = joint_scores.argmax(1).tolist()
    reverse = {(item[1], item[2]): index for index, item in enumerate(VALID_CLASSES)}
    true_joint = [reverse[(int(t), int(p))] for t, p in zip(true_type, true_pattern)]

    type_matrix = confusion_matrix(true_type, pred_type, labels=range(3)).tolist()
    pattern_matrix = confusion_matrix(true_pattern, pred_pattern, labels=range(3)).tolist()
    joint_matrix = confusion_matrix(true_joint, pred_joint, labels=range(6)).tolist()
    analysis = {
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint_epoch": checkpoint["epoch"],
        "design": design,
        "six_class_labels": class_names,
        "six_class_confusion_matrix": joint_matrix,
        "six_class_report": classification_report(true_joint, pred_joint, target_names=class_names, output_dict=True, zero_division=0),
        "type_labels": TYPE_NAMES,
        "type_confusion_matrix": type_matrix,
        "type_report": classification_report(true_type, pred_type, target_names=TYPE_NAMES, output_dict=True, zero_division=0),
        "pattern_labels": PATTERN_NAMES,
        "pattern_confusion_matrix": pattern_matrix,
        "pattern_report": classification_report(true_pattern, pred_pattern, target_names=PATTERN_NAMES, output_dict=True, zero_division=0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clinical_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    write_matrix(args.output_dir / "six_class_confusion_matrix.csv", class_names, joint_matrix)
    write_matrix(args.output_dir / "type_confusion_matrix.csv", TYPE_NAMES, type_matrix)
    write_matrix(args.output_dir / "pattern_confusion_matrix.csv", PATTERN_NAMES, pattern_matrix)
    print(json.dumps({"variant": args.variant, "output": str(args.output_dir), "six_class_accuracy": analysis["six_class_report"]["accuracy"]}))


if __name__ == "__main__":
    main()
