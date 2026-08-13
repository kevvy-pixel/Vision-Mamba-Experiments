#!/usr/bin/env python3
"""Fast proof-of-concept: frozen MobileNetV3-Small + Clinical Mamba + attention."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small

from pure_torch_mamba import MambaResidualBlock
from train_lightweight_cnn_mamba_attention import GatedAttentionPool, VALID_CLASSES
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--feature-batch-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    return p.parse_args()


def load_backbone(weights, device):
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.classifier = nn.Identity()
    return model.eval().requires_grad_(False).to(device)


@torch.inference_mode()
def extract(samples, args, device):
    loader = DataLoader(NineGazeImages(samples), batch_size=args.feature_batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_backbone(args.weights, device)
    features, types, patterns, paths = [], [], [], []
    started = time.time()
    for index, (video, type_ids, pattern_ids, batch_paths) in enumerate(loader, 1):
        b, t, c, h, w = video.shape
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            encoded = model(video.reshape(b * t, c, h, w).to(device)).reshape(b, t, -1)
        features.append(encoded.float().cpu().half()); types.append(type_ids); patterns.append(pattern_ids); paths.extend(batch_paths)
        if index % 25 == 0 or index == len(loader):
            print(json.dumps({"feature_batches": index, "total": len(loader), "elapsed_seconds": round(time.time()-started, 1)}), flush=True)
    payload = {"features": torch.cat(features), "type_ids": torch.cat(types), "pattern_ids": torch.cat(patterns), "paths": paths, "backbone": "MobileNetV3-Small ImageNet1K V1", "feature_dim": 576}
    args.cache.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.cache)
    return payload


class FrozenClassifier(nn.Module):
    def __init__(self, mode, feature_dim, d_model, d_state, depth, dropout):
        super().__init__(); self.mode = mode
        self.project = nn.Linear(feature_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, 9, d_model)); nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)] if mode != "mean" else [])
        self.norm, self.dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.attention = GatedAttentionPool(d_model, dropout) if mode == "mamba_attention" else None
        self.type_head, self.pattern_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3)

    def forward(self, features, return_attention=False):
        hidden = self.project(features.float())
        if self.mode != "mean":
            hidden = hidden + self.position
            for block in self.blocks: hidden = block(hidden)
        hidden = self.norm(hidden)
        if self.attention is None: pooled, weights = self.dropout(hidden.mean(1)), None
        else: pooled, weights = self.attention(hidden)
        outputs = self.type_head(pooled), self.pattern_head(pooled)
        return (*outputs, weights) if return_attention else outputs


def epoch(model, loader, device, optimizer=None, collect=False):
    training = optimizer is not None; model.train(training)
    loss_sum, tls, pls, tlog, plog, attention = 0.0, [], [], [], [], []
    for features, type_ids, pattern_ids in loader:
        features, type_ids, pattern_ids = features.to(device), type_ids.to(device), pattern_ids.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            type_logits, pattern_logits, weights = model(features, return_attention=True)
            loss = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(pattern_logits, pattern_ids)
            if training: loss.backward(); optimizer.step()
        loss_sum += loss.item()*features.shape[0]; tls.append(type_ids.cpu()); pls.append(pattern_ids.cpu())
        tlog.append(type_logits.detach().cpu()); plog.append(pattern_logits.detach().cpu())
        if weights is not None: attention.append(weights.detach().cpu())
    type_ids, pattern_ids, type_logits, pattern_logits = torch.cat(tls), torch.cat(pls), torch.cat(tlog), torch.cat(plog)
    type_pred, pattern_pred = type_logits.argmax(1), pattern_logits.argmax(1)
    truth = [f"{a.item()}_{b.item()}" for a,b in zip(type_ids,pattern_ids)]; pred = [f"{a.item()}_{b.item()}" for a,b in zip(type_pred,pattern_pred)]
    result = {"loss":loss_sum/len(loader.dataset), "exact_accuracy":((type_pred==type_ids)&(pattern_pred==pattern_ids)).float().mean().item(), "type_accuracy":(type_pred==type_ids).float().mean().item(), "pattern_accuracy":(pattern_pred==pattern_ids).float().mean().item(), "joint_weighted_f1":f1_score(truth,pred,average="weighted",zero_division=0), "joint_macro_f1":f1_score(truth,pred,average="macro",zero_division=0)}
    if collect:
        pairs=torch.tensor([[x[1],x[2]] for x in VALID_CLASSES]); scores=torch.log_softmax(type_logits,1)[:,pairs[:,0]]+torch.log_softmax(pattern_logits,1)[:,pairs[:,1]]
        joint_pred=scores.argmax(1).tolist(); reverse={(x[1],x[2]):i for i,x in enumerate(VALID_CLASSES)}; joint_true=[reverse[(int(a),int(b))] for a,b in zip(type_ids,pattern_ids)]; names=[x[0] for x in VALID_CLASSES]
        result.update({"six_class_labels":names,"six_class_confusion_matrix":confusion_matrix(joint_true,joint_pred,labels=range(6)).tolist(),"six_class_report":classification_report(joint_true,joint_pred,target_names=names,output_dict=True,zero_division=0)})
        if attention:
            weights=torch.cat(attention); result["attention_mean_clinical_order"]=weights.mean(0).tolist(); result["attention_std_clinical_order"]=weights.std(0,unbiased=False).tolist()
    return result


def main():
    args=args_parser(); seed_all(args.seed); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples=scan(args.data_dir); train,val,test=stratified_split(samples,args.seed); ordered=train+val+test
    if args.cache.is_file():
        payload=torch.load(args.cache,map_location="cpu",weights_only=False)
        if payload["paths"] != [str(x[0]) for x in ordered]: raise RuntimeError("Cache does not match split")
    else: payload=extract(ordered,args,device)
    n_train,n_val=len(train),len(val); f,t,p=payload["features"],payload["type_ids"],payload["pattern_ids"]
    datasets={"train":TensorDataset(f[:n_train],t[:n_train],p[:n_train]),"val":TensorDataset(f[n_train:n_train+n_val],t[n_train:n_train+n_val],p[n_train:n_train+n_val]),"test":TensorDataset(f[n_train+n_val:],t[n_train+n_val:],p[n_train+n_val:])}
    for mode in ("mean","mamba_mean","mamba_attention"):
        seed_all(args.seed); out=args.output_root/(mode+f"_seed{args.seed}"); out.mkdir(parents=True,exist_ok=True)
        loaders={name:DataLoader(ds,batch_size=args.batch_size,shuffle=name=="train",generator=torch.Generator().manual_seed(args.seed) if name=="train" else None) for name,ds in datasets.items()}
        model=FrozenClassifier(mode,payload["feature_dim"],args.d_model,args.d_state,args.depth,args.dropout).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
        params=sum(x.numel() for x in model.parameters()); config={"mode":mode,"seed":args.seed,"epochs":args.epochs,"backbone":payload["backbone"],"backbone_frozen":True,"d_model":args.d_model,"d_state":args.d_state,"depth":args.depth,"parameters":params,"gaze_order":"5-2-3-6-9-8-7-4-1"}
        (out/"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        with (out/"split_manifest.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle);writer.writerow(("split","class","path"));[writer.writerows((split,x[3],str(x[0])) for x in group) for split,group in (("train",train),("val",val),("test",test))]
        history=[];best=-1.;best_path=out/"best.pt"
        for index in range(1,args.epochs+1):
            tm=epoch(model,loaders["train"],device,optimizer);vm=epoch(model,loaders["val"],device);row={"epoch":index,"lr":optimizer.param_groups[0]["lr"],"train":tm,"val":vm};history.append(row);print(json.dumps({"mode":mode,**row}),flush=True)
            if vm["exact_accuracy"]>best: best=vm["exact_accuracy"];torch.save({"model":model.state_dict(),"epoch":index,"val":vm,"config":config},best_path)
            scheduler.step()
        checkpoint=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"]);test_metrics=epoch(model,loaders["test"],device,collect=True)
        result={"best_epoch":checkpoint["epoch"],"best_val":checkpoint["val"],"test":test_metrics,"config":config};(out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8");(out/"test_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps({"completed":mode,"test":test_metrics}),flush=True)


if __name__ == "__main__": main()
