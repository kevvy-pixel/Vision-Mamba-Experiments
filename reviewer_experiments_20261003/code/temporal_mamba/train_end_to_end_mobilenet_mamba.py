#!/usr/bin/env python3
"""Two-stage end-to-end MobileNetV3-Small + Clinical Mamba + mean pooling."""

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
from torchvision.models import mobilenet_v3_small

from pure_torch_mamba import MambaResidualBlock
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


VALID_CLASSES = (
    ("dvd_no", 0, 0), ("eso_no", 1, 0), ("eso_V", 1, 2),
    ("exo_A", 2, 1), ("exo_no", 2, 0), ("exo_V", 2, 2),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--backbone-weights", type=Path, required=True)
    p.add_argument("--head-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backbone-lr", type=float, default=2e-5)
    p.add_argument("--temporal-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


class EndToEndMobileNetMamba(nn.Module):
    def __init__(self, weights, d_model, d_state, depth, dropout):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=None)
        self.backbone.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.backbone.classifier = nn.Identity()
        self.project = nn.Linear(576, d_model)
        self.position = nn.Parameter(torch.zeros(1, 9, d_model)); nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList([MambaResidualBlock(d_model, d_state, dropout) for _ in range(depth)])
        self.norm, self.dropout = nn.LayerNorm(d_model), nn.Dropout(dropout)
        self.type_head, self.pattern_head = nn.Linear(d_model, 3), nn.Linear(d_model, 3)

    def forward(self, video):
        b, t, c, h, w = video.shape
        features = self.backbone(video.reshape(b*t, c, h, w)).reshape(b, t, -1)
        hidden = self.project(features) + self.position
        for block in self.blocks: hidden = block(hidden)
        pooled = self.dropout(self.norm(hidden).mean(1))
        return self.type_head(pooled), self.pattern_head(pooled)


def load_head(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    non_backbone_missing = [x for x in incompatible.missing_keys if not x.startswith("backbone.")]
    if unexpected or non_backbone_missing:
        raise RuntimeError(f"Head checkpoint mismatch: unexpected={unexpected}, missing={non_backbone_missing}")
    return checkpoint.get("epoch"), checkpoint.get("val")


def run_epoch(model, loader, device, amp, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None; model.train(training)
    loss_sum, tls, pls, tlog, plog = 0.0, [], [], [], []
    for video, type_ids, pattern_ids, _ in loader:
        video, type_ids, pattern_ids = video.to(device), type_ids.to(device), pattern_ids.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            type_logits, pattern_logits = model(video)
            loss = F.cross_entropy(type_logits, type_ids) + F.cross_entropy(pattern_logits, pattern_ids)
        if training:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        loss_sum += loss.item()*video.shape[0]; tls.append(type_ids.cpu()); pls.append(pattern_ids.cpu()); tlog.append(type_logits.detach().float().cpu()); plog.append(pattern_logits.detach().float().cpu())
    type_ids, pattern_ids, type_logits, pattern_logits = torch.cat(tls), torch.cat(pls), torch.cat(tlog), torch.cat(plog)
    type_pred, pattern_pred = type_logits.argmax(1), pattern_logits.argmax(1)
    truth=[f"{a.item()}_{b.item()}" for a,b in zip(type_ids,pattern_ids)]; pred=[f"{a.item()}_{b.item()}" for a,b in zip(type_pred,pattern_pred)]
    result={"loss":loss_sum/len(loader.dataset),"exact_accuracy":((type_pred==type_ids)&(pattern_pred==pattern_ids)).float().mean().item(),"type_accuracy":(type_pred==type_ids).float().mean().item(),"pattern_accuracy":(pattern_pred==pattern_ids).float().mean().item(),"joint_weighted_f1":f1_score(truth,pred,average="weighted",zero_division=0),"joint_macro_f1":f1_score(truth,pred,average="macro",zero_division=0)}
    if collect:
        pairs=torch.tensor([[x[1],x[2]] for x in VALID_CLASSES]); scores=torch.log_softmax(type_logits,1)[:,pairs[:,0]]+torch.log_softmax(pattern_logits,1)[:,pairs[:,1]]
        joint_pred=scores.argmax(1).tolist(); reverse={(x[1],x[2]):i for i,x in enumerate(VALID_CLASSES)}; joint_true=[reverse[(int(a),int(b))] for a,b in zip(type_ids,pattern_ids)]; names=[x[0] for x in VALID_CLASSES]
        result.update({"six_class_labels":names,"six_class_confusion_matrix":confusion_matrix(joint_true,joint_pred,labels=range(6)).tolist(),"six_class_report":classification_report(joint_true,joint_pred,target_names=names,output_dict=True,zero_division=0),"type_confusion_matrix":confusion_matrix(type_ids,type_pred,labels=range(3)).tolist(),"pattern_confusion_matrix":confusion_matrix(pattern_ids,pattern_pred,labels=range(3)).tolist()})
    return result


def main():
    args=parse_args(); seed_all(args.seed); device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples=scan(args.data_dir); train,val,test=stratified_split(samples,args.split_seed)
    loaders={"train":DataLoader(NineGazeImages(train),batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,generator=torch.Generator().manual_seed(args.seed)),"val":DataLoader(NineGazeImages(val),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers),"test":DataLoader(NineGazeImages(test),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)}
    model=EndToEndMobileNetMamba(args.backbone_weights,args.d_model,args.d_state,args.depth,args.dropout); head_epoch,head_val=load_head(model,args.head_checkpoint); model=model.to(device)
    temporal=[p for name,p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer=torch.optim.AdamW([{"params":model.backbone.parameters(),"lr":args.backbone_lr},{"params":temporal,"lr":args.temporal_lr}],weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs); scaler=torch.amp.GradScaler("cuda",enabled=args.amp and device.type=="cuda")
    params={"total":sum(p.numel() for p in model.parameters()),"backbone":sum(p.numel() for p in model.backbone.parameters()),"temporal_and_heads":sum(p.numel() for p in temporal)}
    if args.dry_run:
        video,type_ids,pattern_ids,_=next(iter(loaders["train"]));video=video.to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=args.amp and device.type=="cuda"):
            outputs=model(video);loss=F.cross_entropy(outputs[0],type_ids.to(device))+F.cross_entropy(outputs[1],pattern_ids.to(device))
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();print(json.dumps({"dry_run":True,"loss":float(loss.detach()),"parameters":params,"head_checkpoint_epoch":head_epoch,"cuda_max_memory_mb":torch.cuda.max_memory_allocated()/1024**2 if device.type=="cuda" else 0}));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/"split_manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle);writer.writerow(("split","class","path"));[writer.writerows((split,x[3],str(x[0])) for x in group) for split,group in (("train",train),("val",val),("test",test))]
    config={"seed":args.seed,"split_seed":args.split_seed,"epochs":args.epochs,"batch_size":args.batch_size,"num_workers":args.num_workers,"amp":args.amp,"backbone":"MobileNetV3-Small ImageNet1K V1","stage1_head_checkpoint":str(args.head_checkpoint),"stage1_best_epoch":head_epoch,"stage1_val":head_val,"backbone_lr":args.backbone_lr,"temporal_lr":args.temporal_lr,"d_model":args.d_model,"d_state":args.d_state,"depth":args.depth,"pooling":"mean","gaze_order":"5-2-3-6-9-8-7-4-1","parameters":params,"joint_fine_tuning":"all MobileNet and Clinical Mamba parameters"}
    (args.output_dir/"model_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    history=[];best=-1.;best_path=args.output_dir/"best.pt"
    for index in range(1,args.epochs+1):
        tm=run_epoch(model,loaders["train"],device,args.amp,optimizer,scaler);vm=run_epoch(model,loaders["val"],device,args.amp);row={"epoch":index,"backbone_lr":optimizer.param_groups[0]["lr"],"temporal_lr":optimizer.param_groups[1]["lr"],"train":tm,"val":vm};history.append(row);(args.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8");print(json.dumps(row),flush=True)
        if vm["exact_accuracy"]>best:best=vm["exact_accuracy"];torch.save({"model":model.state_dict(),"epoch":index,"val":vm,"config":config},best_path)
        scheduler.step()
    checkpoint=torch.load(best_path,map_location=device,weights_only=False);model.load_state_dict(checkpoint["model"]);test_metrics=run_epoch(model,loaders["test"],device,args.amp,collect=True);result={"best_epoch":checkpoint["epoch"],"best_val":checkpoint["val"],"test":test_metrics,"config":config};(args.output_dir/"test_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
