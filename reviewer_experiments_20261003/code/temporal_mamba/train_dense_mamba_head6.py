#!/usr/bin/env python3
"""Fast six-class head validation on frozen DenseNet + Clinical Mamba latents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_end_to_end_densenet_mamba import EndToEndDenseNetMamba, VALID_CLASSES
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


CLASS_ORDER = tuple(x[0] for x in VALID_CLASSES)
CLASS_COMBOS = tuple((x[1], x[2]) for x in VALID_CLASSES)
COMBO_TO_CLASS = {combo: i for i, combo in enumerate(CLASS_COMBOS)}


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--pretrained",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True);p.add_argument("--cache",type=Path,required=True);p.add_argument("--epochs",type=int,default=40);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--feature-batch-size",type=int,default=4)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--split-seed",type=int,default=42);p.add_argument("--device",default="cuda:0");p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--weight-decay",type=float,default=0.05);p.add_argument("--amp",action="store_true")
    return p.parse_args()


def encode_latents(model, video):
    b,t,c,h,w=video.shape
    spatial=model.backbone(video.reshape(b*t,c,h,w)).reshape(b,t,-1)
    hidden=model.project(spatial)+model.position
    for block in model.blocks:hidden=block(hidden)
    return model.norm(hidden).mean(1)


@torch.inference_mode()
def make_cache(model,samples,args,device):
    loader=DataLoader(NineGazeImages(samples),batch_size=args.feature_batch_size,shuffle=False,num_workers=0)
    latents,labels,paths=[],[],[]
    for idx,(video,type_ids,pattern_ids,batch_paths) in enumerate(loader,1):
        with torch.autocast(device_type=device.type,enabled=args.amp and device.type=="cuda"):
            latent=encode_latents(model,video.to(device))
        latents.append(latent.float().cpu());labels.extend(COMBO_TO_CLASS[(int(t),int(p))] for t,p in zip(type_ids,pattern_ids));paths.extend(batch_paths)
        if idx%50==0 or idx==len(loader):print(json.dumps({"feature_batches":idx,"total":len(loader)}),flush=True)
    payload={"latents":torch.cat(latents),"labels":torch.tensor(labels),"paths":paths};args.cache.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,args.cache);return payload


def metrics(logits,labels):
    pred=logits.argmax(1).cpu().numpy();truth=labels.cpu().numpy();pc=precision_recall_fscore_support(truth,pred,labels=range(6),zero_division=0)
    return {"accuracy":accuracy_score(truth,pred),"weighted_f1":f1_score(truth,pred,average="weighted",zero_division=0),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0),"confusion_matrix":confusion_matrix(truth,pred,labels=range(6)).tolist(),"per_class":{CLASS_ORDER[i]:{"precision":float(pc[0][i]),"recall":float(pc[1][i]),"f1":float(pc[2][i]),"support":int(pc[3][i])} for i in range(6)}}


def evaluate(head,loader,device):
    head.eval();all_logits,all_labels=[],[]
    with torch.inference_mode():
        for x,y in loader:all_logits.append(head(x.to(device)).cpu());all_labels.append(y)
    return metrics(torch.cat(all_logits),torch.cat(all_labels))


def train_variant(name,smoothing,payload,counts,dual_state,args,device):
    seed_all(args.seed);out=args.output_root/name;out.mkdir(parents=True,exist_ok=True);head=nn.Linear(payload["latents"].shape[1],6)
    with torch.no_grad():
        for k,(t,p) in enumerate(CLASS_COMBOS):
            head.weight[k].copy_(dual_state["type_head.weight"][t]+dual_state["pattern_head.weight"][p]);head.bias[k].copy_(dual_state["type_head.bias"][t]+dual_state["pattern_head.bias"][p])
    head=head.to(device);ntr,nv=counts
    sets={"train":TensorDataset(payload["latents"][:ntr],payload["labels"][:ntr]),"val":TensorDataset(payload["latents"][ntr:ntr+nv],payload["labels"][ntr:ntr+nv]),"test":TensorDataset(payload["latents"][ntr+nv:],payload["labels"][ntr+nv:])}
    loaders={k:DataLoader(v,batch_size=args.batch_size,shuffle=k=="train",generator=torch.Generator().manual_seed(args.seed) if k=="train" else None) for k,v in sets.items()}
    initial={k:evaluate(head,v,device) for k,v in loaders.items()};opt=torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=args.weight_decay);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);criterion=nn.CrossEntropyLoss(label_smoothing=smoothing)
    history=[];best=(-1.,-1.);best_path=out/"best.pt"
    for epoch in range(1,args.epochs+1):
        head.train();loss_sum=0.
        for x,y in loaders["train"]:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);logits=head(x);loss=criterion(logits,y);loss.backward();opt.step();loss_sum+=loss.item()*x.shape[0]
        val=evaluate(head,loaders["val"],device);row={"epoch":epoch,"lr":opt.param_groups[0]["lr"],"train_loss":loss_sum/ntr,"val":val};history.append(row)
        score=(val["accuracy"],val["macro_f1"])
        if score>best:best=score;torch.save({"head6":head.state_dict(),"epoch":epoch,"val":val,"label_smoothing":smoothing},best_path)
        sched.step()
    ckpt=torch.load(best_path,map_location=device,weights_only=False);head.load_state_dict(ckpt["head6"]);test=evaluate(head,loaders["test"],device)
    result={"variant":name,"epochs":args.epochs,"trainable_parameters":sum(p.numel() for p in head.parameters()),"warm_start_initial":initial,"best_epoch":ckpt["epoch"],"best_val":ckpt["val"],"test":test,"label_smoothing":smoothing}
    (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8");(out/"test_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2),flush=True);return result


def main():
    args=parse_args();seed_all(args.seed);device=torch.device(args.device if torch.cuda.is_available() else "cpu");samples=scan(args.data_dir);train,val,test=stratified_split(samples,args.split_seed);ordered=train+val+test
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False);cfg=checkpoint["config"];model=EndToEndDenseNetMamba(args.pretrained,cfg["d_model"],cfg["d_state"],cfg["depth"],0.1);model.load_state_dict(checkpoint["model"]);model.to(device).eval().requires_grad_(False)
    if args.cache.is_file():
        payload=torch.load(args.cache,map_location="cpu",weights_only=False)
        if payload["paths"] != [str(x[0]) for x in ordered]:raise RuntimeError("Latent cache does not match split")
    else:payload=make_cache(model,ordered,args,device)
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/"split_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(("split","class","path"));[w.writerows((s,x[3],str(x[0])) for x in g) for s,g in (("train",train),("val",val),("test",test))]
    results=[train_variant("head6_ce",0.0,payload,(len(train),len(val)),checkpoint["model"],args,device),train_variant("head6_label_smoothing_01",0.1,payload,(len(train),len(val)),checkpoint["model"],args,device)]
    (args.output_root/"comparison.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":main()
