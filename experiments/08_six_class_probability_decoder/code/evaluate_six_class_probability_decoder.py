#!/usr/bin/env python3
"""Evaluate constrained six-class probability decoding on the best DenseNet-Mamba."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from train_end_to_end_densenet_mamba import EndToEndDenseNetMamba, VALID_CLASSES
from train_temporal_ablation import NineGazeImages, scan, seed_all, stratified_split


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True);p.add_argument("--pretrained",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0");p.add_argument("--amp",action="store_true")
    return p.parse_args()


def scores(y_true,y_pred):
    return {"accuracy":accuracy_score(y_true,y_pred),"weighted_f1":f1_score(y_true,y_pred,average="weighted",zero_division=0),"macro_f1":f1_score(y_true,y_pred,average="macro",zero_division=0)}


def main():
    args=parse_args();seed_all(args.seed);device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False);cfg=payload["config"]
    model=EndToEndDenseNetMamba(args.pretrained,cfg["d_model"],cfg["d_state"],cfg["depth"],0.1);model.load_state_dict(payload["model"]);model.to(device).eval()
    _,_,test=stratified_split(scan(args.data_dir),args.seed);loader=DataLoader(NineGazeImages(test),batch_size=args.batch_size,shuffle=False,num_workers=0)
    names=[x[0] for x in VALID_CLASSES];combos=[(x[1],x[2]) for x in VALID_CLASSES];reverse={combo:i for i,combo in enumerate(combos)}
    records=[];invalid=0
    with torch.inference_mode():
        for video,type_ids,pattern_ids,paths in loader:
            with torch.autocast(device_type=device.type,enabled=args.amp and device.type=="cuda"):
                tl,pl=model(video.to(device));pt=torch.softmax(tl,1);pp=torch.softmax(pl,1);joint=torch.stack([pt[:,t]*pp[:,p] for t,p in combos],1);joint=joint/joint.sum(1,keepdim=True)
            tl,pl,pt,pp,joint=tl.float().cpu(),pl.float().cpu(),pt.float().cpu(),pp.float().cpu(),joint.float().cpu()
            for i,path in enumerate(paths):
                true=reverse[(int(type_ids[i]),int(pattern_ids[i]))];ind_combo=(int(tl[i].argmax()),int(pl[i].argmax()));ind=reverse.get(ind_combo,-1);constrained=int(joint[i].argmax());invalid+=int(ind<0)
                records.append({"path":path,"true":true,"independent":ind,"constrained":constrained,"probabilities":joint[i].tolist(),"p_type":pt[i].tolist(),"p_pattern":pp[i].tolist()})
    y=[r["true"] for r in records];ind=[r["independent"] for r in records];con=[r["constrained"] for r in records]
    probs=np.asarray([r["probabilities"] for r in records]);onehot=np.eye(6)[y]
    summary={"checkpoint":str(args.checkpoint),"best_epoch":payload.get("epoch"),"n_samples":len(records),"independent_dual_argmax":scores(y,ind),"constrained_joint_probability":scores(y,con),"changed_predictions":sum(a!=b for a,b in zip(ind,con)),"invalid_independent_combinations":invalid,"mean_top1_confidence":float(probs.max(1).mean()),"multiclass_nll":float(-np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1)).mean()),"multiclass_brier":float(np.square(probs-onehot).sum(1).mean()),"class_order":names}
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");np.save(args.output_dir/"six_class_probabilities.npy",probs.astype(np.float32))
    fields=["path","true_class","independent_class","constrained_class","correct","top1_conf"]+names
    with (args.output_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            row={"path":r["path"],"true_class":names[r["true"]],"independent_class":names[r["independent"]] if r["independent"]>=0 else "INVALID","constrained_class":names[r["constrained"]],"correct":int(r["true"]==r["constrained"]),"top1_conf":max(r["probabilities"])};row.update({name:r["probabilities"][i] for i,name in enumerate(names)});w.writerow(row)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
