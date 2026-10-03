"""Same-protocol Adapted CI-GNN using the authors' graph network classes.

The supplied pic2npy starts from full-face photographs, while the current
dataset contains composed eye crops.  We preserve its geometric variables and
the authors' GCN/classifier, replacing only full-face MediaPipe localization
with a deterministic eye-crop locator.  Results must be named Adapted CI-GNN.
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys, types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES=["dvd_no","eso_no","eso_V","exo_A","exo_no","exo_V"]
LEGAL=[(2,2),(0,2),(0,1),(1,0),(1,2),(1,1)] # type: eso/exo/dvd; pattern: A/V/no


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    counts=Counter(r["split"] for r in rows)
    if counts!=Counter({"train":1202,"val":343,"test":171}): raise ValueError(counts)
    return rows


def centroid(mask,weight,fallback):
    v=mask.astype(np.float32)*weight.astype(np.float32); s=float(v.sum())
    if s<1e-6:return fallback
    yy,xx=np.indices(mask.shape); return float((xx*v).sum()/s),float((yy*v).sum()/s)


def locate_eye(tile,x0,x1):
    h,w=tile.shape[:2]; xa,xb=int(x0*w),int(x1*w); ya,yb=int(.16*h),int(.84*h); roi=tile[ya:yb,xa:xb]
    gray=cv2.GaussianBlur(cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY),(7,7),0); yy,xx=np.indices(gray.shape)
    cx,cy=.5*gray.shape[1],.52*gray.shape[0]; q=float(np.percentile(gray,28)); dark=np.clip((q-gray)/max(q,1),0,1)
    prior=np.exp(-(((xx-cx)/(.48*gray.shape[1]))**2+((yy-cy)/(.55*gray.shape[0]))**2))
    ix,iy=centroid(gray<q,dark*prior,(cx,cy)); radius=max(5.,.115*min(h,w))
    b,g,r=cv2.split(roi.astype(np.float32)); chroma=np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])
    local=(xx-ix)**2+(yy-iy)**2<(1.25*radius)**2; bright=(gray>215)&(chroma<48)&local
    lx,ly=centroid(bright,np.maximum(gray-180,1),(ix,iy))
    return np.array([(ix+xa)/w,(iy+ya)/h,(lx+xa)/w,(ly+ya)/h],np.float32),radius


def gaze_features(tile):
    left,lr=locate_eye(tile,.03,.49); right,rr=locate_eye(tile,.51,.97)
    li,ll=left[:2],left[2:]; ri,rl=right[:2],right[2:]
    # pic2npy stores centers relative to the cropped-eye-region center.
    ex=np.array([li[0]-.5,ri[0]-.5,ll[0]-.5,rl[0]-.5,ri[1]-.5],np.float32)
    orb_l=np.array([.26,.50],np.float32); orb_r=np.array([.74,.50],np.float32)
    dl,dr=li-orb_l,ri-orb_r
    left_angle=math.atan2(abs(float(dl[1])),float(dl[0]))/math.pi
    right_angle=math.atan2(abs(float(dr[1])),float(dr[0]))/math.pi
    return ex,np.array([left_angle,right_angle],np.float32),(li,ri,ll,rl,lr,rr)


def extract_one(path,overlay_path=None):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise RuntimeError(path)
    h,w=image.shape[:2]; tiles=[image[r*h//3:(r+1)*h//3,c*w//3:(c+1)*w//3] for r in range(3) for c in range(3)]
    ex=[]; angles=[]; audit=[]
    for tile in tiles:
        e,a,d=gaze_features(tile); ex.append(e); angles.append(a)
        if overlay_path is not None:
            z=tile.copy(); hh,ww=z.shape[:2]
            for p,col,rad in [(d[0],(0,255,0),d[4]),(d[1],(0,255,0),d[5]),(d[2],(0,0,255),3),(d[3],(0,0,255),3)]: cv2.circle(z,(int(p[0]*ww),int(p[1]*hh)),int(rad),col,2)
            audit.append(z)
    angles=np.asarray(angles,np.float32); std=np.array([.75,.5,.25],np.float32)
    delta=np.asarray([angles[i,1]-(angles[i,0]-std[i%3]) for i in range(9)],np.float32)
    shared=np.array([delta[1],delta[6],delta[7],delta[8],delta[1]-delta[7],delta[2]-delta[8],delta[0]-delta[6]],np.float32)
    av=np.stack([np.concatenate([shared,[angles[i,1],angles[i,0]]]).astype(np.float32) for i in range(9)])
    if overlay_path is not None:
        hh=min(x.shape[0] for x in audit); ww=min(x.shape[1] for x in audit)
        canvas=np.vstack([np.hstack([cv2.resize(audit[3*r+c],(ww,hh)) for c in range(3)]) for r in range(3)])
        overlay_path.parent.mkdir(parents=True,exist_ok=True); cv2.imencode(".jpg",canvas)[1].tofile(str(overlay_path))
    return np.asarray(ex,np.float32),av


def cache_features(rows,path,overlay_dir):
    if path.exists():
        z=np.load(path,allow_pickle=True); return z["ex"],z["av"],z["joint"],z["split"],z["paths"]
    exs=[]; avs=[]; ys=[]; splits=[]; paths=[]
    for i,r in enumerate(rows):
        overlay=overlay_dir/f"{i:04d}_{r['class']}.jpg" if i<12 else None; e,a=extract_one(r["path"],overlay)
        exs.append(e); avs.append(a); ys.append(CLASSES.index(r["class"])); splits.append(r["split"]); paths.append(r["path"])
        if (i+1)%100==0 or i+1==len(rows): print(f"features {i+1}/{len(rows)}",flush=True)
    ex=np.asarray(exs,np.float32); av=np.asarray(avs,np.float32); y=np.asarray(ys,np.int64); split=np.asarray(splits); paths=np.asarray(paths)
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,ex=ex,av=av,joint=y,split=split,paths=paths)
    return ex,av,y,split,paths


def load_official_classes(source_root):
    for name in ["thop","torchsummary","torchinfo"]:
        m=types.ModuleType(name); m.profile=lambda model,inputs=():(0,sum(p.numel() for p in model.parameters())); m.summary=lambda *a,**k:None; sys.modules.setdefault(name,m)
    gnn=Path(source_root)/"GNN"; sys.path.insert(0,str(gnn))
    from models.GCN import GNN_single
    from models.NormClassifier import LinearClsHead
    return GNN_single,LinearClsHead


class OfficialTask(nn.Module):
    def __init__(self,GNN,Head,num_head):
        super().__init__(); self.gnn=GNN(in_channels=9,hidden_channels=(2048,2048,2048),dropout=None); self.head=Head(num_classes=3,feat_dim=2048,use_effect=True,num_head=num_head,tau=8.,alpha=.5,gamma=.0625)
    def forward(self,x):
        z=self.gnn(x); logits,_=self.head(z,torch.zeros(len(x),dtype=torch.long,device=x.device),None); return logits


def make_loaders(x,y,split,batch,seed):
    out={}; gen=torch.Generator().manual_seed(seed)
    for s in ["train","val","test"]:
        m=split==s; out[s]=DataLoader(TensorDataset(torch.from_numpy(x[m]),torch.from_numpy(y[m])),batch_size=batch,shuffle=s=="train",generator=gen if s=="train" else None,num_workers=0)
    return out


@torch.no_grad()
def task_eval(model,loader,device):
    model.eval(); yy=[]; pp=[]; logits=[]; loss=0.
    for x,y in loader:
        x,y=x.to(device),y.to(device); z=model(x); loss+=nn.functional.cross_entropy(z,y).item()*len(y); yy.extend(y.cpu().tolist()); pp.extend(z.argmax(1).cpu().tolist()); logits.append(z.cpu())
    return {"loss":loss/len(yy),"accuracy":float(accuracy_score(yy,pp)),"weighted_f1":float(f1_score(yy,pp,average="weighted",zero_division=0)),"macro_f1":float(f1_score(yy,pp,average="macro",zero_division=0))},torch.cat(logits),np.asarray(yy)


def train_task(name,x,y,split,GNN,Head,out,args,device):
    loaders=make_loaders(x,y,split,args.batch_size,args.seed); model=OfficialTask(GNN,Head,8 if name=="pattern" else 4).to(device)
    lr=args.pattern_lr if name=="pattern" else args.type_lr; opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=.001)
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[120,160],gamma=.1); best=-1.; stale=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.
        for xb,yb in loaders["train"]:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
        sched.step(); val,_,_=task_eval(model,loaders["val"],device); rec={"epoch":epoch,"train_loss":total/len(loaders['train'].dataset),"lr":opt.param_groups[0]['lr'],**{f"val_{k}":v for k,v in val.items()}}; history.append(rec)
        if val["accuracy"]>best+1e-9:
            best=val["accuracy"]; stale=0; torch.save({"model":model.state_dict(),"epoch":epoch,"val":val},out/f"best_{name}.pt")
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f"{name} epoch={epoch:03d} val_acc={val['accuracy']:.4f} val_wf1={val['weighted_f1']:.4f} best={best:.4f}",flush=True)
        if epoch>=args.min_epochs and stale>=args.patience: print(f"{name} early stop {epoch}",flush=True); break
    ck=torch.load(out/f"best_{name}.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"]); test,logits,truth=task_eval(model,loaders["test"],device)
    (out/f"history_{name}.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); return model,ck,test,logits,truth


def joint_metrics(y,p):
    pr,rc,f,s=precision_recall_fscore_support(y,p,labels=range(6),zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"confusion_matrix":confusion_matrix(y,p,labels=range(6)).tolist(),"per_class":{CLASSES[i]:{"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f[i]),"support":int(s[i])} for i in range(6)}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--source-root",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=300); ap.add_argument("--min-epochs",type=int,default=170); ap.add_argument("--patience",type=int,default=60); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--type-lr",type=float,default=.05); ap.add_argument("--pattern-lr",type=float,default=.01); ap.add_argument("--extract-only",action="store_true"); args=ap.parse_args()
    seed_all(args.seed); args.output_dir.mkdir(parents=True,exist_ok=True); rows=read_manifest(args.manifest); ex,av,joint,split,paths=cache_features(rows,args.output_dir/"pic2npy_adapted_features.npz",args.output_dir/"landmark_overlays")
    if args.extract_only: print({"ex":ex.shape,"av":av.shape}); return
    pairs=np.asarray([LEGAL[i] for i in joint]); type_y,patt_y=pairs[:,0],pairs[:,1]; GNN,Head=load_official_classes(args.source_root); device=torch.device("cuda")
    type_model,type_ck,type_test,type_logits,_=train_task("type",ex,type_y,split,GNN,Head,args.output_dir,args,device)
    patt_model,patt_ck,patt_test,patt_logits,_=train_task("pattern",av,patt_y,split,GNN,Head,args.output_dir,args,device)
    lt=type_logits.log_softmax(1); lp=patt_logits.log_softmax(1); scores=torch.stack([lt[:,t]+lp[:,p] for t,p in LEGAL],1); pred=scores.argmax(1).numpy(); truth=joint[split=="test"]; result=joint_metrics(truth,pred)
    result.update({"type_task":type_test,"pattern_task":patt_test,"best_type_epoch":type_ck["epoch"],"best_pattern_epoch":patt_ck["epoch"],"method_name":"Adapted CI-GNN","input_adapter":"pic2npy geometric semantics with deterministic eye-crop localization","official_network_classes":True,"test_samples":171})
    (args.output_dir/"test_results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    cfg={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},"split_counts":dict(Counter(split)),"ex_shape":list(ex.shape),"av_shape":list(av.shape),"limitation":"MediaPipe full-face localization is incompatible with composed eye crops; only localization is adapted."}; (args.output_dir/"config.json").write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
    with (args.output_dir/"test_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","true_class","predicted_class"])
        for path,t,p in zip(paths[split=="test"],truth,pred): w.writerow([path,CLASSES[int(t)],CLASSES[int(p)]])
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
