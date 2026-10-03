"""Frozen DenseNet gaze-feature baselines requested by reviewer 365."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES=("dvd_no","eso_no","eso_V","exo_A","exo_no","exo_V")
PAIRS=((0,0),(1,0),(1,2),(2,1),(2,0),(2,2))

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def metrics(t,p,yt,yp):
    true=[CLASSES[i] for i in yt.tolist()]; pred=[CLASSES[i] for i in yp.tolist()]
    pr,rc,ff,ss=precision_recall_fscore_support(true,pred,labels=CLASSES,zero_division=0)
    return {"accuracy":float(accuracy_score(true,pred)),"weighted_f1":float(f1_score(true,pred,average="weighted",zero_division=0)),"macro_f1":float(f1_score(true,pred,average="macro",zero_division=0)),"per_class":{CLASSES[i]:{"recall":float(rc[i]),"f1":float(ff[i]),"support":int(ss[i])} for i in range(6)}}

class Model(nn.Module):
    def __init__(self, kind, d=1024, h=192):
        super().__init__(); self.kind=kind; self.proj=nn.Linear(d,h)
        if kind=="bilstm": self.seq=nn.LSTM(h,h//2,batch_first=True,bidirectional=True)
        elif kind=="bigru": self.seq=nn.GRU(h,h//2,batch_first=True,bidirectional=True)
        elif kind=="transformer": self.seq=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,4,h*2,.1,batch_first=True,norm_first=True),2)
        elif kind=="cross_attention": self.seq=nn.MultiheadAttention(h,4,batch_first=True,dropout=.1)
        self.norm=nn.LayerNorm(h); self.drop=nn.Dropout(.1); self.type=nn.Linear(h,3); self.pattern=nn.Linear(h,3)
    def forward(self,x):
        z=self.proj(x)
        if self.kind in ("bilstm","bigru"): z,_=self.seq(z)
        elif self.kind=="transformer": z=self.seq(z)
        elif self.kind=="cross_attention": z,_=self.seq(z,z,z,need_weights=False)
        z=self.drop(self.norm(z).mean(1)); return self.type(z),self.pattern(z)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--kind",choices=("bilstm","bigru","transformer","cross_attention"),required=True); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--split-seed",type=int,default=42); ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args(); seed_all(args.seed)
    x=torch.load(args.cache,map_location="cpu",weights_only=False); n=1202; v=343; features=x["features"].float(); types=x["type_ids"].long(); patterns=x["pattern_ids"].long(); y=torch.tensor([CLASSES.index(next(c for c,p in zip(CLASSES,PAIRS) if p==(int(a),int(b)))) for a,b in zip(types,patterns)])
    # The cache was generated in the fixed seed-42 train/val/test order.
    ds={"train":TensorDataset(features[:n],types[:n],patterns[:n]),"val":TensorDataset(features[n:n+v],types[n:n+v],patterns[n:n+v]),"test":TensorDataset(features[n+v:],types[n+v:],patterns[n+v:])}; gen=torch.Generator().manual_seed(args.seed)
    loaders={k:DataLoader(z,batch_size=args.batch_size,shuffle=k=="train",generator=gen if k=="train" else None) for k,z in ds.items()}; device=torch.device(args.device if torch.cuda.is_available() else "cpu"); model=Model(args.kind,features.shape[-1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.05); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs); args.output_dir.mkdir(parents=True,exist_ok=True); best=-1.; history=[]
    def run(loader,train=False):
        model.train(train); total=0.; tl=[]; pl=[]; ta=[]; pa=[]
        for xb,tb,pb in loader:
            xb,tb,pb=xb.to(device),tb.to(device),pb.to(device)
            if train: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                a,b=model(xb); loss=nn.functional.cross_entropy(a,tb)+nn.functional.cross_entropy(b,pb)
                if train: loss.backward(); opt.step()
            total+=loss.item()*len(xb); ta.append(tb.cpu());pa.append(pb.cpu());tl.append(a.detach().cpu());pl.append(b.detach().cpu())
        ta=torch.cat(ta);pa=torch.cat(pa);tl=torch.cat(tl);pl=torch.cat(pl); pred=torch.stack((tl.argmax(1),pl.argmax(1)),1); truth=torch.stack((ta,pa),1); exact=(pred==truth).all(1); joint=[]; actual=[]
        for i,(a,b) in enumerate(zip(truth.tolist(),pred.tolist())):
            actual.append(PAIRS.index(tuple(a))); joint.append(PAIRS.index(tuple(b)) if tuple(b) in PAIRS else int(torch.tensor([torch.log_softmax(tl[i],0)[t]+torch.log_softmax(pl[i],0)[p] for t,p in PAIRS]).argmax()))
        m=metrics(tl,pl,torch.tensor(actual),torch.tensor(joint)); m.update({"loss":total/len(loader.dataset),"exact_accuracy":float(exact.float().mean())}); return m
    for ep in range(1,args.epochs+1):
        tr=run(loaders["train"],True); va=run(loaders["val"]); history.append({"epoch":ep,"train":tr,"val":va}); print(json.dumps(history[-1]),flush=True)
        if va["exact_accuracy"]>best: best=va["exact_accuracy"]; torch.save({"model":model.state_dict(),"epoch":ep,"val":va},args.output_dir/"best.pt")
        sch.step()
    ck=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False); model.load_state_dict(ck["model"]); te=run(loaders["test"]); (args.output_dir/"history.json").write_text(json.dumps(history,indent=2)); (args.output_dir/"test_results.json").write_text(json.dumps({"kind":args.kind,"seed":args.seed,"split_seed":42,"best_epoch":ck["epoch"],"test":te},indent=2)); print(json.dumps({"kind":args.kind,"seed":args.seed,"test":te},indent=2))
if __name__=="__main__": main()
