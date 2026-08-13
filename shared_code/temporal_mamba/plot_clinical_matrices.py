#!/usr/bin/env python3
"""Render publication-ready confusion matrices from clinical_analysis.json."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--title',required=True);return p.parse_args()

def draw(matrix, labels, title, path, normalize=False):
    values=np.asarray(matrix,dtype=float)
    if normalize:
        denominator=values.sum(axis=1,keepdims=True);values=np.divide(values,denominator,out=np.zeros_like(values),where=denominator!=0)
        fmt='.2f';vmin,vmax=0,1
    else: fmt='g';vmin,vmax=None,None
    width=max(6.2,0.9*len(labels)+2.5)
    plt.figure(figsize=(width,5.4));sns.heatmap(values,annot=True,fmt=fmt,cmap='Blues',xticklabels=labels,yticklabels=labels,cbar=True,vmin=vmin,vmax=vmax,square=True)
    plt.xlabel('Predicted label');plt.ylabel('True label');plt.title(title);plt.tight_layout();plt.savefig(path,dpi=240,bbox_inches='tight');plt.close()

def main():
    a=parse_args();d=json.loads(a.input.read_text(encoding='utf-8'));a.output_dir.mkdir(parents=True,exist_ok=True)
    for key,label in (('six_class','Six-class'),('type','Strabismus type'),('pattern','Pattern')):
        matrix=d[f'{key}_confusion_matrix'];labels=d[f'{key}_labels']
        draw(matrix,labels,f'{a.title}: {label} confusion matrix',a.output_dir/f'{key}_confusion_matrix.png')
        draw(matrix,labels,f'{a.title}: normalized {label.lower()} confusion matrix',a.output_dir/f'{key}_confusion_matrix_normalized.png',True)
if __name__=='__main__':main()
