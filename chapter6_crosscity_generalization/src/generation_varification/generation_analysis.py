#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_prediction.py

Performs inference using a fine‑tuned DualBranchGNN, exports predictions for vector points,
validates generalization by extracting and visualizing 64‑dimensional fused features across cities.

Steps:
 1) Preprocess features (LULC encoding, numeric normalization).
 2) Construct k‑hop spatial graph with Gaussian + learned noise weights.
 3) Load trained DualBranchGNN model and EdgeWeight layer.
 4) Predict meandBA and export CSVs for vectorization in GIS.
 5) Visualize spatial scatter maps of predictions.
 6) Extract 64‑dim fused node features + predictions into a combined dataset.
 7) Validate generalization via t‑SNE and UMAP projections of fused features, colored by city.

Usage:
    python model_prediction.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.manifold import TSNE
import umap
import colorsys

# ===== GLOBAL SETTINGS =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 30.0  # Gaussian kernel width
K_HOP = 3     # Grid hop radius\SAVE_DIR = "result"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# 1. FEATURE PREPROCESSING
# -------------------------
def preprocess_features(df):
    df['meandBA'] = pd.to_numeric(df.get('meandBA',0), errors='coerce').fillna(0.0)
    cat_cols = ["located_lulc","local_dominant_lulc","global_dominant_lulc"]
    categories = [
        '11100','11210','11220','11230','11240','12210','14100','12100',
        '12220','23000','12300','12230','14200','13300','31000','21000',
        '12400','13100','13400','40000','50000','32000','33000','11300'
    ]
    for col in cat_cols:
        if col in df:
            df[col] = pd.Categorical(df[col], categories=categories)
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col,axis=1), dummies], axis=1)
    local_feats = [c for c in df if ('r30' in c or 'r60' in c or 'r120' in c or c.startswith('local_'))]
    global_feats = [c for c in df if ('r500' in c or 'r1000' in c or c.startswith('global_'))]
    df[local_feats] = df[local_feats].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    df[global_feats] = df[global_feats].apply(pd.to_numeric,errors='coerce').fillna(0.0)
    if 'domain' not in df: df['domain']=1
    return df, local_feats, global_feats

# -------------------------
# 2. SMOOTH MAPPING
# -------------------------
def smooth_mapping(x, k=1.0, m=1.0):
    lower, upper = 0.1, 2.0
    return lower + (upper-lower)*torch.sigmoid(k*(x-m))

# -------------------------
# 3. GRAPH CONSTRUCTION
# -------------------------
def construct_graph(df, local_feats, global_feats, save_path=None):
    if save_path and os.path.exists(save_path): return torch.load(save_path)
    df = df.reset_index(drop=True)
    N = len(df)
    x_local = torch.tensor(df[local_feats].values, dtype=torch.float32)
    x_global = torch.tensor(df[global_feats].values, dtype=torch.float32)
    y = torch.tensor(df['meandBA'].values, dtype=torch.float32)
    set_ids = torch.tensor([3]*N, dtype=torch.long)
    domain = torch.tensor(df['domain'].values, dtype=torch.long)
    coords = df[['row','col']].astype(int).values
    coord_map = {tuple(rc):i for i,rc in enumerate(coords)}
    edges, wts = [], []
    for i,(r,c) in enumerate(coords):
        for dx in range(-K_HOP,K_HOP+1):
            for dy in range(-K_HOP,K_HOP+1):
                if dx==0 and dy==0: continue
                nb = (r+dx,c+dy)
                if nb in coord_map:
                    j = coord_map[nb]
                    edges.append([i,j])
                    dist=30.0*np.hypot(dx,dy)
                    gw=np.exp(-dist**2/(2*sigma**2))
                    w=smooth_mapping(torch.tensor(gw)).item()
                    wts.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(wts, dtype=torch.float32)
    data = Data(x_local=x_local,x_global=x_global,y=y,
                edge_index=edge_index,edge_weight=edge_weight)
    data.set_ids=set_ids; data.domain=domain; data.coord=torch.tensor(coords,dtype=torch.float32)
    if save_path: torch.save(data,save_path)
    return data

# -------------------------
# 4. EDGE WEIGHT LAYER
# -------------------------
class EdgeWeightLayer(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(10,10); nn.init.eye_(self.fc1.weight)
        self.fc2=nn.Linear(10,1); init_w=torch.tensor([[1.0,0.7,0.5,0.3,-0.5,-0.8,-0.3,-0.5,0.2,-0.2]])
        with torch.no_grad(): self.fc2.weight.copy_(init_w)
    def forward(self,x): return self.fc2(torch.elu(self.fc1(x))).squeeze(1)

# -------------------------
# 5. DUAL BRANCH GNN MODEL
# -------------------------
class DualBranchGNN(nn.Module):
    def __init__(self, in_l, in_g, hid=64,dropout=0.3):
        super().__init__()
        self.local_conv=GATConv(in_l,hid)
        self.global_conv=GCNConv(in_g,hid)
        self.edge_layer=EdgeWeightLayer()
        self.fuse=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hid,1))
    def forward(self,data):
        src,dst=data.edge_index
        noise_feat=data.noise_features[src]
        noise_ind=self.edge_layer(noise_feat)
        dist=(data.coord[src]-data.coord[dst]).norm(dim=1)
        gw=torch.exp(-dist**2/(2*sigma**2))
        w=smooth_mapping(gw*(1+noise_ind))
        mask=w>=0.2; e_idx=data.edge_index[:,mask]; e_w=w[mask]
        x_l=F.elu(self.local_conv(data.x_local,e_idx,e_w))
        x_g=F.relu(self.global_conv(data.x_global,e_idx,e_w))
        cat=torch.cat([x_l,x_g],dim=1)
        return self.fuse(cat).view(-1), cat

# -------------------------
# 6. EMPHASIZE FEATURES
# -------------------------
def emphasize_features(noise_tensor, cols, factor=3.0):
    idxs=[i for i,c in enumerate(cols) if '11210' in c]
    if idxs: noise_tensor[:,idxs]*=factor
    return noise_tensor

# -------------------------
# 7. PREDICT & SPATIAL PLOTS
# -------------------------
def predict_and_plot(city):
    df=pd.read_csv(f".../{city}_predictions.csv")
    df,loc_feats,glob_feats=preprocess_features(df)
    data=construct_graph(df,loc_feats,glob_feats,save_path=f"{SAVE_DIR}/{city}.pt").to(DEVICE)
    nf=torch.zeros((data.num_nodes,len(loc_feats))) if not hasattr(data,'noise_features') else data.noise_features
    data.noise_features=nf
    model=DualBranchGNN(len(loc_feats),len(glob_feats)).to(DEVICE)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pth"))
    model.eval()
    preds,fused=model(data)
    preds=preds.cpu().numpy(); df['pred_meandBA']=preds
    df.to_csv(f"{SAVE_DIR}/{city}_pred.geo.csv",index=False)
    plt.figure(figsize=(8,6))
    norm=TwoSlopeNorm(vmin=preds.min(),vcenter=np.median(preds),vmax=preds.max())
    plt.scatter(df['col'],df['row'],c=preds,cmap='RdYlBu_r',norm=norm,s=5)
    plt.gca().invert_yaxis(); plt.colorbar(label='dBA')
    plt.title(f"{city} Predictions"); plt.savefig(f"{SAVE_DIR}/{city}_map.png",dpi=300); plt.close()
    return fused.cpu().numpy()

# -------------------------
# 8. EXTRACT 64D FEATURES + PREDICTIONS
# -------------------------
def generate_features_dataset(cities, model_path, factor=3.0):
    all_df=pd.DataFrame()
    mapping={'southampton':0,'cardiff':1,'portsmouth':2,'nottingham':3,'liverpool':4}
    sample=cities[0]; df0=pd.read_csv(f".../{sample}_predictions.csv")
    df0,loc0,glob0=preprocess_features(df0)
    model=DualBranchGNN(len(loc0),len(glob0)).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    for city in cities:
        df=pd.read_csv(f".../{city}_predictions.csv")
        df,loc_feats,glob_feats=preprocess_features(df)
        data=construct_graph(df,loc_feats,glob_feats).to(DEVICE)
        nf=torch.tensor(df[loc_feats].values,dtype=torch.float32).to(DEVICE)
        data.noise_features=emphasize_features(nf,loc_feats,factor)
        with torch.no_grad(): preds,fused=model(data)
        fused=fused.cpu().numpy(); preds=preds.cpu().numpy()
        tmp=pd.DataFrame(fused,columns=[f"f{i}" for i in range(fused.shape[1])])
        tmp['city']=city; tmp['domain']=mapping[city]; tmp['pred_meandBA']=preds
        all_df=pd.concat([all_df,tmp],ignore_index=True)
    out=f"all_city_features_64d_with_preds.csv"; all_df.to_csv(out,index=False)
    return all_df

# -------------------------
# 9. t-SNE & UMAP VISUALIZATION
# -------------------------
def visualize_embeddings(csv_path):
    df=pd.read_csv(csv_path); feats=[c for c in df if c.startswith('f')]
    emb_tsne=TSNE(n_components=2,perplexity=50,learning_rate=200,n_iter=1000,random_state=42).fit_transform(df[feats])
    df['TSNE1'],df['TSNE2']=emb_tsne[:,0],emb_tsne[:,1]
    emb_umap=umap.UMAP(n_neighbors=30,min_dist=0.3,random_state=42).fit_transform(df[feats])
    df['UMAP1'],df['UMAP2']=emb_umap[:,0],emb_umap[:,1]
    cities=sorted(df['city'].unique())
    colors=generate_pastel_colors(len(cities))
    cmap={city:col for city,col in zip(cities,colors)}
    # Combined plot
    plt.figure(figsize=(12,5))
    for ax_idx,(col,title) in enumerate([('TSNE','t-SNE'),('UMAP','UMAP')]):
        plt.subplot(1,2,ax_idx+1)
        for city in cities:
            sub=df[df['city']==city]
            plt.scatter(sub[col+'1'],sub[col+'2'],c=[cmap[city]],label=city,s=5,alpha=0.5)
        plt.title(f"{title} of Fused Features"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{SAVE_DIR}/embeddings_combined.png",dpi=300); plt.close()

def generate_pastel_colors(n):
    base=[0.0,0.1,0.2,0.3,0.4]; cols=[]
    for i in range(n):
        h=base[i%len(base)]; s=0.5+0.1*np.sin(i*0.5); v=0.9-0.05*i
        cols.append(colorsys.hsv_to_rgb(h,s,v))
    return cols

# -------------------------
# MAIN
# -------------------------
if __name__=='__main__':
    cities=["cardiff","portsmouth","nottingham","liverpool","southampton"]
    # 1) Spatial predictions & maps
    for c in cities: predict_and_plot(c)
    # 2) Build 64d features dataset
    generate_features_dataset(cities,f"{SAVE_DIR}/multi_domain_best_model_r2.pth",factor=3.0)
    # 3) Embedding visualization
    visualize_embeddings("all_city_features_64d_with_preds.csv")
