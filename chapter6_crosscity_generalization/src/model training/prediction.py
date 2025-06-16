#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_prediction.py

This script loads a fine-tuned Dual-Branch GNN model to perform noise level predictions
and generates simple spatial feature maps for multiple cities.

Steps:
 1. Environment & dependencies: Torch, PyG, Matplotlib
 2. Preprocess features: one-hot encode LULC, normalize numeric inputs, set domain
 3. Build graph: k-hop grid adjacency with Gaussian + learned noise weights
 4. Load trained model weights and reconstruct model architecture
 5. Emphasize specific LULC features (e.g., code '11210') if desired
 6. Predict meandBA for each vector point
 7. Assign predictions back to GeoDataFrame for export in ArcGIS Pro (vectorization step)
 8. Visualize spatial distribution of predictions (scatter plots saved under `result/`)

Note: Example result maps are available in the `result/` folder for reference.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

# ===== GLOBAL CONFIGURATION =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 30.0      # Gaussian kernel width for edge weights
K_HOP = 3         # Neighborhood radius (in grid cells)
SAVE_DIR = "result"  # Directory to save output CSVs and plots
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Data Preprocessing
# -------------------------
def preprocess_features(df):
    """
    1. Convert target to numeric and fill NaNs with zero
    2. One-hot encode LULC categorical columns
    3. Identify local/global feature subsets and fill numeric columns
    4. Ensure a domain column exists for graph construction
    """
    df['meandBA'] = pd.to_numeric(df.get('meandBA', 0), errors='coerce').fillna(0.0)
    cat_cols = ["located_lulc", "local_dominant_lulc", "global_dominant_lulc"]
    categories = [
        '11100','11210','11220','11230','11240','12210','14100','12100',
        '12220','23000','12300','12230','14200','13300','31000','21000',
        '12400','13100','13400','40000','50000','32000','33000','11300'
    ]
    for col in cat_cols:
        if col in df:
            df[col] = pd.Categorical(df[col], categories=categories)
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    local_feats = [c for c in df.columns if ('r30' in c or 'r60' in c or 'r120' in c or c.startswith('local_'))]
    global_feats = [c for c in df.columns if ('r500' in c or 'r1000' in c or c.startswith('global_'))]
    df[local_feats] = df[local_feats].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df[global_feats] = df[global_feats].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if 'domain' not in df:
        df['domain'] = 1  # Single-domain for prediction
    return df, local_feats, global_feats

# -------------------------
# Smooth Mapping for Weights
# -------------------------
def smooth_mapping(x, k=1.0, m=1.0):
    """
    Maps raw edge weights into [0.1, 2.0] range using a sigmoid-based transform.
    """
    lower, upper = 0.1, 2.0
    return lower + (upper-lower) * torch.sigmoid(k * (x - m))

# -------------------------
# Graph Construction
# -------------------------
def construct_global_graph(df, local_feats, global_feats, save_path=None):
    """
    Build a PyG Data object:
      - x_local, x_global: node features
      - edge_index, edge_weight: k-hop spatial graph
      - coord, set_ids, domain for later use
    Caches to `save_path` if provided.
    """
    if save_path and os.path.exists(save_path):
        return torch.load(save_path)
    df = df.reset_index(drop=True)
    N = len(df)
    x_local = torch.tensor(df[local_feats].values, dtype=torch.float32)
    x_global = torch.tensor(df[global_feats].values, dtype=torch.float32)
    y = torch.tensor(df['meandBA'].values, dtype=torch.float32)
    set_ids = torch.tensor([3]*N, dtype=torch.long)
    domain = torch.tensor(df['domain'].values, dtype=torch.long)
    coords = df[['row','col']].astype(int).values
    coord_map = {tuple(rc):i for i, rc in enumerate(coords)}
    edges, weights = [], []
    for i,(r,c) in enumerate(coords):
        for dx in range(-K_HOP, K_HOP+1):
            for dy in range(-K_HOP, K_HOP+1):
                if dx==0 and dy==0: continue
                nb = (r+dx, c+dy)
                if nb in coord_map:
                    j = coord_map[nb]
                    edges.append([i,j])
                    dist = 30.0*np.hypot(dx,dy)
                    gw = np.exp(-dist**2/(2*sigma**2))
                    w = smooth_mapping(torch.tensor(gw)).item()
                    weights.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    data = Data(x_local=x_local, x_global=x_global, y=y,
                edge_index=edge_index, edge_weight=edge_weight)
    data.set_ids = set_ids
    data.domain = domain
    data.coord = torch.tensor(coords, dtype=torch.float32)
    if save_path:
        torch.save(data, save_path)
    return data

# -------------------------
# Emphasize Specific LULC Features
# -------------------------
def emphasize_features(noise_tensor, noise_cols, factor=3.0):
    """
    Amplify columns containing '11210' by given factor in the noise feature tensor.
    """
    idxs = [i for i,c in enumerate(noise_cols) if '11210' in c]
    if idxs:
        noise_tensor[:, idxs] *= factor
    return noise_tensor

# -------------------------
# Dual-Branch GNN Definition
# -------------------------
class EdgeWeightLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        nn.init.eye_(self.fc1.weight)
        self.fc2 = nn.Linear(10,1)
        init_w = torch.tensor([[1.0,0.7,0.5,0.3,-0.5,-0.8,-0.3,-0.5,0.2,-0.2]])
        with torch.no_grad(): self.fc2.weight.copy_(init_w)
        nn.init.zeros_(self.fc2.bias)
        self.act = nn.ELU()
    def forward(self,x): return self.fc2(self.act(self.fc1(x))).squeeze(1)

class DualBranchGNN(nn.Module):
    def __init__(self, in_local, in_global, hidden=64, dropout=0.3):
        super().__init__()
        # Local branch: GATConv stack
        self.local_convs = nn.ModuleList([GATConv(in_local,hidden)])
        # Global branch: GCNConv stack
        self.global_convs = nn.ModuleList([GCNConv(in_global,hidden)])
        self.edge_layer = EdgeWeightLayer()
        self.fusion = nn.Sequential(
            nn.Linear(hidden*2,hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden,1)
        )
    def forward(self,data):
        src,dst = data.edge_index
        noise_feats = data.noise_features[src]
        noise_ind = self.edge_layer(noise_feats)
        coords = data.coord
        dist = (coords[src]-coords[dst]).norm(dim=1)
        gw = torch.exp(-dist**2/(2*sigma**2))
        w = smooth_mapping(gw*(1+noise_ind))
        mask = w>=0.2
        e_idx = data.edge_index[:,mask]
        e_w = w[mask]
        x_l = F.elu(self.local_convs[0](data.x_local,e_idx,e_w))
        x_g = F.relu(self.global_convs[0](data.x_global,e_idx,e_w))
        cat = torch.cat([x_l,x_g],dim=1)
        out = self.fusion(cat)
        return out.view(-1)

# -------------------------
# Prediction & Visualization
# -------------------------
def predict_and_visualize(city_list, factor=3.0):
    # Load model
    model = DualBranchGNN(10,10).to(DEVICE)  # use correct in_local/in_global dims
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR,'best_model.pth'), map_location=DEVICE))
    model.eval()
    noise_cols_sample = []  # set actual noise feature columns list
    for city in city_list:
        # Load city data
        csv = f"/content/drive/MyDrive/dataset/{city}_predictions.csv"
        df = pd.read_csv(csv)
        df, local_feats, global_feats = preprocess_features(df)
        data = construct_global_graph(df, local_feats, global_feats, save_path=None)
        data = data.to(DEVICE)
        if hasattr(data,'noise_features'):
            nf = data.noise_features
        else:
            nf = torch.zeros((data.num_nodes, len(noise_cols_sample)))
        nf = emphasize_features(nf, noise_cols_sample, factor)
        data.noise_features = nf
        with torch.no_grad():
            preds = model(data).cpu().numpy()
        df['pred_meandBA'] = preds
        # Save vector-ready CSV for ArcGIS Pro
        out_csv = os.path.join(SAVE_DIR,f"{city}_predictions.geo.csv")
        df.to_csv(out_csv,index=False)
        # Simple scatter map
        plt.figure(figsize=(8,6))
        norm = TwoSlopeNorm(vmin=preds.min(), vcenter=np.median(preds), vmax=preds.max())
        plt.scatter(df['col'], df['row'], c=preds, cmap='RdYlBu_r', norm=norm, s=5)
        plt.title(f"{city.capitalize()} Noise Predictions")
        plt.colorbar(label='meandBA')
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(SAVE_DIR,f"{city}_map.png"), dpi=300)
        plt.close()

if __name__ == '__main__':
    cities = ["cardiff","portsmouth","nottingham","liverpool","southampton"]
    predict_and_visualize(cities, factor=3.0)
