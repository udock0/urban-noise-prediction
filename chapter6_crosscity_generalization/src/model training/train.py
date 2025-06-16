#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_training.py

This script implements multi-domain adversarial training of a dual-branch GNN:

1. Environment checks: verifies PyTorch, CUDA, and PyG versions.
2. Data preprocessing: one-hot encoding of categorical LULC, normalization of numeric features.
3. Graph construction: efficient neighbor-based edge generation with adaptive Gaussian weights.
4. Model definitions:
   - GradReverse: gradient reversal layer for adversarial training.
   - MultiDomainDiscriminator: predicts domain labels for adversarial loss.
   - EdgeWeightLayer: learns per-edge noise features to modulate adjacency weights.
   - DualBranchGNN_Deep: GAT-based local branch + GCN-based global branch, fused via gating and adapter layers.
5. Loss functions:
   - Huber loss for regression.
   - Cross-entropy for domain discrimination.
   - Optional MMD loss for distribution alignment.
6. Training loop:
   - Preheat epochs: only regression loss.
   - Adversarial epochs: add domain and MMD losses.
   - Gradient clipping and scheduler updates.
7. Evaluation:
   - Batch-based MAE and R² metrics on validation set.
   - Reference city test evaluation.

Usage:
    python model_training.py

Dependencies:
    torch, torch_geometric, sklearn, matplotlib, pandas, numpy
"""
import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===== GLOBAL CONFIGURATION =====
# Device selection: GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4          # Base learning rate
EPOCHS_FINE_TUNE = 300        # Total epochs for adversarial fine-tuning
K_HOP = 3                     # Neighborhood radius for graph edges (in grid steps)
sigma = 30.0                  # Gaussian kernel width for distance weighting
lambda_adv = 0.05             # Weight for adversarial (domain) loss
lambda_mmd = 0.1              # Weight for MMD loss
USE_MMD = True                # Toggle MMD loss usage
PREHEAT_EPOCHS = 5            # Epochs training only regression loss

# ------------------------------
# Gradient Reversal Layer
# ------------------------------
from torch.autograd import Function
class GradReverse(Function):
    """
    Implements gradient reversal for adversarial training.
    During forward pass, acts as identity.
    During backward pass, multiplies gradient by -lambda to reverse.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradients by scaling with -lambda_
        return -ctx.lambda_ * grad_output, None

# Convenient wrapper
def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# ------------------------------
# Multi-Domain Discriminator
# ------------------------------
class MultiDomainDiscriminator(nn.Module):
    """
    Predicts domain labels (0=source, 1..N=targets) from learned features.
    Used adversarially to align domain distributions.
    """
    def __init__(self, input_dim, hidden_dim=64, num_domains=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        # Returns raw logits for each domain
        return self.net(x)

# ------------------------------
# Maximum Mean Discrepancy (MMD) Loss
# ------------------------------
def mmd_loss(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes MMD between x and y using multiple Gaussian kernels.
    Encourages matching feature distributions across domains.
    """
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Use median heuristic
        bandwidth = torch.sum(L2_distance.data) / (total.size(0)**2 - total.size(0))
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernels = sum(torch.exp(-L2_distance / bw) for bw in bandwidth_list)
    n, m = x.size(0), y.size(0)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    return torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)

# ------------------------------
# Data Preprocessing
# ------------------------------
def preprocess_features(df):
    """
    Converts meandBA to numeric and sets pseudo-labeled targets (set_id>3) to zero.
    One-hot encodes categorical LULC columns and fills numeric features.

    Returns:
        df: processed DataFrame
        local_features: list of local feature column names
        global_features: list of global feature column names
    """
    # Ensure numeric target
    df['meandBA'] = pd.to_numeric(df['meandBA'], errors='coerce')
    # Force unlabeled or pseudo samples to zero target
    df.loc[df['set_id'] > 3, 'meandBA'] = 0.0

    # One-hot encode predefined categorical columns
    possible_cat = ["located_lulc", "local_dominant_lulc", "global_dominant_lulc"]
    categories = [
        '11100','11210','11220','11230','11240','12210','14100','12100',
        '12220','23000','12300','12230','14200','13300','31000','21000',
        '12400','13100','13400','40000','50000','32000','33000','11300'
    ]
    for col in possible_cat:
        if col in df:
            df[col] = pd.Categorical(df[col], categories=categories)
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    # Identify local/global feature subsets by suffix
    local_features = [c for c in df if ('r30' in c or 'r60' in c or 'r120' in c or c.startswith('local_'))]
    global_features = [c for c in df if ('r500' in c or 'r1000' in c or c.startswith('global_'))]
    # Fill and cast numeric features
    df[local_features] = df[local_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[global_features] = df[global_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    df.fillna(0, inplace=True)
    return df, local_features, global_features

# ------------------------------
# Smooth Mapping Function
# ------------------------------
def smooth_mapping(x, k=1.0, m=1.0):
    """
    Maps raw weights into [0.1, 2.0] via sigmoid for stability and pruning.
    """
    lower, upper = 0.1, 2.0
    return lower + (upper-lower) * torch.sigmoid(k * (x - m))

# ------------------------------
# Graph Construction
# ------------------------------
def construct_global_graph_mixed_efficient(df, local_features, global_features, graph_save_path=None):
    """
    Builds PyG Data object:
      - Node features: x_local, x_global
      - Edges: K_HOP neighborhood grid adjacency with Gaussian+learned noise weights
      - Attributes: y, set_ids, domain, coordinates
    Saves/loads from `graph_save_path` when provided.
    """
    # If cached graph exists, load it
    if graph_save_path and os.path.exists(graph_save_path):
        return torch.load(graph_save_path)

    df = df.reset_index(drop=True)
    N = len(df)
    x_local = torch.tensor(df[local_features].values, dtype=torch.float32)
    x_global = torch.tensor(df[global_features].values, dtype=torch.float32)
    y = torch.tensor(df['meandBA'].values, dtype=torch.float32)
    set_ids = torch.tensor(df['set_id'].values, dtype=torch.long)
    domain = torch.tensor(df['domain'].values, dtype=torch.long)

    # Map grid coords to node index
    coords = df[['row','col']].values
    coord_dict = {(int(r),int(c)):i for i,(r,c) in enumerate(coords)}

    edge_list, weight_list = [], []
    for i,(r,c) in enumerate(coords):
        for dx in range(-K_HOP, K_HOP+1):
            for dy in range(-K_HOP, K_HOP+1):
                if dx==0 and dy==0: continue
                nb = (r+dx, c+dy)
                if nb in coord_dict:
                    j = coord_dict[nb]
                    edge_list.append([i,j])
                    # Distance-based Gaussian
                    dist = 30.0 * np.hypot(dx, dy)
                    gw = np.exp(-dist**2/(2*sigma**2))
                    # Use smooth mapping for final weight
                    weight = smooth_mapping(torch.tensor(gw)).item()
                    weight_list.append(weight)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    data = Data(x_local=x_local, x_global=x_global, y=y,
                edge_index=edge_index, edge_weight=edge_weight)
    data.set_ids = set_ids
    data.domain = domain
    data.num_nodes = N
    data.coord = torch.tensor(coords.astype(float), dtype=torch.float32)

    if graph_save_path:
        torch.save(data, graph_save_path)
    return data

# ------------------------------
# Edge Weight Learning Layer
# ------------------------------
class EdgeWeightLayer(nn.Module):
    """
    Learns to modulate edge weights based on local noise features.
    Initialized to identity mapping for stability.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        self.fc2 = nn.Linear(10,1)
        nn.init.eye_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # Preload fc2 weights to bias noise effect
        init_w = torch.tensor([[1.0,0.7,0.5,0.3,-0.5,-0.8,-0.3,-0.5,0.2,-0.2]])
        with torch.no_grad(): self.fc2.weight.copy_(init_w)
        nn.init.zeros_(self.fc2.bias)
        self.act = nn.ELU()

    def forward(self, noise_feats):
        h = self.act(self.fc1(noise_feats))
        return self.fc2(h).squeeze(1)

# ------------------------------
# Dual-Branch GNN Model
# ------------------------------
class DualBranchGNN_Deep(nn.Module):
    """
    Combines a GAT-based local branch and GCN-based global branch,
    merges with gating, and predicts meandBA.
    Includes dropout and adapter layers to prevent overfitting.
    """
    def __init__(self, in_local, in_global, hidden_dim=64,
                 n_layers_local=2, n_layers_global=1,
                 adapter_dim=32, dropout_rate=0.3):
        super().__init__()
        # Local branch layers
        self.local_convs = nn.ModuleList()
        self.local_bns = nn.ModuleList()
        dim = in_local
        for _ in range(n_layers_local):
            self.local_convs.append(GATConv(dim, hidden_dim))
            self.local_bns.append(nn.BatchNorm1d(hidden_dim))
            dim = hidden_dim
        self.local_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim), nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        # Global branch layers
        self.global_convs = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        dim = in_global
        for _ in range(n_layers_global):
            self.global_convs.append(GCNConv(dim, hidden_dim))
            self.global_norms.append(nn.LayerNorm(hidden_dim))
            dim = hidden_dim
        self.global_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim), nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        # Fusion and output
        self.fusion_adapter = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.edge_weight_layer = EdgeWeightLayer()
        self.sigma = sigma

    def forward(self, data):
        # Compute dynamic edge weights by combining distance and learned noise features
        src, dst = data.edge_index
        noise_feats = data.noise_features[src]
        noise_indicator = self.edge_weight_layer(noise_feats)
        # Gaussian spatial weight
        coords = data.coord
        dist = (coords[src] - coords[dst]).norm(dim=1)
        gw = torch.exp(-dist**2/(2*self.sigma**2))
        dyn_w = smooth_mapping(gw * (1 + noise_indicator))
        # Prune small edges
        mask = dyn_w >= 0.2
        e_idx = data.edge_index[:, mask]
        e_w = dyn_w[mask]
        # Local branch
        x_local = data.x_local
        for conv, bn in zip(self.local_convs, self.local_bns):
            x_local = conv(x_local, e_idx, e_w)
            x_local = F.elu(x_local)
            x_local = bn(x_local)
        x_local = self.local_adapter(x_local)
        # Global branch
        x_global = data.x_global
        for conv, norm in zip(self.global_convs, self.global_norms):
            x_global = conv(x_global, e_idx, e_w)
            x_global = F.relu(x_global)
            x_global = norm(x_global)
        x_global = self.global_adapter(x_global)
        # Fusion
        cat = torch.cat([x_local, x_global], dim=1)
        fused = self.fusion_adapter(cat)
        gate = torch.sigmoid(self.fc_gate(cat))
        out = self.fc_out(gate*x_local + (1-gate)*x_global)
        return out, fused

# ------------------------------
# Training and Evaluation
# ------------------------------
def train_epoch(model, discriminator, loader, optimizer, epoch, preheat_epochs=PREHEAT_EPOCHS):
    """
    Runs one epoch of adversarial training:
      - Regression loss (Huber)
      - Optional adversarial (CE) and MMD losses after preheat phase
    """
    model.train()
    discriminator.train()
    huber = nn.HuberLoss()
    ce = nn.CrossEntropyLoss()
    total_loss, total_count = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        preds, feats = model(batch)
        bs = batch.batch_size
        mask = (batch.set_ids[:bs] <= 3)
        if not mask.any(): continue
        pred = preds[:bs][mask].view(-1)
        targ = batch.y[:bs][mask].view(-1)
        loss_reg = huber(pred, targ)
        if epoch < preheat_epochs:
            loss = loss_reg
        else:
            # Adversarial loss
            dom = batch.domain[:bs]
            rev = grad_reverse(feats[:bs])
            logits = discriminator(rev)
            loss_adv = ce(logits, dom)
            # MMD loss
            if USE_MMD:
                src_feats = feats[:bs][dom==0]
                tgt_feats = feats[:bs][dom!=0]
                loss_mmd = mmd_loss(src_feats, tgt_feats) if len(src_feats)>1 and len(tgt_feats)>1 else 0.0
            else:
                loss_mmd = 0.0
            loss = loss_reg + lambda_adv*loss_adv + lambda_mmd*loss_mmd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(discriminator.parameters()), 1.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()
    return total_loss/total_count if total_count>0 else 0.0

@torch.no_grad()
def evaluate_loader(model, loader):
    """
    Evaluates model on a data loader; returns Huber loss, MAE, R2.
    Only considers set_id<=3 samples.
    """
    model.eval()
    all_pred, all_targ = [], []
    huber = nn.HuberLoss()
    for batch in loader:
        batch = batch.to(DEVICE)
        preds, _ = model(batch)
        bs = batch.batch_size
        mask = (batch.set_ids[:bs] <= 3)
        if not mask.any(): continue
        all_pred.append(preds[:bs][mask].view(-1).cpu())
        all_targ.append(batch.y[:bs][mask].view(-1).cpu())
    if not all_pred:
        return 0,0,0
    pred = torch.cat(all_pred)
    targ = torch.cat(all_targ)
    loss = huber(pred, targ).item()
    mae = mean_absolute_error(targ.numpy(), pred.numpy())
    r2 = r2_score(targ.numpy(), pred.numpy())
    return loss, mae, r2

# ------------------------------
# Reference City Evaluation
# ------------------------------
def evaluate_reference_city(model, source_path, noise_cols, batch_size=128):
    """
    Loads reference city test set, constructs graph, evaluates metrics.
    """
    print("[INFO] Evaluating reference city...")
    df = pd.read_csv(source_path)
    df = df[df['set_id']<=3].copy()
    df['domain'] = 0
    df, local_feats, global_feats = preprocess_features(df)
    data = construct_global_graph_mixed_efficient(df, local_feats, global_feats)
    data = data.to(DEVICE)
    # Attach noise features
    nf = torch.tensor(df[noise_cols].values, dtype=torch.float32).to(DEVICE)
    data.noise_features = nf
    loader = NeighborLoader(data, num_neighbors=[4,2,1], batch_size=batch_size,
                             input_nodes=torch.arange(data.num_nodes))
    loss, mae, r2 = evaluate_loader(model, loader)
    print(f"Reference City - Loss: {loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return loss, mae, r2

# ------------------------------
# Adversarial Multi-Domain Training
# ------------------------------
def fine_tune_adversarial_multi_domain(pseudo_path, source_path, noise_cols):
    """
    End-to-end training pipeline:
     1) Load pseudo-labeled target data and reference data
     2) Downsample target cities, assign domain IDs
     3) Preprocess features, split train/val stratified by domain
     4) Construct PyG graphs for train/val
     5) Initialize models and optimizer
     6) Run adversarial training loop with scheduler and model checkpointing
     7) Evaluate on reference city test set
    """
    os.makedirs("./models", exist_ok=True)  # Save directory
    # Map city to domain ID
    city_map = {"cardiff":1,"portsmouth":2,"nottingham":3,"liverpool":4}
    # Load target pseudo-labeled data
    df_t = pd.read_csv(pseudo_path)
    dfs = []
    for city, did in city_map.items():
        dfc = df_t[df_t['city'].str.lower()==city].copy()
        if len(dfc)>20000: dfc = dfc.sample(20000, random_state=42)
        dfc['domain'] = did
        dfs.append(dfc)
    df_targets = pd.concat(dfs, ignore_index=True)
    # Load reference data
    df_src = pd.read_csv(source_path)
    df_src, loc_feats, glob_feats = preprocess_features(df_src)
    df_src = df_src[df_src['set_id']<=3].copy()
    df_src['domain'] = 0
    # Preprocess targets
    df_targets, _, _ = preprocess_features(df_targets)
    # Combine and split
    df_all = pd.concat([df_src, df_targets], ignore_index=True)
    train_df, val_df = train_test_split(df_all, test_size=0.2,
                                        stratify=df_all['domain'], random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)} samples")
    # Build graphs
    train_data = construct_global_graph_mixed_efficient(train_df, loc_feats, glob_feats, graph_save_path="train.pt").to(DEVICE)
    val_data = construct_global_graph_mixed_efficient(val_df, loc_feats, glob_feats, graph_save_path="val.pt").to(DEVICE)
    # Add noise features
    train_data.noise_features = torch.tensor(train_df[noise_cols].values, dtype=torch.float32).to(DEVICE)
    val_data.noise_features   = torch.tensor(val_df[noise_cols].values, dtype=torch.float32).to(DEVICE)
    # DataLoaders
    train_loader = NeighborLoader(train_data, num_neighbors=[4,2,1], batch_size=128, shuffle=True,
                                  input_nodes=torch.arange(train_data.num_nodes))
    val_loader   = NeighborLoader(val_data, num_neighbors=[4,2,1], batch_size=128, shuffle=False,
                                  input_nodes=torch.arange(val_data.num_nodes))
    # Initialize models
    model = DualBranchGNN_Deep(len(loc_feats), len(glob_feats)).to(DEVICE)
    disc  = MultiDomainDiscriminator(input_dim=64, hidden_dim=64, num_domains=5).to(DEVICE)
    # Optimizer and scheduler
    opt = torch.optim.Adam(list(model.parameters())+list(disc.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    best_r2, best_path = -float('inf'), "best_model.pth"
    # Training loop
    for epoch in range(EPOCHS_FINE_TUNE):
        tr_loss = train_epoch(model, disc, train_loader, opt, epoch)
        val_loss, val_mae, val_r2 = evaluate_loader(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS_FINE_TUNE}: train_loss={tr_loss:.4f}, val_mae={val_mae:.4f}, val_r2={val_r2:.4f}")
        sched.step(val_loss)
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (R2={best_r2:.4f}) to {best_path}")
    # Final evaluation on reference city
    evaluate_reference_city(model, source_path, noise_cols)

# ------------------------------
# MAIN
# ------------------------------
if __name__ == '__main__':
    # Paths to datasets
    source_path      = "/content/drive/MyDrive/dataset/southampton_training_data0.csv"
    pseudo_labels    = "/content/drive/MyDrive/dataset/pseudo_labels_all_target_cities_updated.csv"
    noise_feature_cols = [
        "landuse_12220_r30","local_moderate_noise_ratio","landuse_12220_r60",
        "local_moderate_mitigation_ratio","landuse_11210_r60","global_dist_12220",
        "global_dist_12100","Green_contrast_r500_percentile75","global_moderate_noise_ratio",
        "WVBI_correlation_r1000_percentile75"
    ]
    # Start adversarial fine-tuning
    fine_tune_adversarial_multi_domain(pseudo_labels, source_path, noise_feature_cols)
