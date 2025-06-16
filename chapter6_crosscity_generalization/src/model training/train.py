#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_training.py

This script implements multi-domain adversarial fine-tuning of a dual-branch Graph Neural Network (GNN) for urban noise prediction.
It includes:
 1) Data preprocessing: encoding categorical LULC, normalizing features, handling pseudo-labels.
 2) Graph construction: efficient neighbor lookup, spatial weighting using Gaussian kernels.
 3) Model definitions:
    - GradReverse layer for adversarial gradient reversal.
    - MultiDomainDiscriminator for domain classification loss.
    - MMD loss for feature alignment across domains.
    - SmoothMapping and EdgeWeightLayer to dynamically weight edges.
    - DualBranchGNN_Deep combining local (GAT) and global (GCN) branches with fusion gates.
 4) Training loops:
    - Preheating with regression loss only.
    - Adversarial fine-tuning including regression, adversarial domain loss, and MMD loss.
 5) Evaluation: MAE and R<sup>2</sup> metrics on validation set.

Usage:
    python model_training.py --source source.csv --pseudo pseudo.csv --save_dir ./models

Command-line arguments allow specifying file paths and training hyperparameters.
"""
import os
import argparse
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------
# Global Configuration
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4      # initial learning rate for Adam
EPOCHS_FINE_TUNE = 300     # total epochs including preheat
PREHEAT_EPOCHS = 5        # epochs using only regression loss
K_HOP = 3                 # graph radius in grid steps
SIGMA = 30.0              # Gaussian kernel width for spatial weighting
LAMBDA_ADV = 0.05         # weight for adversarial (domain) loss
LAMBDA_MMD = 0.1          # weight for MMD alignment loss
USE_MMD = True            # toggle MMD loss in training

# -------------------------
# 1) Gradient Reversal Layer
# -------------------------
class GradReverse(Function):
    """
    Gradient Reversal: during forward pass returns input unchanged;
    during backward pass multiplies gradients by -lambda_, implementing adversarial objective.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        # reverse gradient
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    """Helper to apply gradient reversal."""
    return GradReverse.apply(x, lambda_)

# -------------------------
# 2) Multi-Domain Discriminator
# -------------------------
class MultiDomainDiscriminator(nn.Module):
    """
    A simple MLP to classify node features into domain IDs.
    Used to adversarially align feature distributions across domains.
    """
    def __init__(self, input_dim, hidden_dim=64, num_domains=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains)
        )
    def forward(self, x):
        """Return raw logits for domain classification."""
        return self.net(x)

# -------------------------
# 3) Maximum Mean Discrepancy (MMD) Loss
# -------------------------
def mmd_loss(x_src, x_tgt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute MMD between source and target feature sets using a mixture of Gaussian kernels.
    Encourages matching distributions across domains.
    """
    total = torch.cat([x_src, x_tgt], dim=0)
    # pairwise squared distances
    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    L2 = ((total0 - total1)**2).sum(2)
    # determine bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        n = total.size(0)
        bandwidth = L2.sum().item() / (n*n - n)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # build kernels
    kernels = sum(torch.exp(-L2 / bw) for bw in bandwidth_list)
    # split
    n_src = x_src.size(0)
    XX = kernels[:n_src, :n_src]
    YY = kernels[n_src:, n_src:]
    XY = kernels[:n_src, n_src:]
    return XX.mean() + YY.mean() - 2*XY.mean()

# -------------------------
# 4) Smooth Mapping for Edge Weights
# -------------------------
def smooth_mapping(x, k=1.0, m=1.0):
    """
    Map input values via a sigmoid into [0.1, 2.0].
    Used to scale dynamic edge weights.
    """
    lower, upper = 0.1, 2.0
    return lower + (upper-lower) * torch.sigmoid(k*(x-m))

class EdgeWeightLayer(nn.Module):
    """
    Learns to adjust raw noise features into a dynamic edge weight indicator.
    Initialized near identity for stable start.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        nn.init.eye_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(10, 1)
        # init second layer to emphasize different feature contributions
        init_w = torch.tensor([[1.0,0.7,0.5,0.3,-0.5,-0.8,-0.3,-0.5,0.2,-0.2]])
        with torch.no_grad(): self.fc2.weight.copy_(init_w)
        nn.init.zeros_(self.fc2.bias)
        self.activation = nn.ELU()
    def forward(self, feats):
        """Return scalar indicator per edge for downstream mapping."""
        h = self.activation(self.fc1(feats))
        return self.fc2(h).squeeze(1)

# -------------------------
# 5) Dual-Branch GNN Model
# -------------------------
class DualBranchGNN_Deep(nn.Module):
    """
    A two-branch GNN: local subgraph via GAT and global subgraph via GCN,
    fused through gating and adapters, with edge re-weighting.
    """
    def __init__(self,
                 input_local, input_global,
                 hidden_dim=64, n_layers_local=2, n_layers_global=1,
                 adapter_dim=32, dropout_rate=0.3):
        super().__init__()
        # Local branch: GAT layers + batchnorm
        self.local_convs = nn.ModuleList()
        self.local_bns = nn.ModuleList()
        ch = input_local
        for _ in range(n_layers_local):
            self.local_convs.append(GATConv(ch, hidden_dim))
            self.local_bns.append(nn.BatchNorm1d(hidden_dim))
            ch = hidden_dim
        self.local_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        # Global branch: GCN layers + layernorm
        self.global_convs = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        cg = input_global
        for _ in range(n_layers_global):
            self.global_convs.append(GCNConv(cg, hidden_dim))
            self.global_norms.append(nn.LayerNorm(hidden_dim))
            cg = hidden_dim
        self.global_adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        # Fusion gate and output
        self.fusion_adapter = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_gate = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.edge_weight_layer = EdgeWeightLayer()
        self.sigma = SIGMA
    def forward(self, data):
        """
        Forward pass:
         1. Compute dynamic edge weights based on spatial & noise features.
         2. Apply GAT on local features and GCN on global features.
         3. Fuse via learned gating and predict node-level output.
        Returns predictions and fused node features.
        """
        # ...implementation omitting for brevity... (see script)        
        pass

# -------------------------
# 6) Data Preprocessing
# -------------------------
def preprocess_features(df):
    """
    - Coerce meandBA to numeric; set pseudo-labels (set_id > 3) to zero.
    - One-hot encode categorical LULC columns.
    - Separate and fill local/global numeric features.
    Returns processed df, list of local and global feature names.
    """
    # ...implementation...        
    return df, local_feats, global_feats

# -------------------------
# 7) Graph Construction
# -------------------------
# Efficiently builds edge_index and edge_weight tensors using row/col lookup.
def construct_global_graph_mixed_efficient(df, local_feats, global_feats, graph_save_path=None):
    """
    Build or load a PyG Data object:
     - Node features: local and global branches
     - Edge index: k-hop grid neighbors
     - Edge weight: Gaussian + dynamic mapping
     - Attributes: labels, domain IDs, coordinates
    """
    # ...implementation...        
    return data

# -------------------------
# 8) Training / Evaluation
# -------------------------
def train_epoch(model, discriminator, loader, optimizer, epoch):
    """
    Single epoch of training:
     - Preheat: only regression loss for first PREHEAT_EPOCHS
     - Then include adversarial domain and MMD losses
     - Gradient clipping for stability
    Returns average training loss.
    """
    # ...implementation...        
    return avg_loss

@torch.no_grad()
def evaluate_loader(model, loader):
    """
    Evaluate on validation loader, computing Huber loss, MAE, and R^2.
    Only uses samples with set_id <= 3 (true labels).
    """
    # ...implementation...        
    return loss, mae, r2

# -------------------------
# 9) Adversarial Fine-Tuning Loop
# -------------------------
def fine_tune_adversarial_multi_domain(source_path, pseudo_path, output_dir):
    """
    Orchestrates:
     a) Load source and pseudo-labeled target datasets, assign domain IDs
     b) Split train/val stratified by domain
     c) Construct graphs and DataLoaders
     d) Instantiate model + discriminator + optimizer + scheduler
     e) Loop epochs: train_epoch + evaluate_loader + checkpoint best R^2
    """
    # ...implementation...        
    return best_model

# -------------------------
# Main CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-domain adversarial GNN training script")
    parser.add_argument('--source', required=True, help="CSV path for source (reference) dataset")
    parser.add_argument('--pseudo', required=True, help="CSV path for pseudo-labeled target datasets")
    parser.add_argument('--save_dir', default="./models", help="Directory to save trained graphs and models")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # Kick off fine-tuning
    fine_tune_adversarial_multi_domain(args.source, args.pseudo, args.save_dir)
