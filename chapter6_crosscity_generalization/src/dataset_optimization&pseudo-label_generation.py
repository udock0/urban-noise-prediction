#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_optimization.py

This script performs dataset optimization and pseudo-label generation:
 1) Computes local and global noise ratio features and writes optimized Parquet datasets.
 2) Selects top N important features for local and global datasets via correlation filtering and RandomForest importance.
 3) Generates pseudo-labels for target cities using k-NN similarity on selected features.

Usage:
    python dataset_optimization.py optimize        # compute noise ratios and save processed datasets
    python dataset_optimization.py select-local    # feature selection for local dataset
    python dataset_optimization.py select-global   # feature selection for global dataset
    python dataset_optimization.py pseudo-label    # generate pseudo-labels for target cities

Ensure dask, scikit-learn, pandas, numpy, matplotlib are installed.
"""

import os
import json
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# ------------------ GLOBAL CONFIG ------------------
OUTPUT_FOLDER = "/content/drive/MyDrive/dataset"
# Parquet paths for raw processed features
LOCAL_PARQUET_RAW = os.path.join(OUTPUT_FOLDER, "local_info_dataset_processed.parquet")
GLOBAL_PARQUET_RAW = os.path.join(OUTPUT_FOLDER, "global_info_dataset_processed.parquet")
# Output paths after noise ratio
LOCAL_PARQUET_OPT = os.path.join(OUTPUT_FOLDER, "local_info_dataset_processed0.parquet")
GLOBAL_PARQUET_OPT = os.path.join(OUTPUT_FOLDER, "global_info_dataset_processed0.parquet")
# Output paths after feature selection
LOCAL_PARQUET_SEL = os.path.join(OUTPUT_FOLDER, "local_info_dataset_feature_selected0.parquet")
GLOBAL_PARQUET_SEL = os.path.join(OUTPUT_FOLDER, "global_info_dataset_feature_selected0.parquet")
LOCAL_FEATURES_JSON = os.path.join(OUTPUT_FOLDER, "selected_local_features0.json")
GLOBAL_FEATURES_JSON = os.path.join(OUTPUT_FOLDER, "selected_global_features0.json")

# Noise categories (LULC codes)
STRONG_NOISE_LULC_CODES = ['12210','12300','12230','12400','13100']
MODERATE_NOISE_LULC_CODES = ['12100','12220']
MODERATE_MITIGATION_LULC_CODES = ['14100','14200','32000','33000','23000']
STRONGEST_MITIGATION_LULC_CODES = ['21000','31000']

# Pseudo-label configuration
TARGET_CITIES = ['cardiff','liverpool','nottingham','portsmouth']
REFERENCE_CITY = 'southampton'
K_NEIGHBORS = 5
STD_THRESHOLD = 10.0
CONFIDENCE_PERCENTILE = 90
SELECTED_PSEUDO_FEATURES = [
    "landuse_12220_r30", "local_moderate_noise_ratio", "landuse_12220_r60",
    "local_moderate_mitigation_ratio", "landuse_11210_r60", "local_dist_12220",
    "global_dist_12220", "global_dist_12100", "global_moderate_noise_ratio", "global_dist_14100"
]

# ------------------ FUNCTIONS ------------------

def remove_unwanted_columns_global(ddf):
    """
    Drop columns not needed in global dataset.
    """
    to_drop = [c for c in ["dominant_lulc","dominant_lulc_category","NEI"] if c in ddf.columns]
    if to_drop:
        print(f"[INFO] Dropping from global dataset: {to_drop}")
        return ddf.drop(columns=to_drop)
    print("[INFO] No unwanted columns in global dataset.")
    return ddf


def add_noise_ratios(df, suffixes, prefix):
    """
    Compute noise ratio features for given landuse suffixes.
    """
    cols = [c for c in df.columns if c.startswith("landuse_") and any(c.endswith(s) for s in suffixes)]
    def compute_ratios(row):
        total = strong = moderate = mod_miti = strong_miti = 0.0
        for c in cols:
            code = c.split('_')[1]
            val = float(row[c]) if not pd.isnull(row[c]) else 0.0
            total += val
            if code in STRONG_NOISE_LULC_CODES: strong += val
            elif code in MODERATE_NOISE_LULC_CODES: moderate += val
            elif code in MODERATE_MITIGATION_LULC_CODES: mod_miti += val
            elif code in STRONGEST_MITIGATION_LULC_CODES: strong_miti += val
        if total>0:
            return (strong/total, moderate/total, mod_miti/total, strong_miti/total)
        return (0.,0.,0.,0.)
    ratios = df.apply(compute_ratios, axis=1, result_type='expand')
    ratios.columns = [f"{prefix}_strong_noise_ratio",
                      f"{prefix}_moderate_noise_ratio",
                      f"{prefix}_moderate_mitigation_ratio",
                      f"{prefix}_strongest_mitigation_ratio"]
    return df.assign(**{col: ratios[col].values for col in ratios.columns})


def process_local_dataset():
    print(f"[INFO] Reading local dataset: {LOCAL_PARQUET_RAW}")
    ddf = dd.read_parquet(LOCAL_PARQUET_RAW)
    print(f"[INFO] Local shape: {ddf.shape}")

    meta = ddf._meta.copy()
    for n in ["local_strong_noise_ratio","local_moderate_noise_ratio",
              "local_moderate_mitigation_ratio","local_strongest_mitigation_ratio"]:
        meta[n] = np.float32()
    ddf = ddf.map_partitions(add_noise_ratios,
                               suffixes=["_r30","_r60","_r120"],
                               prefix="local",meta=meta)
    print(f"[INFO] Writing optimized local dataset to: {LOCAL_PARQUET_OPT}")
    with ProgressBar(): ddf.to_parquet(LOCAL_PARQUET_OPT, write_index=False)
    print("[INFO] Local dataset processing complete.")


def process_global_dataset():
    print(f"[INFO] Reading global dataset: {GLOBAL_PARQUET_RAW}")
    ddf = dd.read_parquet(GLOBAL_PARQUET_RAW)
    print(f"[INFO] Global shape: {ddf.shape}")
    ddf = remove_unwanted_columns_global(ddf)
    meta = ddf._meta.copy()
    for n in ["global_strong_noise_ratio","global_moderate_noise_ratio",
              "global_moderate_mitigation_ratio","global_strongest_mitigation_ratio"]:
        meta[n] = np.float32()
    ddf = ddf.map_partitions(add_noise_ratios,
                               suffixes=["_r500","_r1000"],
                               prefix="global",meta=meta)
    print(f"[INFO] Writing optimized global dataset to: {GLOBAL_PARQUET_OPT}")
    with ProgressBar(): ddf.to_parquet(GLOBAL_PARQUET_OPT, write_index=False)
    print("[INFO] Global dataset processing complete.")


def feature_selection(parquet_path, non_feature_cols, keep_n, output_parquet, output_json, scope):
    print(f"[INFO] Loading {scope} dataset for feature selection: {parquet_path}")
    ddf = dd.read_parquet(parquet_path)
    cols = list(ddf.columns)
    non_feats = [c for c in non_feature_cols if c in cols]
    feats = [c for c in cols if c not in non_feats]
    print(f"[INFO] {scope} initial feature count: {len(feats)}")
    # correlation filter
    df_lab = ddf[ddf['set_id']==3][feats]
    sample = df_lab.sample(frac=0.25, random_state=42).compute()
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col]>0.95)]
    feats = [c for c in feats if c not in to_drop]
    print(f"[INFO] After correlation filter: {len(feats)} features")
    # random forest importance
    df_imp = ddf[ddf['set_id']==3][feats+['meandBA']].compute()
    X = df_imp[feats].fillna(0)
    y = df_imp['meandBA'].fillna(0)
    rf = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
    rf.fit(X,y)
    imp_df = pd.DataFrame({'feature':feats,'importance':rf.feature_importances_})
    top = imp_df.sort_values('importance',ascending=False).head(keep_n)['feature'].tolist()
    print(f"[INFO] Top {keep_n} {scope} features selected.")
    final_cols = non_feats + top
    # save JSON
    with open(output_json,'w',encoding='utf-8') as f: json.dump(top,f,indent=2)
    # write parquet
    ddf_final = ddf[final_cols]
    with ProgressBar(): ddf_final.to_parquet(output_parquet, write_index=False)
    print(f"[INFO] {scope} feature selection complete. Output: {output_parquet}")


def generate_pseudo_labels(csv_path):
    print(f"[INFO] Loading merged CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    mask = (df['set_id']==3)&df['meandBA'].notna()&(df['city'].str.lower()==REFERENCE_CITY)
    df_ref = df[mask].copy()
    print(f"[INFO] {REFERENCE_CITY} labeled samples: {len(df_ref)}")
    X_ref = df_ref[SELECTED_PSEUDO_FEATURES].values
    scaler = StandardScaler().fit(X_ref)
    X_ref_s = scaler.transform(X_ref)
    results = []
    for city in TARGET_CITIES:
        df_city = df[df['city'].str.lower()==city].copy()
        X_city_s = scaler.transform(df_city[SELECTED_PSEUDO_FEATURES].values)
        nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(X_ref_s)
        dist, idx = nbrs.kneighbors(X_city_s)
        pseudo = []
        conf = []
        stds = []
        for d,i in zip(dist,idx):
            labs = df_ref.iloc[i]['meandBA'].values
            pseudo.append(labs.mean())
            avgd = d.mean()
            conf.append(1.0/(avgd+1e-5))
            stds.append(labs.std())
        df_city['pseudo_meandBA'] = pseudo
        df_city['confidence'] = conf
        df_city['label_std'] = stds
        thr = np.percentile(df_city['confidence'],CONFIDENCE_PERCENTILE)
        mask_high = (df_city['confidence']>=thr)&(df_city['label_std']<=STD_THRESHOLD)
        df_city['meandBA'] = np.where(mask_high,df_city['pseudo_meandBA'],0)
        df_city['set_id'] = np.where(mask_high,3,9)
        # plot
        plt.figure(figsize=(6,3)); plt.hist(df_city['confidence'],bins=50); plt.title(f"{city} confidence"); plt.show()
        out = os.path.join(OUTPUT_FOLDER,f"pseudo_labels_{city}.csv")
        df_city.to_csv(out,index=False)
        print(f"[INFO] Saved {city} pseudo labels to {out}")
        results.append(df_city)
    df_all = pd.concat(results,ignore_index=True)
    all_out = os.path.join(OUTPUT_FOLDER,"pseudo_labels_all_target_cities.csv")
    df_all.to_csv(all_out,index=False)
    print(f"[INFO] Saved all pseudo labels to {all_out}")

# ------------------ MAIN ------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['optimize','select-local','select-global','pseudo-label'])
    args = parser.parse_args()
    if args.command=='optimize':
        process_global_dataset(); process_local_dataset()
    elif args.command=='select-local':
        non_feat = ["set_id","meandBA","row","col","city","located_lulc","local_dominant_lulc",
                    "local_strong_noise_ratio","local_moderate_noise_ratio","local_moderate_mitigation_ratio","local_strongest_mitigation_ratio",
                    "local_dist_12100","local_dist_12210","local_dist_12220","local_dist_12230","local_dist_12300"]
        feature_selection(LOCAL_PARQUET_OPT,non_feat,80,LOCAL_PARQUET_SEL,LOCAL_FEATURES_JSON,'local')
    elif args.command=='select-global':
        non_feat = ["Index","set_id","meandBA","row","col","city","located_lulc","global_dominant_lulc",
                    "global_strong_noise_ratio","global_moderate_noise_ratio","global_moderate_mitigation_ratio","global_strongest_mitigation_ratio",
                    "global_dist_12100","global_dist_12210","global_dist_12220","global_dist_12230","global_dist_12300"]
        feature_selection(GLOBAL_PARQUET_OPT,non_feat,80,GLOBAL_PARQUET_SEL,GLOBAL_FEATURES_JSON,'global')
    elif args.command=='pseudo-label':
        csv = os.path.join(OUTPUT_FOLDER,'all_cities.csv')
        generate_pseudo_labels(csv)
    else:
        parser.print_help()
