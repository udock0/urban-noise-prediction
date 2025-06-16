#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zscore_tanh_features.py

Purpose:
1. For multispectral imagery, compute local z-score + tanh maps at different window sizes
   for selected bands (e.g., NIR1, Green, RedEdge).
2. For a directory of single-band remote sensing indices, compute the same features.

- All outputs are GeoTIFF, with values in [-1,1], ready for machine learning or GIS.
- Assumes imagery is radiometrically and atmospherically corrected, and standardized.

Author: [Your Name]
Date: 2025-06-16
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
import glob

# -------------------------------
# Robust global normalization
# -------------------------------
def global_normalize(band_arr, method="median_mad"):
    """
    Global robust normalization using median/MAD, to reduce cross-city bias.
    """
    if method == "median_mad":
        median = np.nanmedian(band_arr)
        mad = np.median(np.abs(band_arr - median))
        normalized = (band_arr - median) / (mad + 1e-5)
    else:
        raise ValueError("Only 'median_mad' supported.")
    return normalized

# -------------------------------
# Local z-score with tanh mapping
# -------------------------------
def local_zscore_band(band_arr, win_size=5, eps=1e-5, pre_normalize=True):
    """
    Compute local z-score for a band (mean, std within window).
    Optionally, robust global normalization is done first.
    """
    if pre_normalize:
        band_arr = global_normalize(band_arr, method="median_mad")
    local_mean = uniform_filter(band_arr, size=win_size, mode='reflect')
    local_sqrmean = uniform_filter(band_arr**2, size=win_size, mode='reflect')
    local_var = local_sqrmean - (local_mean**2)
    local_std = np.sqrt(np.maximum(local_var, 0.0))
    zscore = (band_arr - local_mean) / (local_std + eps)
    return zscore.astype(np.float32)

def compute_tanh_zscore(band_arr, win_size=15, eps=1e-5, pre_normalize=True):
    """
    Local z-score followed by tanh mapping to [-1,1].
    """
    zscore = local_zscore_band(band_arr, win_size=win_size, eps=eps, pre_normalize=pre_normalize)
    z_tanh = np.tanh(zscore)
    return z_tanh.astype(np.float32)

# -------------------------------
# Process multispectral bands
# -------------------------------
def process_multispectral_rgb(in_path, out_dir, bands_to_process=[7,3,6], win_sizes=[15,31,77]):
    """
    For each band and each window size, compute local z-score + tanh map and save as GeoTIFF.
    Band numbers are 1-based (rasterio).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        bands_count = src.count
        H, W = src.height, src.width
        print("Processing multispectral image:", in_path)
        print(f"  Size = ({H}x{W}), Bands = {bands_count}")
        data_3d = src.read()
    
    for b_id in bands_to_process:
        if b_id < 1 or b_id > bands_count:
            print(f"Band {b_id} out of range (1~{bands_count}), skip.")
            continue
        zero_idx = b_id - 1
        print(f"Processing band {b_id} (index {zero_idx}) ...")
        band_arr = data_3d[zero_idx].astype(np.float32)
        
        for win_size in win_sizes:
            if win_size % 2 == 0:
                raise ValueError(f"win_size must be odd, got {win_size}")
            print(f"  Window size: {win_size}")
            z_tanh = compute_tanh_zscore(band_arr, win_size=win_size, pre_normalize=True)
            out_name = os.path.join(out_dir, f"z_band{b_id}_win{win_size}_tanh.tif")
            print(f"    Writing => {out_name}")
            profile_1band = profile.copy()
            profile_1band.update({
                "count": 1,
                "dtype": "float32"
            })
            with rasterio.open(out_name, "w", **profile_1band) as dst:
                dst.write(z_tanh, 1)
    print(f"Multispectral processing done. Results in => {out_dir}")

# -------------------------------
# Process indices in folder
# -------------------------------
def process_indices_folder(indices_folder, out_dir, win_sizes=[15,31,77]):
    """
    For each single-band .tif file in folder, compute local z-score+tanh map at all window sizes.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tif_files = glob.glob(os.path.join(indices_folder, "*.tif"))
    if not tif_files:
        print(f"No tif files in {indices_folder}")
        return
    print("Processing indices folder:", indices_folder)
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        name_no_ext = os.path.splitext(filename)[0]
        print(f"Processing index file: {filename}")
        with rasterio.open(tif_path) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(np.float32)
        for win_size in win_sizes:
            if win_size % 2 == 0:
                raise ValueError(f"win_size must be odd, got {win_size}")
            print(f"  Window size: {win_size}")
            z_tanh = compute_tanh_zscore(data, win_size=win_size, pre_normalize=True)
            out_name = os.path.join(out_dir, f"z_{name_no_ext}_win{win_size}_tanh.tif")
            print(f"    Writing => {out_name}")
            profile_1band = profile.copy()
            profile_1band.update({
                "count": 1,
                "dtype": "float32"
            })
            with rasterio.open(out_name, "w", **profile_1band) as dst:
                dst.write(z_tanh, 1)
    print(f"Indices processed. Results in => {out_dir}")

# -------------------------------
# Example batch processing for all cities
# -------------------------------
if __name__ == "__main__":
    # Paths for one city; modify for batch processing as needed
    multispectral_path = r"D:/PhD dissertation/portsmouth/raster/WV2_in_Portsmouth_final.tif"
    indices_folder = r"D:/PhD dissertation/portsmouth/rsindex"
    out_dir = r"D:/PhD dissertation/portsmouth/zscore"
    win_sizes = [15, 31, 77]
    process_multispectral_rgb(multispectral_path, out_dir, bands_to_process=[7,3,6], win_sizes=win_sizes)
    process_indices_folder(indices_folder, out_dir, win_sizes=win_sizes)
    print("All z-score+tanh processing finished.")

"""
NOTES:
- Each output TIF is georeferenced, float32, with values in [-1,1].
- Output is suitable for use as features in spatial ML or graph-based noise mapping.
- If handling extremely large rasters, block processing may be required (not included here).
"""
