"""
morphology_features.py

Morphological feature extraction for remote sensing images:
  - Local Binary Pattern (LBP)
  - Morphological profiles (opening/closing by disk structuring element)

Supports both single band (Red, Green, NIR1) and remote sensing index images (NDVI, BSI).
All outputs are saved as GeoTIFFs with proper georeference for downstream ML or GIS analysis.

Author: [Your Name]
Date: 2025-06-16
"""

import os
import numpy as np
import rasterio
from skimage.feature import local_binary_pattern
from skimage.morphology import disk, opening, closing
import glob

# ===============================
# --- Band normalization utils ---
# ===============================
def z_score_normalize(band):
    """Z-score normalization for a single band."""
    band = band.astype(np.float32)
    mean = np.mean(band)
    std = np.std(band)
    return (band - mean) / (std + 1e-8)

def minmax_normalize(band):
    """Min-max normalization to [0,1]."""
    b_min = band.min()
    b_max = band.max()
    return (band - b_min) / (b_max - b_min + 1e-8)

# ================================
# --- Save array as GeoTIFF file --
# ================================
def save_single_band(band_arr, out_path, ref_profile, dtype='float32'):
    """
    Save array as GeoTIFF, filling invalid pixels and clipping to [0,1].
    """
    valid_mask = (band_arr != 0) & (~np.isnan(band_arr))
    valid_min = band_arr[valid_mask].min() if valid_mask.sum() > 0 else 0
    band_arr_filled = np.where(valid_mask, band_arr, valid_min)
    band_arr_clipped = np.clip(band_arr_filled, 0, 1)

    profile_out = ref_profile.copy()
    profile_out.update({'count': 1, 'dtype': dtype})

    print(f"Saving {out_path}: shape={band_arr_clipped.shape}, min={band_arr_clipped.min():.4f}, max={band_arr_clipped.max():.4f}")
    try:
        with rasterio.open(out_path, 'w', **profile_out) as dst:
            dst.write(band_arr_clipped, 1)
        print(f"Saved => {out_path}")
    except Exception as e:
        print(f"ERROR: Failed to save {out_path} -> {e}")

# ========================================
# --- Morphological/LBP feature extract ---
# ========================================
def generate_lbp_feature(band_arr, radius=2, n_points=16, method='uniform'):
    """
    Compute Local Binary Pattern (LBP) feature for a band.
    Returns normalized LBP map.
    """
    valid_mask = (band_arr != 0) & (~np.isnan(band_arr))
    if valid_mask.sum() == 0:
        raise ValueError("No valid data!")
    valid_min = band_arr[valid_mask].min()
    valid_max = band_arr[valid_mask].max()
    band_arr_filled = np.where(valid_mask, band_arr, valid_min)
    band_norm = (band_arr_filled - valid_min) / (valid_max - valid_min + 1e-8)
    band_uint8 = (band_norm * 255).astype(np.uint8)
    lbp_map = local_binary_pattern(band_uint8, n_points, radius, method=method)
    lbp_min = np.nanmin(lbp_map)
    lbp_map = np.where(np.isnan(lbp_map), lbp_min, lbp_map)
    lbp_min, lbp_max = lbp_map.min(), lbp_map.max()
    if lbp_max > lbp_min:
        lbp_map = (lbp_map - lbp_min) / (lbp_max - lbp_min + 1e-8)
    print(f"LBP computed: min={lbp_map.min():.4f}, max={lbp_map.max():.4f}")
    return lbp_map.astype(np.float32)

def generate_morphological_profiles(band_arr, radius=5):
    """
    Compute morphological opening and closing (by disk) for a band.
    Returns dict with normalized results.
    """
    valid_mask = (band_arr != 0) & (~np.isnan(band_arr))
    if valid_mask.sum() == 0:
        raise ValueError("No valid data!")
    valid_min = band_arr[valid_mask].min()
    valid_max = band_arr[valid_mask].max()
    band_arr_filled = np.where(valid_mask, band_arr, valid_min)
    band_norm = (band_arr_filled - valid_min) / (valid_max - valid_min + 1e-8)
    band_uint8 = (band_norm * 255).astype(np.uint8)
    selem = disk(radius)
    opened = opening(band_uint8, selem).astype(np.float32)
    closed = closing(band_uint8, selem).astype(np.float32)

    # Normalize opened/closed
    for arr, name in zip([opened, closed], ["open", "close"]):
        arr_min, arr_max = arr.min(), arr.max()
        norm_arr = (arr - arr_min) / (arr_max - arr_min + 1e-8) if arr_max > arr_min else arr
        if name == "open":
            opened = norm_arr
        else:
            closed = norm_arr

    print(f"MP computed: open min={opened.min():.4f}, max={opened.max():.4f} | close min={closed.min():.4f}, max={closed.max():.4f}")
    return {f"open_r{radius}": opened, f"close_r{radius}": closed}

# ==============================================
# --- Batch process bands for morphological maps
# ==============================================
def process_features_for_bands(multispectral_path, out_dir,
                               band_list=[(5, "Red"), (3, "Green"), (7, "NIR1")],
                               lbp_radius=2, lbp_points=16, mp_radius=5):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing multispectral image: {multispectral_path}")
    with rasterio.open(multispectral_path) as src:
        ms_profile = src.profile.copy()
    for (b_idx, b_name) in band_list:
        print(f"Processing {b_name} (Band {b_idx})")
        with rasterio.open(multispectral_path) as src:
            band_arr = src.read(b_idx).astype(np.float32)
        if ((band_arr == 0).all() or np.isnan(band_arr).all()):
            print(f"Skipping {b_name}: contains only 0 or NaN")
            continue
        # LBP
        lbp_map = generate_lbp_feature(band_arr, radius=lbp_radius, n_points=lbp_points)
        save_single_band(lbp_map, os.path.join(out_dir, f"LBP_{b_name}.tif"), ms_profile)
        # Morphological open/close
        mp_results = generate_morphological_profiles(band_arr, radius=mp_radius)
        for key, arr in mp_results.items():
            save_single_band(arr, os.path.join(out_dir, f"MP_{b_name}_{key}.tif"), ms_profile)
    print(f"Multispectral bands processed: results saved in {out_dir}")

# ====================================================
# --- Batch process indices (NDVI, BSI) for morph maps
# ====================================================
def process_features_for_indices(indices_folder, out_dir, lbp_radius=2, lbp_points=16, mp_radius=5):
    os.makedirs(out_dir, exist_ok=True)
    valid_names = {"NDVI", "BSI"}
    tif_files = glob.glob(os.path.join(indices_folder, "*.tif"))
    for tif_file in tif_files:
        basename = os.path.splitext(os.path.basename(tif_file))[0]
        if basename.upper() not in valid_names:
            print(f"Skipping file {basename}, not in {valid_names}.")
            continue
        print(f"Processing index image: {basename}")
        with rasterio.open(tif_file) as src:
            index_profile = src.profile.copy()
            index_data = src.read(1).astype(np.float32)
        if ((index_data == 0).all() or np.isnan(index_data).all()):
            print(f"Skipping {basename}: contains only 0 or NaN")
            continue
        lbp_index = generate_lbp_feature(index_data, radius=lbp_radius, n_points=lbp_points)
        save_single_band(lbp_index, os.path.join(out_dir, f"LBP_{basename}.tif"), index_profile)
        mp_index = generate_morphological_profiles(index_data, radius=mp_radius)
        for key, arr in mp_index.items():
            save_single_band(arr, os.path.join(out_dir, f"MP_{basename}_{key}.tif"), index_profile)
    print(f"Indices processed: results saved in {out_dir}")

# ==================================================
# --- Batch demo for all cities and datasets -------
# ==================================================
if __name__ == "__main__":
    cities = [
        ("soton", "D:/PhD dissertation/soton/raster/WV2_in_Soton_bd_4m.tif", "D:/PhD dissertation/soton/lmp/", "D:/PhD dissertation/soton/rsindex/"),
        ("portsmouth", "D:/PhD dissertation/portsmouth/raster/WV2_in_Portsmouth_final.tif", "D:/PhD dissertation/portsmouth/lmp/", "D:/PhD dissertation/portsmouth/rsindex/"),
        ("liverpool", "D:/PhD dissertation/liverpool/raster/WV2_in_Liverpool_final.tif", "D:/PhD dissertation/liverpool/lmp/", "D:/PhD dissertation/liverpool/rsindex/"),
        ("cardiff", "D:/PhD dissertation/cardiff/raster/WV2_in_Cardiff_final.tif", "D:/PhD dissertation/cardiff/lmp/", "D:/PhD dissertation/cardiff/rsindex/"),
        ("nottingham", "D:/PhD dissertation/nottingham/raster/WV2_in_Nottingham_final.tif", "D:/PhD dissertation/nottingham/lmp/", "D:/PhD dissertation/nottingham/rsindex/"),
    ]
    bands = [(5, "Red"), (3, "Green"), (7, "NIR1")]
    for city, mspath, outdir, idxdir in cities:
        print(f"\n=== Processing {city.upper()} ===")
        process_features_for_bands(mspath, outdir, band_list=bands, lbp_radius=2, lbp_points=16, mp_radius=5)
        process_features_for_indices(idxdir, outdir, lbp_radius=2, lbp_points=16, mp_radius=5)
    print("\nAll cities and indices finished.")
