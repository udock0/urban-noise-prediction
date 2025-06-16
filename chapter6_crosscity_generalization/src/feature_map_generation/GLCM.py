"""
GLCM.py

Feature map generator for multispectral remote sensing images:
  - Extracts Haralick (GLCM) texture features: contrast, homogeneity, energy, correlation
  - Supports single bands or remote sensing indices (e.g., NDVI, RGI)
  - Output: 4 GLCM feature maps per band, saved as GeoTIFF with georeference.

NOTE: This script is suitable for grid-based noise mapping and machine learning workflows.
      Please cite the dissertation if used in publications.

Author: [Your Name]
Date: 2025-06-16
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import glob
import mahotas as mh

#############################################
# Band normalization and quantization utilities

def z_score_normalize(band):
    """Standardize input band with z-score normalization."""
    band = band.astype(np.float32)
    mean = np.mean(band)
    std = np.std(band)
    return (band - mean) / (std + 1e-8)

def minmax_normalize(band):
    """Linearly scale image to [0,1]."""
    b_min = band.min()
    b_max = band.max()
    return (band - b_min) / (b_max - b_min + 1e-8)

def quantize_image(band, levels=32):
    """Quantize normalized band to integer levels [0, levels-1]."""
    quantized = np.floor(band * (levels - 1)).astype(np.uint8)
    return quantized

def tanh_normalize(data):
    """Z-score then apply tanh for [-1,1] output (for contrast feature)."""
    data_std = (data - np.mean(data)) / (np.std(data) + 1e-8)
    return np.tanh(data_std)

#############################################
# Haralick GLCM feature computation (mean of four directions)

def compute_haralick_features(window, levels=32):
    """
    Compute Haralick GLCM features for a window.
    Returns: contrast, homogeneity, energy, correlation
    """
    window = window.astype(np.uint8)
    feats = mh.features.haralick(window, return_mean=True)
    contrast = feats[1]
    homogeneity = feats[4]
    energy = np.sqrt(feats[0])
    correlation = feats[2]
    return contrast, homogeneity, energy, correlation

#############################################
# Sliding window GLCM feature extraction for an image

def create_haralick_feature_maps(band, window_size=16, step=8, levels=32):
    """
    Compute Haralick features in a sliding window across the band.
    Returns 4 feature maps with reduced spatial resolution.
    Output shape: ((H-window_size)//step+1, (W-window_size)//step+1)
    """
    H, W = band.shape
    n_rows = (H - window_size) // step + 1
    n_cols = (W - window_size) // step + 1

    contrast_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    homogeneity_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    energy_map = np.zeros((n_rows, n_cols), dtype=np.float32)
    correlation_map = np.zeros((n_rows, n_cols), dtype=np.float32)

    for i in range(n_rows):
        for j in range(n_cols):
            r = i * step
            c = j * step
            window = band[r:r+window_size, c:c+window_size]
            contrast, homogeneity, energy, correlation = compute_haralick_features(window, levels=levels)
            contrast_map[i, j] = contrast
            homogeneity_map[i, j] = homogeneity
            energy_map[i, j] = energy
            correlation_map[i, j] = correlation
    return contrast_map, homogeneity_map, energy_map, correlation_map

#############################################
# Save a feature map as a GeoTIFF with correct geotransform

def save_feature_map_as_tif(feature_map, output_path, base_transform, base_crs, step_factor=1):
    """
    Save a 2D feature map as GeoTIFF, adjusting pixel size according to step_factor.
    """
    pixel_width = base_transform.a
    pixel_height = -base_transform.e  # usually negative
    new_pixel_width = pixel_width * step_factor
    new_pixel_height = pixel_height * step_factor
    new_transform = from_origin(base_transform.c, base_transform.f, new_pixel_width, new_pixel_height)
    profile = {
        'driver': 'GTiff',
        'height': feature_map.shape[0],
        'width': feature_map.shape[1],
        'count': 1,
        'dtype': feature_map.dtype,
        'crs': base_crs,
        'transform': new_transform,
    }
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(feature_map, 1)
    print(f"Saved feature map to {output_path}")

#############################################
# Process one band and output feature maps

def process_single_band(file_path, band_index=1, levels=32, window_size=16, step=8):
    with rasterio.open(file_path) as src:
        band = src.read(band_index).astype(np.float32)
        base_transform = src.transform
        base_crs = src.crs
    norm_band = z_score_normalize(band)
    norm_mm = minmax_normalize(norm_band)
    quant_band = quantize_image(norm_mm, levels=levels)
    glcm_maps = create_haralick_feature_maps(quant_band, window_size=window_size, step=step, levels=levels)

    # Optional: tanh normalization for contrast
    contrast_map = tanh_normalize(glcm_maps[0])
    glcm_maps = (contrast_map, glcm_maps[1], glcm_maps[2], glcm_maps[3])

    return norm_band, quant_band, glcm_maps, base_transform, base_crs

#############################################
# Batch process multispectral bands

def process_multispectral(multispectral_path, out_folder, band_mapping, levels=32, window_size=16, step=8):
    """
    Generate and save GLCM features for selected bands.
    band_mapping: {band_index: output_prefix}, e.g. {7:"NIR1", 3:"Green", 6:"RedEdge"}
    """
    os.makedirs(out_folder, exist_ok=True)
    with rasterio.open(multispectral_path) as src:
        base_transform = src.transform
        base_crs = src.crs

    for b_index, prefix in band_mapping.items():
        with rasterio.open(multispectral_path) as src:
            if b_index < 1 or b_index > src.count:
                print(f"Band {b_index} out of range (1~{src.count}), skipping {prefix}.")
                continue
            band = src.read(b_index).astype(np.float32)
        print(f"Processing {prefix} (band {b_index}) ...")
        norm_band, quant_band, glcm_maps, base_transform, base_crs = process_single_band(
            multispectral_path, band_index=b_index, levels=levels, window_size=window_size, step=step
        )
        step_factor = step
        feature_names = ["contrast", "homogeneity", "energy", "correlation"]
        for feat, name in zip(glcm_maps, feature_names):
            out_path = os.path.join(out_folder, f"{prefix}_{name}.tif")
            save_feature_map_as_tif(feat, out_path, base_transform, base_crs, step_factor=step_factor)
        # Optionally save quantized band as reference
        # out_path_quant = os.path.join(out_folder, f"{prefix}_quant.tif")
        # save_feature_map_as_tif(quant_band, out_path_quant, base_transform, base_crs, step_factor=1)

#############################################
# Batch process index files (NDVI, WVBI, RGI)

def process_indices_folder(indices_folder, out_folder, levels=32, window_size=16, step=8):
    os.makedirs(out_folder, exist_ok=True)
    valid_names = {"NDVI", "WVBI", "RGI"}
    tif_files = glob.glob(os.path.join(indices_folder, "*.tif"))
    for tif_file in tif_files:
        basename = os.path.splitext(os.path.basename(tif_file))[0]
        if basename.upper() not in valid_names:
            print(f"Skipping {basename}, not a required index.")
            continue
        print(f"Processing index file: {basename}")
        norm_band, quant_band, glcm_maps, base_transform, base_crs = process_single_band(
            tif_file, band_index=1, levels=levels, window_size=window_size, step=step
        )
        step_factor = step
        feature_names = ["contrast", "homogeneity", "energy", "correlation"]
        for feat, name in zip(glcm_maps, feature_names):
            out_path = os.path.join(out_folder, f"{basename}_{name}.tif")
            save_feature_map_as_tif(feat, out_path, base_transform, base_crs, step_factor=step_factor)
        # Optionally save quantized band
        # out_path_quant = os.path.join(out_folder, f"{basename}_quant.tif")
        # save_feature_map_as_tif(quant_band, out_path_quant, base_transform, base_crs, step_factor=1)

#############################################
# Example main process (edit paths for your city/dataset)

if __name__ == "__main__":
    # Multispectral image path (at least 8 bands)
    multispectral_path = "D:/PhD dissertation/soton/raster/WV2_in_Soton_bd_4m.tif"
    out_folder_features = "D:/PhD dissertation/soton/feature_map"
    # Band mapping: {band_index: short_name}
    band_mapping = {7: "NIR1", 3: "Green", 6: "RedEdge"}
    process_multispectral(multispectral_path, out_folder_features, band_mapping, levels=32, window_size=16, step=8)
    
    # Process indices (NDVI, WVBI, RGI)
    indices_folder = "D:/PhD dissertation/soton/rsindex"
    process_indices_folder(indices_folder, out_folder_features, levels=32, window_size=16, step=8)
    
    print("All feature maps generated successfully.")

"""
NOTES:
- All outputs inherit georeference from source image, with pixel size scaled by window step.
- Edit main() for other cities/paths.
- Features are suitable for node attributes or edge construction in urban graph models.
"""
