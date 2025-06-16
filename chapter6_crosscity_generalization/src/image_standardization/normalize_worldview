"""
normalize_worldview.py

Image normalization workflow for cross-city remote sensing transfer in urban noise prediction.
NOTE: Input LULC shapefiles must be pre-processed using Dissolve on 'code_2012',
      so each class is a single multipart polygon, ensuring category-level statistics.

Author: [Your Name]
Date: 2025-06-16

Usage:
    # Edit paths and uncomment for your city of interest in the main process.
    # Recommended for reproducibility in pseudo-label cross-city noise modeling.

"""

import numpy as np
import rasterio
from rasterio.mask import mask
from scipy.interpolate import interp1d
import geopandas as gpd

# --------- Core Functions ---------

def extract_pif_pixels(image_path, shapefile, land_use_codes):
    """
    Extracts PIF region pixels from input raster and LULC vector, organized by band and class.

    Input:
        image_path: path to remote sensing raster (e.g. WorldView-2)
        shapefile: path to dissolved LULC polygons (each code_2012 class = one feature)
        land_use_codes: list of code_2012 strings
    Returns:
        dict {land_use_code: list of np.ndarray by band}
    """
    with rasterio.open(image_path) as src:
        vector_data = gpd.read_file(shapefile)
        pif_pixels = {code: [[] for _ in range(src.count)] for code in land_use_codes}

        for code in land_use_codes:
            polygons = vector_data[vector_data['code_2012'] == code].geometry
            if not polygons.empty:
                masked_data, _ = mask(src, polygons, crop=False)
                for band_idx in range(masked_data.shape[0]):
                    valid_pixels = masked_data[band_idx][(~np.isnan(masked_data[band_idx])) & (masked_data[band_idx] > 0)]
                    pif_pixels[code][band_idx].extend(valid_pixels)

    # Convert to np.array for downstream processing
    for code in pif_pixels:
        pif_pixels[code] = [np.array(band_pixels, dtype=np.float32) for band_pixels in pif_pixels[code]]
    return pif_pixels

def normalize_bandwise(data, band_min, band_max, target_min=0, target_max=1):
    """
    Min-max normalization for each band, mapping values to [target_min, target_max].
    """
    normalized_data = np.empty_like(data, dtype=np.float32)
    for band_idx in range(data.shape[0]):
        band = data[band_idx]
        normalized_band = (band - band_min[band_idx]) / (band_max[band_idx] - band_min[band_idx])
        normalized_band = normalized_band * (target_max - target_min) + target_min
        normalized_data[band_idx] = np.clip(normalized_band, target_min, target_max)
    return normalized_data

def calculate_bandwise_percentiles(image_path, shapefile, land_use_codes, percentiles=[2, 98]):
    """
    Calculate global PIF percentiles for each band in the image.
    Used for robust linear stretching.
    """
    with rasterio.open(image_path) as src:
        vector_data = gpd.read_file(shapefile)
        bandwise_percentiles = []

        for band_idx in range(src.count):
            band_pixels = []
            for code in land_use_codes:
                polygons = vector_data[vector_data['code_2012'] == code].geometry
                if not polygons.empty:
                    masked_data, _ = mask(src, polygons, crop=False)
                    valid_pixels = masked_data[band_idx]
                    valid_pixels = valid_pixels[(~np.isnan(valid_pixels)) & (valid_pixels > 0)]
                    band_pixels.extend(valid_pixels)
            if len(band_pixels) == 0:
                raise ValueError(f"Band {band_idx + 1}: no valid PIF pixels.")
            band_percentiles = np.percentile(band_pixels, percentiles)
            bandwise_percentiles.append(band_percentiles)
        return bandwise_percentiles

def linear_stretch_bandwise(target_image_path, ref_pif_extrema, target_bandwise_percentiles, output_path):
    """
    Apply linear stretching on the target image based on PIF extrema and quantiles.
    """
    with rasterio.open(target_image_path) as src:
        data = src.read().astype(np.float32)
        for band_idx in range(src.count):
            ref_min, ref_max = ref_pif_extrema[band_idx]
            target_min, target_max = target_bandwise_percentiles[band_idx]
            # Stretch
            scale = (ref_max - ref_min) / (target_max - target_min)
            offset = ref_min - scale * target_min
            stretched_band = scale * data[band_idx] + offset
            stretched_band = np.clip(stretched_band, ref_min, ref_max)
            stretched_band[stretched_band < 1.0] = 1.0
            data[band_idx] = stretched_band
        meta = src.meta
        meta.update(dtype=rasterio.float32)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)
    return data

# --------- Main Workflow Example ---------

if __name__ == "__main__":
    # --- IMPORTANT: LULC shapefiles must be dissolved by code_2012 before use! ---
    # Example for Portsmouth; switch comments for other cities as needed.
    reference_image = "D:/PhD dissertation/soton/raster/WV2_in_Soton_bd_4m.tif"
    reference_shapefile = "D:/PhD dissertation/PIF/PIF_soton_4o_Dissolve.shp"

    # Uncomment below to process other cities:
    # target_image = "D:/PhD dissertation/portsmouth/raster/WV2_in_Portsmouth_bd_4m.tif"
    # target_shapefile = "D:/PhD dissertation/PIF/PIF_portsmouth_4o_Dissolve.shp"
    # stretched_image = "D:/PhD dissertation/portsmouth/raster/WV2_in_Portsmouth_stretched.tif"
    # output_image = "D:/PhD dissertation/portsmouth/raster/WV2_in_Portsmouth_final.tif"
    #
    # # liverpool
    # target_image = ...
    # target_shapefile = ...
    # ...
    # # cardiff
    # # nottingham

    land_use_codes = ['11100', '11210', '11220', '12100', '12210', '12220', '12230']

    print("Extracting PIF pixels...")
    ref_pif_pixels = extract_pif_pixels(reference_image, reference_shapefile, land_use_codes)
    target_pif_pixels = extract_pif_pixels(target_image, target_shapefile, land_use_codes)

    # Compute PIF extrema (min, max per band)
    ref_pif_extrema = []
    for band_pixels in zip(*ref_pif_pixels.values()):
        band_data = np.concatenate([np.array(pixels) for pixels in band_pixels if len(pixels) > 0])
        ref_pif_extrema.append((np.min(band_data), np.max(band_data)))

    target_bandwise_percentiles = calculate_bandwise_percentiles(target_image, target_shapefile, land_use_codes)

    print("\nPerforming bandwise linear stretching...")
    stretched_data = linear_stretch_bandwise(target_image, ref_pif_extrema, target_bandwise_percentiles, stretched_image)

    # More advanced histogram matching and evaluation can be added as extra steps

    print("Processing complete.")

"""
NOTE:
- Dissolve your LULC vector data using 'code_2012' BEFORE running this code,
  so that each land use class is a single feature for robust masking/statistics.
- Uncomment appropriate city blocks in the main section to process each city.
"""
