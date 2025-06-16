#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_generation.py

This script generates an environmental feature dataset for a set of vector points:
 1) Filters points fitting full windows in a reference raster and computes pixel coordinates.
 2) Assigns grid row and column indices based on a uniform cell size.
 3) Computes ring-based statistics on multiple raster layers.
 4) Computes ring-based land-use/cover (LULC) ratios, distance metrics, noise exposure index (NEI), and GNN edge weights.
 5) Merges all feature tables, filters invalid rows, and outputs a final CSV for model training.

Usage:
    python dataset_generation.py

Ensure all file paths and constants are configured before running.
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from skimage.measure import shannon_entropy
from shapely.geometry import Point
from shapely.affinity import translate
from shapely.prepared import prep
from rtree import index
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ------------------ CONFIGURATION ------------------
VECTOR_POINTS = r"D:/PhD dissertation/soton/vector/soton_30m_allnode.shp"
OUTPUT_DIR = r"D:/PhD dissertation/soton/vector"
RASTER_DIRS = [
    r"D:/PhD dissertation/soton/zscore",
    r"D:/PhD dissertation/soton/lmp",
    r"D:/PhD dissertation/soton/glc"
]
CELL_SIZE = 30.0  # meters
WINDOW_RADIUS = 500  # pixels for raster window check
INNER_RADII = [0, 30, 60, 120, 500]
OUTER_RADII = [30, 60, 120, 500, 1000]
RESOLUTION = 4  # meters per pixel for ring templates

# LULC configuration
LULC_VECTOR = r"D:/PhD dissertation/liverpool/vector/UrbanAtlas_in_liverpool_Dissolve.shp"
FIXED_LULC_CODES = [
    '11100','11210','11220','11230','11240','12210','14100','12100',
    '12220','23000','12300','12230','14200','13300','31000','21000',
    '12400','13100','13400','40000','50000','32000','33000','11300'
]
LULC_CLASS = {
    '12230':'noise_source','12300':'noise_source','12210':'noise_source','13300':'noise_source','12100':'noise_source','12400':'noise_source','13100':'noise_source',
    '12220':'secondary_noise_source',
    '14100':'noise_mitigation','31000':'noise_mitigation','14200':'noise_mitigation','23000':'noise_mitigation','21000':'noise_mitigation','40000':'noise_mitigation','50000':'noise_mitigation','13400':'noise_mitigation',
    '32000':'secondary_noise_mitigation','33000':'secondary_noise_mitigation','11300':'secondary_noise_mitigation'
}
HIGH_NOISE_CODES = ['12230','12300','12210','13300','12100','12400','13100']
SECONDARY_NOISE_CODES = ['12220']
LOW_NOISE_CODES = ['14100','31000','14200','23000','21000','40000','50000','13400']
SECONDARY_MITIGATION_CODES = ['32000','33000','11300']
DEFAULT_WEIGHT = 1.0
DISTANCE_FILL = 9999.0
MAX_SEARCH_RADIUS = 1000.0  # meters

CURRENT_CITY = 'Liverpool'

# Output dataset
DATASET_DIR = r"D:/PhD dissertation/liverpool/dataset"
FINAL_OUTPUT_CSV = os.path.join(DATASET_DIR, "GNN_pred_liverpool.csv")

# ------------------ STEP 1: FILTER VALID POINTS ------------------
def get_valid_points(vector_path, raster_path, window_radius):
    """
    Read vector points, retain those whose surrounding square window of given radius fits inside raster,
    and compute pixel_x, pixel_y for each.
    Returns GeoDataFrame with added pixel_x, pixel_y.
    """
    pts = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as rst:
        height, width = rst.height, rst.width
        valid = []
        for feat in pts.itertuples():
            px_y, px_x = rst.index(feat.geometry.x, feat.geometry.y)
            half = window_radius // 2
            if (px_y-half >=0 and px_y+half < height and px_x-half >=0 and px_x+half < width):
                rec = feat._asdict()
                rec['pixel_x'], rec['pixel_y'] = px_x, px_y
                valid.append(rec)
    gdf = gpd.GeoDataFrame(valid, geometry=[r['geometry'] for r in valid], crs=pts.crs)
    return gdf

# ------------------ STEP 2: ASSIGN GRID ROW/COL ------------------
def assign_rowcol(gdf, cell_size=CELL_SIZE, invert_y=True):
    """
    Assign integer 'row' and 'col' based on spatial extents and uniform cell size.
    """
    xs = gdf.geometry.x.values
    ys = gdf.geometry.y.values
    minx, maxy = xs.min(), ys.max()
    gdf['col'] = np.round((xs - xs.min()) / cell_size).astype(int)
    if invert_y:
        gdf['row'] = np.round((maxy - ys) / cell_size).astype(int)
    else:
        gdf['row'] = np.round((ys - ys.min()) / cell_size).astype(int)
    return gdf

# ------------------ STEP 3: RASTER RING STATISTICS ------------------
stats_functions = {
    'mean': lambda arr: np.nanmean(arr).astype(np.float32),
    'std': lambda arr: np.nanstd(arr).astype(np.float32),
    'median': lambda arr: np.nanmedian(arr).astype(np.float32),
    'iqr': lambda arr: (np.nanpercentile(arr,75)-np.nanpercentile(arr,25)).astype(np.float32),
    'percentile10': lambda arr: np.nanpercentile(arr,10).astype(np.float32),
    'percentile25': lambda arr: np.nanpercentile(arr,25).astype(np.float32),
    'percentile75': lambda arr: np.nanpercentile(arr,75).astype(np.float32),
    'percentile90': lambda arr: np.nanpercentile(arr,90).astype(np.float32),
    'entropy': lambda arr: shannon_entropy(arr.astype(np.float32)),
    'weighted_mean': lambda data, wt: np.average(data, weights=wt).astype(np.float32) if data.size>0 else np.float32(np.nan)
}

def prepare_ring_masks(inner_radii, outer_radii, resolution=RESOLUTION):
    masks = {}
    for inn, out in zip(inner_radii, outer_radii):
        p = int(round(out/resolution))
        i = int(round(inn/resolution))
        size = p*2
        ys, xs = np.ogrid[:size, :size]
        dist = np.sqrt((ys-p)**2 + (xs-p)**2)
        mask = (dist <= p) & (dist >= i)
        wt = (1.0/(dist+1e-6)).astype(np.float32)
        masks[p] = (mask, wt)
    return masks


def calculate_ring_statistics(arr, px, py, inner_radii, outer_radii, masks, prefix):
    results = {}
    h, w = arr.shape
    for inn, out in zip(inner_radii, outer_radii):
        p = int(round(out/RESOLUTION))
        mask, wt = masks[p]
        top, left = py-p, px-p
        win = arr[top:top+mask.shape[0], left:left+mask.shape[1]]
        if win.shape == mask.shape:
            data = win[mask]
            weights = wt[mask]
        else:
            # dynamic mask
            ys, xs = np.ogrid[:win.shape[0], :win.shape[1]]
            dist = np.sqrt((ys-p)**2 + (xs-p)**2)
            m2 = (dist<=p)&(dist>=int(round(inn/RESOLUTION)))
            data = win[m2]
            weights = (1.0/(dist[m2]+1e-6)).astype(np.float32)
        for stat, fn in stats_functions.items():
            key = f"{prefix}_r{out}_{stat}"
            results[key] = fn(data) if stat!='weighted_mean' else fn(data, weights)
    return results


def process_raster_features(points_gdf, raster_path, masks):
    arr = rasterio.open(raster_path).read(1).astype(np.float32)
    prefix = os.path.splitext(os.path.basename(raster_path))[0]
    feats = []
    for pt in points_gdf.itertuples():
        feats.append(calculate_ring_statistics(arr, pt.pixel_x, pt.pixel_y, INNER_RADII, OUTER_RADII, masks, prefix))
    return pd.DataFrame(feats)

# ------------------ STEP 4: LULC FEATURES ------------------
def load_and_prepare_lulc(path):
    gdf = gpd.read_file(path)
    gdf['code'] = gdf['code_2012'].astype(str).str.strip()
    valid = gdf[gdf['code'].isin(FIXED_LULC_CODES)].copy()
    idx = index.Index()
    for i, geom in enumerate(valid.geometry): idx.insert(i, geom.bounds)
    return valid.reset_index(drop=True), idx


def calculate_landuse_ratios(vector_gdf, idx_tree, center_geom, templates, bounds, radius):
    ratios = {}
    for (inn,out), tmpl in templates.items():
        ring = translate(tmpl, xoff=center_geom.x, yoff=center_geom.y)
        b = bounds[(inn,out)]
        minx, miny, maxx, maxy = [b[i] + (center_geom.x if i%2==0 else center_geom.y) for i in range(4)]
        candidates = vector_gdf.iloc[list(idx_tree.intersection((minx,miny,maxx,maxy)))]
        if candidates.empty:
            for code in FIXED_LULC_CODES:
                ratios[f"landuse_{code}_r{out}"] = 0.0
            continue
        prep_ring = prep(ring)
        inter = candidates[candidates.geometry.apply(lambda g: prep_ring.intersects(g))]
        areas = inter.geometry.intersection(ring).area
        totals = areas.sum()
        for code in FIXED_LULC_CODES:
            area = areas[candidates['code']==code].sum() if totals>0 else 0.0
            ratios[f"landuse_{code}_r{out}"] = float(area/totals) if totals>0 else 0.0
    return ratios


def calculate_min_distances(vector_gdf, idx_tree, center_geom, codes):
    dists = {f"dist_{c}":DISTANCE_FILL for c in codes}
    minx,miny = center_geom.x-MAX_SEARCH_RADIUS, center_geom.y-MAX_SEARCH_RADIUS
    maxx,maxy = center_geom.x+MAX_SEARCH_RADIUS, center_geom.y+MAX_SEARCH_RADIUS
    cand = vector_gdf.iloc[list(idx_tree.intersection((minx,miny,maxx,maxy)))]
    for row in cand.itertuples():
        c = row.code
        if c not in codes: continue
        dist = center_geom.distance(row.geometry)
        key = f"dist_{c}"
        if dist<dists[key] and dist<=MAX_SEARCH_RADIUS:
            dists[key] = dist
    return dists


def calculate_noise_exposure_index(ratios, dists, out_radius):
    nei=0.0
    for code, ratio in {k.split('_')[1]:v for k,v in ratios.items() if k.endswith(f"_r{out_radius}")}.items():
        dist = dists.get(f"dist_{code}",DISTANCE_FILL)
        weight = (1.2 if code in HIGH_NOISE_CODES 
                  else 1.1 if code in SECONDARY_NOISE_CODES 
                  else 0.8 if code in LOW_NOISE_CODES 
                  else 0.9 if code in SECONDARY_MITIGATION_CODES 
                  else DEFAULT_WEIGHT)
        if dist<MAX_SEARCH_RADIUS:
            nei += (ratio/(dist+1e-5))*weight
    return nei


# Precompute LULC ring templates
LULC_TEMPLATES, LULC_BOUNDS = None, None

# ------------------ MAIN PROCESS ------------------
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # 1. Filter valid points
    sample_raster = glob.glob(os.path.join(RASTER_DIRS[-1],'*.tif'))[-1]
    pts = get_valid_points(VECTOR_POINTS, sample_raster, WINDOW_RADIUS)
    pts.to_file(os.path.join(OUTPUT_DIR,'valid_points_UTM.shp'))

    # 2. Assign row/col
    pts_rc = assign_rowcol(pts)
    pts_rc.to_file(os.path.join(OUTPUT_DIR,'valid_points_rowcol.shp'))

    # 3. Raster ring stats
    masks = prepare_ring_masks(INNER_RADII, OUTER_RADII)
    raster_dfs = []
    for rd in RASTER_DIRS:
        for rf in glob.glob(os.path.join(rd,'*.tif')):
            df_rf = process_raster_features(pts_rc, rf, masks)
            raster_dfs.append(df_rf)

    # 4. LULC features
    vector_lulc, idx_lulc = load_and_prepare_lulc(LULC_VECTOR)
    from shapely.geometry import Point
    # prepare ring polygons and bounds
    LULC_TEMPLATES = {}
    LULC_BOUNDS = {}
    for inn,out in zip(INNER_RADII, OUTER_RADII):
        ring = Point(0,0).buffer(out).difference(Point(0,0).buffer(inn))
        LULC_TEMPLATES[(inn,out)] = ring
        LULC_BOUNDS[(inn,out)] = ring.bounds

    lulc_rows = []
    for pt in tqdm(pts_rc.geometry, desc="LULC extraction"):
        ratios = calculate_landuse_ratios(vector_lulc, idx_lulc, pt, LULC_TEMPLATES, LULC_BOUNDS, OUTER_RADII[-1])
        dists = calculate_min_distances(vector_lulc, idx_lulc, pt, HIGH_NOISE_CODES+LOW_NOISE_CODES+SECONDARY_NOISE_CODES+SECONDARY_MITIGATION_CODES)
        nei = calculate_noise_exposure_index(ratios, dists, OUTER_RADII[-1])
        rec = {**ratios, **dists}
        lulc_rows.append(rec)
    lulc_df = pd.DataFrame(lulc_rows)

    # 5. Merge and filter
    merged = pd.concat([pts_rc.reset_index(drop=True)] + raster_dfs + [lulc_df], axis=1)
    # drop rows with NaN except in 'meandBA'
    cols = [c for c in merged.columns if c!='meandBA']
    merged = merged[merged[cols].notna().all(axis=1)].reset_index(drop=True)
    merged.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"Final dataset written to {FINAL_OUTPUT_CSV}, shape={merged.shape}")
