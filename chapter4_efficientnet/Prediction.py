# predict_pipeline.py

#!/usr/bin/env python3
"""
predict_pipeline.py

Load a trained model, apply it to TFRecord files for prediction,
and export results to a GeoDataFrame/​shapefile.
"""

import os
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import geopandas as gpd

# --- Data parsing for prediction ---
def parse_tfrecord_predict(example_proto):
    """
    Parse a TFRecord with:
      - 'image': raw bytes for 250×250×84 float16 array
      - 'x','y' coordinates (float32) for each sample
    """
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'x':     tf.io.FixedLenFeature([], tf.float32),
        'y':     tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.io.decode_raw(parsed['image'], tf.float16)
    image = tf.reshape(image, [250, 250, 84])
    coords = (parsed['x'], parsed['y'])
    return image, coords

def load_predict_dataset(file_path, batch_size=8):
    """Return a batched dataset of images for a single TFRecord file."""
    ds = tf.data.TFRecordDataset(file_path)
    ds = ds.map(lambda x, c: parse_tfrecord_predict(x)[0], 
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def unscale_label(norm_label):
    """Convert normalized [0,1] label back to original dBA."""
    return norm_label * (100.1 - 39.1) + 39.1

# --- Prediction entry point ---
def main():
    # Path to your saved model
    model = tf.keras.models.load_model('best_model.h5')

    # Input shapefile holding point geometries (one per TFRecord sample)
    shp_path = '/path/to/valid_pred_points.shp'
    point_gdf = gpd.read_file(shp_path)

    # Directory containing pred_####.tfrecord files
    pred_dir = '/path/to/pred_dst'
    predictions = []

    # Iterate over all predicted TFRecord files
    for fname in tqdm(sorted(os.listdir(pred_dir))):
        if not fname.endswith('.tfrecord'):
            continue
        full_path = os.path.join(pred_dir, fname)
        ds = load_predict_dataset(full_path)
        preds = model.predict(ds)
        # Un-normalize and collect
        predictions.extend(unscale_label(preds).flatten())

    # Attach predictions to GeoDataFrame
    point_gdf['predicted_dBA'] = np.array(predictions)

    # Save to new shapefile
    out_path = '/path/to/output/predictions.shp'
    point_gdf.to_file(out_path)
    print(f"Saved predictions to {out_path}")

if __name__ == '__main__':
    main()
