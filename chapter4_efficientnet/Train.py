# train_pipeline.py

#!/usr/bin/env python3
"""
train_pipeline.py

This script builds TFRecord-based datasets, defines an EfficientNetB0 model
for urban noise prediction, and trains it with mixed precision.
"""

import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError

# --- GPU & precision configuration ---
def configure_gpu_and_precision():
    """Enable GPU memory growth and set mixed precision policy."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy('mixed_float16')

# --- Data parsing & augmentation ---
def scale_label(label):
    """Normalize dBA labels to [0,1]."""
    return (label - 39.1) / (100.1 - 39.1)

def parse_tfrecord(example_proto):
    """
    Parse a single TFRecord containing:
      - 'image': raw bytes for a 250×250×84 float16 array
      - 'label': a float32 dBA value
    """
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32)
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    image = tf.io.decode_raw(parsed['image'], tf.float16)
    image = tf.reshape(image, [250, 250, 84])
    label = scale_label(parsed['label'])
    return image, label

def augment_data(image, label):
    """Add small Gaussian noise for data augmentation."""
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.01, dtype=tf.float32)
    image = tf.cast(image, tf.float32) + noise
    image = tf.clip_by_value(image, 0.0, 1.0)
    return tf.cast(image, tf.float16), label

def load_dataset(pattern, batch_size=8, shuffle=False, cache=False):
    """
    Build a tf.data.Dataset from TFRecord files matching `pattern`.
    - shuffle: whether to shuffle examples
    - cache:  whether to cache decoded records in memory
    """
    files = tf.data.Dataset.list_files(pattern, shuffle=shuffle)
    ds = files.interleave(tf.data.TFRecordDataset,
                          cycle_length=4,
                          num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=500)
    ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Model definition ---
def build_model():
    """
    Create an EfficientNetB0-based regression model.
    Input: 250×250×84 (float16)
    Output: single normalized dBA value (float32)
    """
    inp = Input(shape=(250, 250, 84), dtype=tf.float16)
    # Apply EfficientNet backbone (no pretrained weights)
    base = EfficientNetB0(include_top=False, weights=None, input_tensor=inp)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(1, activation='sigmoid', dtype=tf.float32)(x)
    return Model(inputs=inp, outputs=out)

# --- Training entry point ---
def main():
    configure_gpu_and_precision()

    # Update these path patterns to point at your TFRecord files
    train_pattern = '/path/to/train_*.tfrecord'
    val_pattern   = '/path/to/test_*.tfrecord'

    # Prepare datasets
    train_ds = load_dataset(train_pattern, shuffle=True, cache=True)
    val_ds   = load_dataset(val_pattern,   shuffle=False, cache=False)

    # Build and compile model
    model = build_model()
    optimizer = Adam(learning_rate=1e-3, clipvalue=1.0)
    model.compile(optimizer=optimizer,
                  loss=Huber(delta=1.2),
                  metrics=[MeanAbsoluteError()])

    # Callback to save best model by validation loss
    checkpoint = ModelCheckpoint('best_model.h5',
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    # Train
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=100,
              callbacks=[checkpoint])

    # Final save
    model.save('final_model.h5')

if __name__ == '__main__':
    main()
