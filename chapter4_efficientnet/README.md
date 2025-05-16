# Chapter 4: CNN-Based Urban Noise Prediction using Multispectral Remote Sensing (EfficientNet)

This directory contains the full implementation of the study described in **Chapter 4** of the dissertation:
**"Predicting Urban Noise Levels Using EfficientNet and Multispectral Remote Sensing Data: A Case Study of Southampton"**.

---

## 📌 Overview

This chapter presents a deep learning pipeline that predicts urban noise levels directly from remote sensing imagery. The method utilizes an EfficientNet-B0 backbone, adapted to handle high-dimensional multispectral feature maps. The model is trained to learn spatial-spectral patterns from 84-channel image patches centered on field-collected noise measurements.

---

## 🚀 Data Preprocessing and Input Design

### 1. Feature Map Construction

* Source: WorldView-2 multispectral imagery
* Tools: ENVI 5.3 and Orfeo Toolbox
* Output: 84-channel feature maps combining spectral, textural, and morphological features

### 2. Patch Extraction

* Noise samples used as center points
* Patch size: `250 × 250 × 84`, corresponding to \~500 m spatial coverage
* Format: Converted to TFRecords for training and prediction

---

## 🧠 Model Architecture

### EfficientNet-B0 with Decay Matrix

* Input: `(250, 250, 84)` image tensors
* A spatial decay matrix is applied to emphasize the center region and reduce edge influence
* Pretrained EfficientNet-B0 backbone (weights initialized from scratch)
* Global average pooling followed by two fully connected layers (256, 64 units)
* Output: A single scalar representing normalized noise level (`sigmoid`, scaled back post-prediction)

---

## ⚙️ Training Pipeline

### Key Techniques

* Label normalization to [0,1] range based on known noise bounds
* Loss function: Huber loss, optionally with penalty for large errors
* Metrics: MAE and R²
* Optimizer: Adam with learning rate decay and gradient clipping
* Mixed precision training enabled for performance

### Execution

* Training and validation datasets loaded from TFRecord files
* Caching and GPU prefetching used to improve speed
* Model checkpoints and training curves are automatically saved and visualized

---

## 🔮 Prediction Workflow

* Predictions are generated for new image patches stored as TFRecords
* Coordinate normalization is applied for spatial reference
* Outputs are denormalized and joined with shapefiles using `geopandas`
* The final predicted maps are exported as shapefiles (`.shp`) for GIS visualization

---

## 📁 File Structure

```
chapter4_efficientnet/
├── notebook_colab.ipynb         ← Unified pipeline notebook (training + prediction)
├── Train.py                     ← Full training script with custom loss, model, and callbacks
├── Prediction.py                ← Prediction and export script using TFRecord + GeoPandas
├── feature_maps/                ← Optional folder for storing 84-band input features
├── trained_models/              ← Saved EfficientNet models (.h5)
├── results/                     ← Evaluation metrics and visualization outputs
└── README.md                    ← This file
```

---

## 🤪 Example Usage

To run this pipeline in Google Colab:

```python


```

---

## 📜 Citation

> Zhu, F. (2025). *Predicting Urban Noise Levels Using EfficientNet and Multispectral Remote Sensing Data: A Case Study of Southampton*. Chapter 4 in PhD Dissertation, University of Southampton.

---

## 📬 Contact

Feiyu Zhu – (5835udock0@gmail.com)]
