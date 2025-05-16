# Chapter 4: CNN-Based Urban Noise Prediction using Multispectral Remote Sensing (EfficientNet)

This directory contains the implementation and data preparation pipeline for the methodology described in **Chapter 4** of the dissertation:  
**"Predicting Urban Noise Levels Using EfficientNet and Multispectral Remote Sensing Data: A Case Study of Southampton"**

---

## 📌 Overview

This study explores the predictive power of high-resolution multispectral remote sensing imagery for modeling urban noise levels. A convolutional neural network (CNN) based on the EfficientNet architecture is used to predict noise values from spatial-spectral patterns extracted around each measurement point.

The model is trained on data from the city of Southampton, using 84-layer feature maps derived from remote sensing imagery and labeled with ground-truth noise measurements (in dB LAeq). This experiment serves as a foundational proof-of-concept for integrating spectral and spatial information in urban noise modeling.

---

## 🛰️ Data Preparation Pipeline

### 1. **Feature Map Generation**

- **Data Source**: WorldView-2 multispectral imagery (8 bands)
- **Tools Used**:  
  - **ENVI 5.3**: for basic radiometric and atmospheric corrections  
  - **Orfeo Toolbox**: for extracting texture and morphological features

- **Total Feature Maps**: `84`
  - Includes:
    - Grey-Level Co-occurrence Matrix (GLCM) features (contrast, homogeneity, entropy, etc.)
    - Morphological transformations
    - Spectral indices and band statistics

### 2. **Sample Extraction**

- **Spatial Reference**: Noise measurement points (1m resolution)
- **Patch Size**: `250 × 250 pixels` (≈500m × 500m at 2m resolution)
- **Feature Tensor Shape**: `250 × 250 × 84`
- **Label**: Measured noise value at the central point

Each sample thus consists of a 3D tensor (`H × W × C`) and a scalar noise label (`LAeq` in dB).

---

## 🧠 Model Architecture: EfficientNetB0 + Regression haed

- Based on EfficientNet-B1, modified to accept input tensors of size `[250, 250, 84]`
- Custom convolutional input layer to adapt to 84-channel input
- Regression output for continuous noise prediction
- Loss Function: Huber Loss 
- Optimizer: Adam

---

## 🚀 Training Procedure

1. Load all samples into memory or use batch-wise HDF5/TFRecord pipeline
2. Split into training, validation and test sets (e.g., 60%/20%/20%)
3. Apply data augmentation (optional)
4. Train EfficientNet-B0 with modified input channel size
5. Track performance using validation MAE and R²

---

## 📊 Prediction & Evaluation

- Predict on test/validation noise points
- Compare predicted noise levels with ground truth
- Evaluate using:
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)

You may also visualize spatial noise distributions by stitching predictions back to spatial maps.

---

## 📁 Folder Structure

