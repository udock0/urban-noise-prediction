# urban-noise-prediction
Code and models for multispectral and GNN-based urban noise prediction across UK cities



# Urban Noise Prediction via Remote Sensing and Graph Learning

This repository contains the full implementation of the methods described in my PhD dissertation, which explores scalable and transferable models for predicting urban noise levels using high-resolution multispectral satellite imagery and graph-based machine learning.

## 📚 Project Structure

- `chapter4_efficientnet/` - Remote sensing-based CNN model (EfficientNet) for noise prediction in Southampton.
- `chapter5_gnn_southampton/` - Graph neural network model integrating spatial structure and urban context (Southampton only).
- `chapter6_crosscity_generalization/` - Generalizable noise prediction across five UK cities using domain adaptation and pseudo-labelling.

## 🌐 Colab Access

Each sub-folder contains a Colab-ready `.ipynb` notebook with detailed annotations. You can open them directly in Google Colab via the button below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](link-to-your-colab-notebook)

## 🛠 Requirements

Install dependencies locally (optional, for running outside Colab):

```bash
pip install -r requirements.txt
