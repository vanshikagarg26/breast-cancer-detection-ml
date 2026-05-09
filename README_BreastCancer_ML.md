# Breast Cancer Detection & Classification using Machine Learning and Deep Learning

[![IEEE Publication](https://img.shields.io/badge/IEEE-Published-blue)](https://ieeexplore.ieee.org/document/9417996)
[![Python](https://img.shields.io/badge/Python-3.7-green)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.1.0-orange)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.3.1-red)](https://keras.io/)

> **Published:** Automated Detection and Classification of Breast Cancer Tumour Cells using Machine Learning and Deep Learning on Histopathological Images — IEEE Conference Publication | [IEEE Xplore](https://ieeexplore.ieee.org/document/9417996)

---

## Overview

This project presents an end-to-end automated pipeline for the detection and classification of breast cancer tumour cells from histopathological images. Breast cancer is the second most commonly diagnosed cancer worldwide and a leading cause of mortality in women. Early and accurate detection dramatically improves survival outcomes.

The system automates the full diagnostic workflow — from image preprocessing and classification to tumour segmentation — removing the need for time-intensive manual analysis by pathologists.

---

## Problem Statement

Manual analysis of histopathological images is slow, costly, and subject to inter-observer variability. This project addresses the challenge of automatically classifying tumour cells as **malignant** or **benign** and segmenting the tumorous regions to support early clinical decision-making.

---

## Pipeline

```
Histopathological Image Dataset
           ↓
  Image Preprocessing & Enhancement
           ↓
  PCA — Dimensionality Reduction (18 dimensions retained)
           ↓
  ┌────────────────────────────────┐
  │  Classification                │
  │  ├── Support Vector Machine    │
  │  └── Convolutional Neural Net  │
  └────────────────────────────────┘
           ↓
  ┌────────────────────────────────┐
  │  Segmentation                  │
  │  ├── K-Means Clustering        │
  │  └── Genetic Algorithm         │
  └────────────────────────────────┘
           ↓
  Evaluation — Accuracy, Precision, Recall, F1, Sensitivity, Specificity
```

---

## Algorithms Used

### Classification
| Algorithm | Description |
|-----------|-------------|
| **Support Vector Machine (SVM)** | Supervised classifier using hyperplane separation in high-dimensional space with kernel transformation |
| **Convolutional Neural Network (CNN)** | Deep learning model with Convolution, ReLU, Pooling, and Fully Connected layers for automated feature extraction from images |

### Segmentation
| Algorithm | Description |
|-----------|-------------|
| **K-Means Clustering** | Unsupervised clustering to partition image pixels into tumour/non-tumour regions based on Euclidean distance |
| **Genetic Algorithm (GA)** | Evolutionary optimisation method using selection, crossover, and mutation operators for segmentation refinement |

---

## Results

### Classification — SVM vs CNN

| Metric | SVM | CNN |
|--------|-----|-----|
| Precision | 0.8947 | 0.921 |
| Recall | 0.309 | 0.951 |
| F1-Score | 0.459 | **0.9357** |
| Sensitivity | 0.8947 | **0.921** |
| Specificity | 0.9097 | **0.928** |
| FNR | 0.9743 | **0.047** |
| NPV | 0.918 | **0.928** |

### Accuracy by Train-Test Split

| Split | SVM | CNN |
|-------|-----|-----|
| 80:20 | 89.09% | 94.00% |
| 70:30 | 86.49% | 98.00% |
| 60:40 | 88.10% | **99.00%** |

> CNN achieved **up to 99% accuracy** and significantly outperformed SVM across all train-test splits — particularly on Recall (0.951 vs 0.309), making it far more clinically reliable for minimising missed cancer diagnoses (false negatives).

### Segmentation — K-Means vs Genetic Algorithm

| Metric | K-Means | Genetic Algorithm |
|--------|---------|-------------------|
| Rand Index (RI) ↑ | **0.86733** | 0.73913 |
| Global Consistency Error (GCE) ↓ | **0.43035** | 0.62381 |
| Variation of Information (VI) ↓ | **5.9552** | 8.9931 |

> K-Means achieved lower segmentation error, though Genetic Algorithm offers better convergence properties for future optimisation.

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.7 |
| TensorFlow | 2.1.0 |
| Keras | 2.3.1 |
| Scikit-learn | — |
| Jupyter Notebook | Anaconda |
| Spyder | Anaconda |

---

## Key Features

- **Full automated pipeline** — preprocessing, classification, segmentation, and evaluation in one workflow
- **Comparative study** — SVM vs CNN for classification; K-Means vs Genetic Algorithm for segmentation
- **Multiple train-test splits** — 80:20, 70:30, 60:40 evaluated for robustness
- **Clinical-grade metrics** — Accuracy, Precision, Recall, F1, Sensitivity, Specificity, FPR, FNR, NPV, FDR
- **PCA dimensionality reduction** — reduced to 18 most informative dimensions before model training
- **IEEE peer-reviewed** — findings published at IEEE conference and indexed on IEEE Xplore

---

## Repository Structure

```
breast-cancer-ml-detection/
│
├── notebooks/
│   ├── 01_preprocessing_pca.ipynb
│   ├── 02_svm_classification.ipynb
│   ├── 03_cnn_classification.ipynb
│   ├── 04_kmeans_segmentation.ipynb
│   └── 05_genetic_algorithm_segmentation.ipynb
│
├── data/
│   └── (histopathological image dataset)
│
├── results/
│   ├── confusion_matrices/
│   ├── accuracy_graphs/
│   └── segmented_images/
│
├── models/
│   ├── cnn_model.h5
│   └── svm_model.pkl
│
├── requirements.txt
└── README.md
```

---

## IEEE Publication

> **Automated Detection and Classification of Breast Cancer Tumour Cells using Machine Learning and Deep Learning on Histopathological Images**
>
> Published at IEEE Conference | Indexed on IEEE Xplore
> [https://ieeexplore.ieee.org/document/9417996](https://ieeexplore.ieee.org/document/9417996)

---

## Future Work

- Combine Genetic Algorithm with **chaos theory** (chaotic mapping) to improve segmentation convergence and avoid local optima
- Explore transfer learning architectures (VGG16, ResNet, InceptionV3) for improved classification
- Extend to multi-class classification across breast cancer subtypes
- Deploy as a clinical decision-support web application

---

## Authors

- **Vanshika Garg** — Manipal University Jaipur (B.Tech Computer Science, 2020–21)
- Vanshika Jain — Manipal University Jaipur

*Project Guide: Dr Anju Yadav | HOD: Mr Pankaj Vyas*

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/vanshikagarg26/breast-cancer-ml-detection.git
cd breast-cancer-ml-detection

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook
```

---

## Requirements

```
tensorflow==2.1.0
keras==2.3.1
scikit-learn
numpy
pandas
matplotlib
opencv-python
jupyter
```
