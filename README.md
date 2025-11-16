# 🎤 Speech Emotion Recognition (RAVDESS)  
**74% Validation Accuracy with Hand-Crafted Features + MLP**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![RAVDESS](https://img.shields.io/badge/Dataset-RAVDESS-green)](https://zenodo.org/record/1188976)

---

## 📊 Results

| Model | Val Accuracy | Train Accuracy | Classes |
|------|--------------|----------------|--------|
| **MLP + 147 Features** | **74.2%** | 80.1% | 8 |

> **Beats typical MLP baselines (60–70%)** — **no CNN, no pre-trained models!**

![Training Curves](results/training_curves.png)

---

## 🎯 Features Used (147-dim)
- MFCC (40) + Δ + ΔΔ
- Chroma, Spectral Contrast, Tonnetz
- Zero-Crossing Rate, RMS Energy
- Mean-pooled over 3-second clips

---
dataset link : https://zenodo.org/records/1188976
