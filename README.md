# MNTD: Malicious Network Traffic Detection Framework

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.4%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This repository contains the implementation of "MNTD: An Efficient Malicious Network Traffic Detection Framework based on Optimized Deep Learning" as described in the original paper.

## Features

- Conv1D + BiLSTM(ELU) + MultiHeadAttention model (TensorFlow/Keras).
- AWDV (PSO-like) hyperparameter optimization.
- Contrastive learning with triplet sampling (pos_cos>=0.5, neg_cos<=0.3).
- Preprocessing pipeline for CSV flow data (min-max scaling, one-hot).
- Train/eval scripts and config file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MNTD-Malicious-Network-Traffic-Detection.git
cd MNTD-Malicious-Network-Traffic-Detection


# Install dependencies:
pip install -r requirements.txt

# Usage
Training the Model
1 Preprocess your dataset (see notebooks in data/ directory)
2 Run training:

python training/train.py --config configs/default_config.yaml

# Hyperparameter Optimization
To optimize hyperparameters using AWDV:

python training/train.py --optimize --config configs/default_config.yaml

# Evaluation
To evaluate a trained model:

python training/train.py --eval --model_path results/saved_models/best_model.h5

# Contributing
Contributions are welcome! Please open an issue or submit a pull request.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
