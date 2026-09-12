# Deep Learning for Perception — Assignment 1
### Building, Breaking, and Fixing a Neural Network (Fashion-MNIST)

Fall 2026 | National University of Computer and Emerging Sciences

## Overview

This repository implements a 7-part study on a feedforward neural network trained on Fashion-MNIST: a from-scratch NumPy backpropagation implementation with a PyTorch gradient check, an activation function study, a loss function comparison, an optimiser comparison, deliberate overfitting, a regularisation study, and hyperparameter tuning via 5-fold cross-validation with a final held-out test evaluation.

## Repository Structure

```
.
├── DLP_ASS01_XXF_YYYY.ipynb   # Main notebook — all 7 parts, executed with visible outputs
├── DLP_Assignment01_Summary.docx  # One-page results summary
└── README.md
```

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, seaborn
- torch, torchvision
- scikit-learn

Kaggle notebooks come with all of these preinstalled. If running locally:

```bash
pip install numpy pandas matplotlib seaborn torch torchvision scikit-learn
```

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) — loaded via Kaggle's "Add Input" feature (dataset: `zalando-research/fashionmnist`), using the CSV format (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`).

## How to Reproduce

1. Open the notebook on [Kaggle](https://www.kaggle.com/) with the Fashion-MNIST dataset added as input.
2. Set the accelerator to **GPU T4 x2** (Session Options → Accelerator) before running Part 2 onward. Part 1 (NumPy-only) runs fine on CPU.
3. Run all cells top to bottom — later parts depend on models and variables created in earlier ones (e.g. Part 6 reuses Part 5's overfitted setup; Part 7 combines Part 6's best regularisation with its own CV-selected configuration).
4. A fixed random seed (`SEED = 42`) is set at the top of the notebook and reused throughout for reproducibility of splits, weight initialisation, and data shuffling.
5. The test set (`fashion-mnist_test.csv`) is loaded once at the start and is only evaluated against in **Part 7**, after all model-selection decisions are finalised — no test-set leakage during tuning.

## Results Summary

| Part | Task | Key Result |
|---|---|---|
| 1 | NumPy backprop + gradient check | Max gradient difference vs. PyTorch: ~1e-8 |
| 2 | Activation study | ReLU dead units: 9.38%; sigmoid gradients ~9x smaller than ReLU at epoch 1 |
| 3 | Loss functions | Cross-entropy 89.19% vs. MSE 88.78% test accuracy |
| 4 | Optimiser comparison | Adam best after tuning: 89.37% validation accuracy |
| 5 | Forced overfitting | 16.04 percentage point train/validation gap |
| 6 | Regularisation study | Dropout (rate 0.6) best: gap reduced to 10.68pp with no accuracy loss |
| 7 | CV tuning + final test evaluation | 88.62% test accuracy (macro F1: 88.48%) |

Full details, plots, and written analysis are in the notebook and in `DLP_Assignment01_Summary.docx`.

## Notes

- Part 7's final model did not improve on Part 2's baseline (-0.51 percentage points), reported as an honest negative result — see the notebook's Part 7 conclusion for the explanation (the CV search itself preferred no dropout; forcing dropout=0.6 from Part 6, tuned for a much more overfitting-prone setup, likely over-constrained this model).
- All plots include axis labels, titles, and legends per the assignment's requirements.
