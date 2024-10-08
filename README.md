# Seizure Prediction using EEG Data

This project aims to predict seizures in dogs using Electroencephalogram (EEG) data. The approach involves extracting handcrafted features from the EEG signals and training a hybrid LSTM model to classify the segments as either interictal (non-seizure) or preictal (pre-seizure).

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Requirements](#requirements)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project consists of the following steps:

1. **Data Loading**: Load preictal and interictal EEG data from `.mat` files.
2. **Feature Extraction**: Generate handcrafted features from the loaded EEG data using a CNN model for spectral analysis.
3. **Model Training**: Train an LSTM model using the extracted features to classify the segments.
4. **Evaluation**: Evaluate the model performance on test data and compute metrics like accuracy, precision, recall, and F1 score.

## Data Sources

The project uses EEG data from dogs, which can be found at the following locations:

- Preictal segments: `/kaggle/input/seizure-prediction/Dog_2/Dog_2/`
- Interictal segments: `/kaggle/input/pakistan-institute-neuroscience-neurology-eeg-data/NDC/`

## Requirements

Ensure you have the following packages installed:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `tensorflow` (for the CNN and LSTM models)
- `sklearn` (for metrics)

You can install the required packages using pip:

```bash
pip install numpy pandas scipy matplotlib seaborn tensorflow scikit-learn
```

## Model Training

The LSTM model is defined in the `create_lstm_model` function. The model consists of several LSTM layers followed by dense layers, using dropout and batch normalization for regularization. The training history is plotted to visualize the model performance over epochs.

## Evaluation

The model is evaluated on test data using various metrics, including accuracy, precision, recall, and F1 score. A confusion matrix is also generated to visualize the model's predictions.

## Results

Results will be printed to the console after the evaluation, including:

- Accuracy
- Recall
- Precision
- Confusion matrix values (TP, TN, FP, FN)
