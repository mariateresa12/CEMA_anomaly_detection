# CEMA Time-Series Anomaly Detection

This repository contains code for anomaly detection in CEMA time-series data using both supervised Machine Learning and Deep Learning approaches. The objective is to detect abnormal patterns in temporal signals by combining physical-layer information and network-related features extracted from Zigbee traffic captures.

The repository includes two main approaches:

- A supervised **Random Forest** model.
- An unsupervised **LSTM Autoencoder** model.

## Repository Structure

```text
.
├── random_forest/
├── lstm_autoencoder/
├── zigbee_dataset_preprocess.py
└── README.md
```

### `random_forest/`

This folder contains the code related to the supervised Machine Learning approach based on a **Random Forest** classifier.

The model is designed for binary classification, distinguishing between normal and anomalous Zigbee traffic samples using preprocessed features extracted from the dataset.

### `lstm_autoencoder/`

This folder contains the code related to the Deep Learning approach based on an **LSTM Autoencoder**.

The model is designed to learn temporal patterns from normal CEMA time-series data and detect anomalies based on reconstruction errors.

### `zigbee_dataset_preprocess.py`

This file contains the preprocessing code used to transform the original Zigbee traffic captures from **PCAP** format into **JSON** format.

The preprocessing step prepares the raw packet capture files so that they can be used by the Random Forest and LSTM Autoencoder models.

## Dataset

The experiments in this repository are based on the **Zigbee dataset** published by Lourme and Hauspie, available at: https://doi.org/10.57745/NDW74U.

The dataset contains Zigbee traffic captures that can be used for security analysis and anomaly detection experiments. In this repository, the data has been preprocessed into JSON format using the `zigbee_dataset_preprocess.py` script.

## Data Preprocessing

The original dataset is provided as PCAP captures. The preprocessing pipeline implemented in `zigbee_dataset_preprocess.py` converts these captures into structured JSON files.
