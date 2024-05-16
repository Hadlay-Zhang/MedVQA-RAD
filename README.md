# MedVQA Framework on VQA-RAD Dataset

This repository is built entirely based on [MICCAI19-MedVQA](https://github.com/aioz-ai/MICCAI19-MedVQA). Gratitude to the original authors for their contributions. It is recommended to visit the original repository first.

## Introduction

The primary goal of this repository is to provide a basic framework for running Medical Visual Question Answering tasks on the VQA-RAD dataset. Unlike traditional methods that use CNNs as Image Encoders and RNNs as Text Encoders, this framework aims to utilize advanced Transformer models as replacements.

## Features

- **Transformer-based Image Encoder**: Replace traditional CNNs with Transformer models for encoding images (e.g., ViT, Swin Transformer).
- **Transformer-based Text Encoder**: Use Transformer models instead of RNNs for encoding text (e.g., BERT, BioBERT).
- **Adapted VQA-RAD Data Loading**: Modified VQA-RAD data loading methods to better suit the Transformer-based models.
- **Optimized Code Execution**: Improved execution with added run scripts and compatibility with newer versions of PyTorch.

## Getting Started

### Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/Hadlay-Zhang/MedVQA-RAD.git
    cd MedVQA-RAD
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Dataset Preparation

1. Download the VQA-RAD dataset from [here](https://vision.aioz.io/f/777a3737ee904924bf0d/?dl=1).
2. Extract the dataset into `data_RAD/`.

### Usage

To train the model, modify the commands in `run.sh` and then run:

```shell
./run.sh train
```

To evaluate the model, modify the commands in `run.sh` and then run:

```shell
./run.sh test
```

## Example Results

The following table showcases the performance of different models on the VQA-RAD dataset (All models utilize BAN as fusion method). The models were trained for 20 epochs on a single A100-PCIE-40GB GPU. The random seed settings can be found in `run.sh`.

| Model (Image+Text) |     Closed      |      Open       |       All       |
| :----------------: | :-------------: | :-------------: | :-------------: |
|   ViTL16+BioBERT   | 0.7362 ± 0.0105 | 0.3301 ± 0.0083 | 0.5740 ± 0.0093 |
|  SwinTV2B+BioBERT  | 0.7427 ± 0.0257 | 0.3138 ± 0.0110 | 0.5714 ± 0.0169 |
|  ConvNeXt+BioBERT  | 0.7416 ± 0.0134 | 0.3707 ± 0.0233 | 0.5935 ± 0.0122 |
