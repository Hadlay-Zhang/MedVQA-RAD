# MedVQA Framework on VQA-RAD Dataset

![GitHub repo size](https://img.shields.io/github/repo-size/Hadlay-Zhang/MedVQA-RAD)  ![GitHub stars](https://img.shields.io/github/stars/Hadlay-Zhang/MedVQA-RAD?color=yellow)  ![GitHub forks](https://img.shields.io/github/forks/Hadlay-Zhang/MedVQA-RAD?color=green&label=Fork) 

This repository is built entirely based on [MICCAI19-MedVQA](https://github.com/aioz-ai/MICCAI19-MedVQA). Gratitude to the original authors for their contributions. It is recommended to visit the original repository first.

💡UPDATES:
- [x] Replace CNNs and LSTMs with Transformers and BERTs.
- [ ] Support advanced attention network (e.x., Co-Attention Network)
- [ ] Support implementation on more datasets

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

### Pretrained Models

BERT-based pretrained models are utilized to encode questions. For example, you can use BERT and BioBERT from [HuggingFace](https://huggingface.co). Download the pretrained model (e.x., [BERT](https://huggingface.co/google-bert/bert-base-uncased/resolve/main), or [BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1/tree/main)), and then modify `args.text_path` to point to the model path.

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

The following tables showcase the performance of different models on the VQA-RAD dataset (All models utilize BAN as fusion method). The models were trained for 20 epochs or 40 epochs on a single A100-PCIE-40GB GPU (average of 10 iterations). The random seed settings can be found in `run.sh`.

1. 20 epochs
| Model (Image+Text) |     Closed      |      Open       |       All       |
| :----------------: | :-------------: | :-------------: | :-------------: |
|   ViTL16+BioBERT   | 0.7259 ± 0.0183 | 0.3138 ± 0.0242 | 0.5614 ± 0.0109 |
|  SwinTV2B+BioBERT  | 0.7238 ± 0.0203 | 0.3220 ± 0.0203 | 0.5633 ± 0.0153 |
|  ConvNeXt+BioBERT  | **0.7330 ± 0.0135** | **0.3553 ± 0.0260** | **0.5821 ± 0.0122** |
|    ConvNeXt+BERT   | 0.6822 ± 0.0136 | 0.3374 ± 0.0142 | 0.5445 ± 0.0081 |

2. 40 epochs
| Model (Image+Text) |     Closed      |      Open       |       All       |
| :----------------: | :-------------: | :-------------: | :-------------: |
|   ViTL16+BioBERT   | 0.7368 ± 0.0162 | 0.3577 ± 0.0196 | 0.5854 ± 0.0088 |
|  SwinTV2B+BioBERT  | 0.7476 ± 0.0269 | 0.3382 ± 0.0137 | 0.5841 ± 0.0183 |
|  ConvNeXt+BioBERT  | 0.7373 ± 0.0262 | **0.3927 ± 0.0189** | **0.5997 ± 0.0161** |