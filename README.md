# DCT-PT: Frequency-Aware Prompt Tuning for Few-Shot Human Activity Recognition

[![IEEE](https://img.shields.io/badge/IEEE-TBD-blue)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **DCT-PT**, a lightweight frequency-aware Parameter-Efficient Fine-Tuning (PEFT) framework for few-shot Human Activity Recognition (HAR) based on the self-supervised pre-trained Time Series Foundation Model (TSFM) **MOMENT**.

> **DCT-PT: Frequency-Aware Prompt Tuning for Few-Shot Human Activity Recognition**  
> Xiaohui Ye, Lei Zhang, Guangjie Chen, Chaoda Song, Shuoyuan Wang, Hao Wu, Aiguo Song  
> *School of Electrical and Automation Engineering, Nanjing Normal University, China*  
> Paper: [arXiv / IEEE Link (TBD)]

---

## Overview

![DCT-PT Framework](docs/model.pdf)

Adapting large-scale pre-trained TSFMs to downstream HAR tasks faces two fundamental challenges: (1) **data scarcity** — labeled sensor data is expensive to annotate, and (2) **representation heterogeneity** — sensor signals exhibit high variability across devices, body positions, and subjects due to distribution shifts.

**DCT-PT** addresses these challenges through a three-component lightweight PEFT framework:

| Module | Function |
|--------|----------|
| **Adaptive DCT-based Sampling (ADS)** | Generates frequency-aware prompts at the input stage using DCT-based adaptive frequency selection, compensating for spectral information lost during tokenization |
| **Multi-scale Frequency-aware residual Module (MFM)** | Decouples low- and high-frequency components via learnable branches with cross-frequency gating, enabling interaction between global structures and local details |
| **Channel-wise Shift Module (CWS)** | Aligns feature distributions between pre-training and downstream domains via learnable channel-wise shifts with ℓ₂ normalization |

### Key Features

- 🚀 **State-of-the-art performance** on HHAR, MotionSense, and PAMAP2 under both few-shot and full-shot settings
- 💡 **Extreme parameter efficiency** — only **0.042M** trainable parameters (< **1%** of backbone), ~162 MB GPU memory
- 🎯 **Stable optimization** — eliminates training oscillation common in soft prompting approaches
- 📱 **Edge-deployable** — real-time inference on Raspberry Pi 5 (612 ms) and cloud-edge collaborative deployment (45 ms)

---

## Results

### Few-shot HAR Performance (Accuracy %)

| Method | Params (M) | HHAR 1-shot | HHAR 5-shot | MotionSense 1-shot | MotionSense 5-shot | PAMAP2 1-shot | PAMAP2 5-shot |
|--------|:----------:|:-----------:|:-----------:|:------------------:|:------------------:|:-------------:|:-------------:|
| DCNN | 35.374 | 27.5 | 34.7 | 35.5 | 39.5 | 32.0 | 45.3 |
| TCN | — | 32.4 | 43.2 | 38.7 | 47.2 | 35.8 | 52.0 |
| ConformerHAR | — | 36.4 | 49.9 | 45.5 | 53.3 | 36.3 | 61.2 |
| TS2ACT | — | 71.1 | 75.2 | 74.7 | 77.3 | 69.2 | 78.0 |
| Vi2ACT | — | **75.5** | **78.5** | 76.1 | 82.2 | 73.3 | 86.0 |
| **DCT-PT (Ours)** | **0.042** | 74.9 | 78.0 | **85.7** | **92.8** | **74.4** | **91.0** |

*DCT-PT achieves up to **+9.6%** over the state-of-the-art (Vi2ACT) on MotionSense 1-shot, with only **0.042M** trainable parameters.*

### PEFT Method Comparison (Accuracy %)

| Method | Params (M) | HHAR Full | MotionSense Full | PAMAP2 Full |
|--------|:----------:|:---------:|:----------------:|:-----------:|
| FT-Full | 35.374 | 91.6 | 94.3 | 94.2 |
| FT-Head | 0.037 | 93.5 | 98.0 | 97.4 |
| Adapter | 0.238 | 93.0 | 98.1 | 98.3 |
| LoRA | 0.266 | 93.1 | 98.1 | 97.5 |
| **DCT-PT (Ours)** | **0.042** | **95.3** | **98.4** | **98.9** |

### Visualization

| Analysis | Description |
|----------|-------------|
| **Loss Landscape** | DCT-PT exhibits a smooth, convex bowl-shaped loss landscape, indicating robust optimization and reduced parameter sensitivity |
| **t-SNE** | Encoder features show well-separated class-wise manifolds with clear geodesic separation compared to LoRA baselines |
| **Frequency Analysis** | DCT-based sampling acts as a learnable filter isolating frequency components physically characteristic of each motion type |

---

## Method

### Architecture

![DCT-PT Architecture](docs/model.pdf)

The overall framework of DCT-PT:

**(a) Adaptive DCT-based Sampling (ADS)** transforms raw sensor signals into the frequency domain via 1D DCT, applies learnable frequency selection via LoRA-based gating, and generates frequency-aware prompts that compensate for spectral information loss during tokenization.

**(b) Multi-scale Frequency-aware residual Module (MFM)** uses depthwise separable convolutions with different kernel sizes ($5 \times 5$ for low-frequency, $3 \times 3$ for high-frequency) to decompose features, with cross-frequency gating for bidirectional information modulation.

**(c) Channel-wise Shift Module (CWS)** applies learnable per-channel shifts with ℓ₂ normalization to align feature distributions, mitigating domain drift between pre-training and downstream tasks.

### Core Formulation

**DCT Basis:**
$$C_{k,n} = \begin{cases} \sqrt{\frac{1}{T}}, & k = 0 \\ \sqrt{\frac{2}{T}} \cos\left(\frac{\pi(2n+1)k}{2T}\right), & k > 0 \end{cases}$$

**Frequency-aware Prompt:**
$$\mathbf{P} = \sum_{k=1}^{K} \mathbf{S}_k \odot \mathbf{F}_k$$

where $\mathbf{S}$ are frequency importance weights from Softmax-gated LoRA.

**Cross-frequency Interaction:**
$$\hat{F}_L = F_L \odot \sigma(\mathcal{G}_{H \rightarrow L}(F_H)), \quad \hat{F}_H = F_H \odot \sigma(\mathcal{G}_{L \rightarrow H}(F_L))$$

**Channel-wise Shift:**
$$Z'_{c} = \frac{Z_{c} + s_{c}}{\|Z_{c} + s_{c}\|_2}$$

---

## Installation

```bash
# Clone the repository
git clone https://github.com/JohnsonJeune/DCT-PT.git
cd DCT-PT

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ (tested on 2.6.0)
- CUDA 11.8+ (for GPU training)
- NumPy, SciPy, scikit-learn
- MOMENT (pre-trained TSFM backbone)

---

## Datasets

| Dataset | Subjects | Activities | Sensors | Samples |
|---------|:--------:|:----------:|:-------:|:-------:|
| HHAR | 9 | 6 | A, G | 24,000 |
| MotionSense | 24 | 6 | A, G | 10,765 |
| PAMAP2 | 9 | 12 | A, G, M | 7,550 |

*A = Accelerometer, G = Gyroscope, M = Magnetometer*

Data preprocessing:
- Sliding window: $T = 500$ samples with 50% overlap
- Train/test split: 80%/20% (random seed 16)
- Few-shot sampling: 1, 5, 10, 20 shots per class

---

## Usage

### Training

```bash
# Few-shot training (1-shot on MotionSense)
python train.py --dataset MotionSense --shots 1 --backbone MOMENT

# Full-shot training
python train.py --dataset HHAR --shots full --backbone MOMENT
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --dataset PAMAP2 --shots 5 --checkpoint ./checkpoints/best_model.pth
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank $r$ | 8 |
| DCT frequency bands | 25 |
| Sequence length $T$ | 500 |
| Epochs | 200 |
| Optimizer | Adam (weight decay $5 \times 10^{-4}$) |
| Learning rate | $\{10^{-2}, 10^{-3}\}$ |
| Temperature $\tau$ | 0.07 |
| Batch size | 8 |

---

## Project Structure

```
DCT-PT/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── config.py              # Configuration
├── models/
│   ├── ads.py             # Adaptive DCT-based Sampling
│   ├── mfm.py             # Multi-scale Frequency-aware Module
│   └── cws.py             # Channel-wise Shift Module
├── datasets/
│   ├── hhar.py            # HHAR dataset loader
│   ├── motionsense.py     # MotionSense dataset loader
│   └── pamap2.py          # PAMAP2 dataset loader
├── utils/
│   └── metrics.py         # Evaluation metrics
├── docs/
│   ├── model.pdf          # Architecture diagram
│   └── figures/           # Result figures
└── checkpoints/           # Saved model weights
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{ye2025dctpt,
  title={DCT-PT: Frequency-Aware Prompt Tuning for Few-Shot Human Activity Recognition},
  author={Ye, Xiaohui and Zhang, Lei and Chen, Guangjie and Song, Chaoda and Wang, Shuoyuan and Wu, Hao and Song, Aiguo},
  journal={IEEE Transactions on ...},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant 62373194, in part by the Cultivating Plan Program for the Leader in Science and Technology of Yunnan Province, China under Grant 202005AC160005, and in part by the Ten Thousand Talent Plans for Young of Yunnan Province, China under Grant YNWR-QNBJ-2019-188.

We thank the authors of [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) for providing the open-source pre-trained TSFM backbone.

---

> **Note:** Code and pre-trained models will be released upon paper acceptance.
