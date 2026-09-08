# EP25 — Image Classification Project (End-to-End)

**CodeToAGI Deep Learning Series · Module 5 Capstone**

Build a complete, production-ready image classifier on **your own dataset** using PyTorch.

No toy datasets. No hand-holding. Real pipeline from folder of images → deployable ONNX model.

## What you will build

| Step | What happens |
|------|--------------|
| 1 | Dataset prep (ImageFolder structure + class balance) |
| 2 | DataLoader + full augmentation pipeline (RandAugment, ColorJitter, RandomErasing…) |
| 3 | Fine-tune ResNet-18 with differential learning rates + label smoothing |
| 4 | Evaluation (accuracy, per-class F1, confusion matrix) |
| 5 | Grad-CAM visualisation |
| 6 | Misclassification analysis |
| 7 | ONNX export + verification |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/CodeToAGI/deep-learning-series.git
cd deep-learning-series/ep25

# 2. Install
pip install torch torchvision scikit-learn matplotlib seaborn pillow onnx onnxruntime

# 3. Put your images here
data/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/

# 4. Run
python ep25_project.py
