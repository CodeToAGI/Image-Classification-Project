"""
EP25 Capstone Challenge — Image Classification on YOUR Dataset
==============================================================
Full pipeline:
  1. Dataset prep (ImageFolder)
  2. DataLoader + full augmentation
  3. Fine-tune ResNet-18 (differential LR)
  4. Evaluation + confusion matrix
  5. Grad-CAM
  6. Misclassification analysis
  7. ONNX export + verification

Usage:
  1. Put your images in:
       data/train/class_a/...
       data/train/class_b/...
       data/val/class_a/...
       data/val/class_b/...
  2. python ep25_project.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import onnx
import onnxruntime as ort

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
TRAIN_DIR     = DATA_DIR / "train"
VAL_DIR       = DATA_DIR / "val"
BATCH_SIZE    = 32
NUM_WORKERS   = 4 if os.name != "nt" else 2
EPOCHS        = 20
LR_BACKBONE   = 1e-4
LR_HEAD       = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42
OUTPUT_DIR    = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ──────────────────────────────────────────────────────────────
# 1. TRANSFORMS (full EP24 pipeline)
# ──────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────────────────────
# 2. DATASETS + DATALOADERS
# ──────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)

class_names = train_ds.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

# Class balance check
counts = defaultdict(int)
for _, label in train_ds.samples:
    counts[class_names[label]] += 1
print("\nClass counts (train):")
for c, n in counts.items():
    print(f"  {c}: {n}")

# Optional: WeightedRandomSampler for imbalance
weights = [1.0 / counts[class_names[label]] for _, label in train_ds.samples]
sampler = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# Quick visual check of one batch
def show_batch(loader, n=8):
    imgs, labels = next(iter(loader))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imgs = imgs[:n] * std + mean
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2.5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].permute(1,2,0).clamp(0,1).numpy())
        ax.set_title(class_names[labels[i]], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_preview.png", dpi=120)
    plt.close()
    print("Saved batch preview → outputs/batch_preview.png")

show_batch(train_loader)

# ──────────────────────────────────────────────────────────────
# 3. MODEL — ResNet-18 + differential LR
# ──────────────────────────────────────────────────────────────
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(DEVICE)

# Differential learning rates
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "fc" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
], weight_decay=1e-4)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ──────────────────────────────────────────────────────────────
# 4. TRAIN + EVAL
# ──────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return correct / total, all_preds, all_labels

best_acc = 0.0
history = {"train_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    train_loss = running_loss / len(train_loader)
    val_acc, _, _ = evaluate(model, val_loader)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
        print(f"  → Saved best model ({best_acc:.4f})")

# Load best
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE))
print(f"\nBest validation accuracy: {best_acc:.4f}")

# ──────────────────────────────────────────────────────────────
# 5. FULL EVALUATION
# ──────────────────────────────────────────────────────────────
val_acc, preds, labels = evaluate(model, val_loader)
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=class_names, digits=4))

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion matrix → outputs/confusion_matrix.png")

# ──────────────────────────────────────────────────────────────
# 6. GRAD-CAM (simple from-scratch version)
# ──────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        x = x.requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

# Run Grad-CAM on a few val images
gradcam = GradCAM(model, model.layer4[-1])
os.makedirs(OUTPUT_DIR / "gradcam", exist_ok=True)

def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t * std + mean).clamp(0, 1)

val_iter = iter(val_loader)
for i in range(6):
    imgs, labels = next(val_iter)
    img = imgs[0:1].to(DEVICE)
    true_label = labels[0].item()
    cam, pred_idx = gradcam(img)
    img_np = denormalize(imgs[0]).permute(1,2,0).numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"True: {class_names[true_label]}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(cam, cmap="jet", alpha=0.4)
    plt.title(f"Pred: {class_names[pred_idx]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradcam" / f"gradcam_{i}.png", dpi=120)
    plt.close()

print("Saved Grad-CAM images → outputs/gradcam/")

# ──────────────────────────────────────────────────────────────
# 7. MISCLASSIFICATION ANALYSIS
# ──────────────────────────────────────────────────────────────
misclassified = []
model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(1)
        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append({
                    "true": class_names[labels[i].item()],
                    "pred": class_names[preds[i].item()],
                    "confidence": probs[i, preds[i]].item(),
                })

misclassified.sort(key=lambda x: -x["confidence"])
print(f"\nTop high-confidence misclassifications ({len(misclassified)} total):")
for m in misclassified[:8]:
    print(f"  True={m['true']} | Pred={m['pred']} | Conf={m['confidence']:.3f}")

# ──────────────────────────────────────────────────────────────
# 8. ONNX EXPORT + VERIFY
# ──────────────────────────────────────────────────────────────
model.eval()
dummy = torch.randn(1, 3, 224, 224, device=DEVICE)
onnx_path = OUTPUT_DIR / "classifier.onnx"

torch.onnx.export(
    model, dummy, str(onnx_path),
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
)
print(f"\nExported ONNX → {onnx_path}")

# Verify
sess = ort.InferenceSession(str(onnx_path))
ort_out = sess.run(None, {"image": dummy.cpu().numpy()})[0]
with torch.no_grad():
    torch_out = model(dummy).cpu().numpy()
assert np.allclose(ort_out, torch_out, atol=1e-4), "ONNX outputs differ!"
print("ONNX verification passed ✓  (outputs match PyTorch)")

print("\n✅ EP25 pipeline complete. Check the outputs/ folder.")
