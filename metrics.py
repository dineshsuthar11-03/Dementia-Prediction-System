"""
metrics.py
----------
Evaluates the trained DenseNet-121 dementia classification model on the test set
and prints Accuracy, Precision, Recall, and F1-Score (per-class and weighted average).
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Configuration ────────────────────────────────────────────────────────────
# Switch to "weights/best_finetune.pth" after running dementia_finetune.py
CHECKPOINT_PATH = "weights/best_finetune.pth" if __import__("os").path.exists("weights/best_finetune.pth") \
                  else "weights/model_checkpoint_22.pth"
TEST_DIR        = "input/test"
BATCH_SIZE      = 16
NUM_CLASSES     = 3
CLASS_NAMES     = ["Moderate Dementia", "Non Demented", "Very mild Dementia"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── Transforms (same as training-time eval — no augmentation) ────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ── Dataset & DataLoader ─────────────────────────────────────────────────────
test_dataset = ImageFolder(root=TEST_DIR, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test samples  : {len(test_dataset)}")
print(f"Classes found : {test_dataset.classes}\n")

# ── Model ────────────────────────────────────────────────────────────────────
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT_PATH}\n")

# ── Inference ────────────────────────────────────────────────────────────────
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ── Metrics ──────────────────────────────────────────────────────────────────
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

# ── Pretty Print ─────────────────────────────────────────────────────────────
separator = "=" * 55

print(separator)
print("         MODEL EVALUATION METRICS (Test Set)")
print(separator)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%  (weighted avg)")
print(f"  Recall    : {recall    * 100:.2f}%  (weighted avg)")
print(f"  F1-Score  : {f1        * 100:.2f}%  (weighted avg)")
print(separator)

print("\n── Per-Class Report ──────────────────────────────────")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
print("── Confusion Matrix ──────────────────────────────────")
header = f"{'':25s}" + "".join(f"{c:>22s}" for c in CLASS_NAMES)
print(header)
for i, row in enumerate(cm):
    row_str = f"{CLASS_NAMES[i]:25s}" + "".join(f"{v:>22d}" for v in row)
    print(row_str)
print()
