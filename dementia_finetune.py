"""
dementia_finetune.py
--------------------
Fine-tuned DenseNet-121 training with all improvements applied:
  1. Unfreeze denseblock3 + denseblock4 (more MRI-specific capacity)
  2. WeightedRandomSampler  (balanced batches — fixes Very Mild / Non-Dem confusion)
  3. Weighted CrossEntropyLoss + label_smoothing=0.1
  4. MRI-specific augmentation (affine, blur, contrast jitter)
  5. LR warmup (5 ep) then CosineAnnealingLR — LR = 5e-6
  6. 50 epochs
  7. Saves best checkpoint by validation F1-score
  8. Prints Precision / Recall / F1 each epoch
"""

from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import trange
import os

# ── Device ───────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Transforms ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    # Geometry augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.Resize((224, 224)),
    # Intensity augmentation (MRI contrast variability)
    transforms.ColorJitter(brightness=0.2, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Dataset helper ───────────────────────────────────────────────────────────
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path

# ── Datasets ─────────────────────────────────────────────────────────────────
train_dataset = ImageFolderWithPaths(root="input/train", transform=train_transform)
test_dataset  = ImageFolderWithPaths(root="input/test",  transform=val_transform)

train_labels = [label for _, label, _ in train_dataset]
class_counts = Counter(train_labels)
print(f"Train distribution: {dict(sorted(class_counts.items()))}")
print(f"Classes: {train_dataset.classes}")

# ── Balanced sampler ──────────────────────────────────────────────────────────
# Each class is sampled proportionally to 1/count so every class gets equal
# representation within each batch, tackling Very Mild / Non-Dem imbalance.
num_samples   = len(train_dataset)
class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, label, _ in train_dataset]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=num_samples,
    replacement=True,
)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False,   num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze denseblock3, transition3, denseblock4, norm5 (classifier neck)
for param in model.features.transition3.parameters():
    param.requires_grad = True
for param in model.features.denseblock3.parameters():
    param.requires_grad = True
for param in model.features.denseblock4.parameters():
    param.requires_grad = True
for param in model.features.norm5.parameters():
    param.requires_grad = True

# Replace classifier head
num_classes = 3
model.classifier = nn.Linear(model.classifier.in_features, num_classes).to(device)

unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
total    = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {unfrozen:,} / {total:,}  ({100*unfrozen/total:.1f}%)")

# ── Weighted loss ─────────────────────────────────────────────────────────────
# Inverse-frequency per-class weights so the loss penalises minority class
# errors more heavily; label_smoothing=0.1 prevents overconfident predictions.
counts_tensor = torch.tensor(
    [class_counts[i] for i in range(num_classes)], dtype=torch.float
)
inv_weights = 1.0 / counts_tensor
inv_weights = inv_weights / inv_weights.sum() * num_classes   # normalise to sum=3
loss_fn = nn.CrossEntropyLoss(weight=inv_weights.to(device), label_smoothing=0.1)

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-6,
    weight_decay=1e-4,
)

# ── LR schedule: linear warmup 5 ep → cosine anneal remaining 45 ep ──────────
epochs       = 50
warmup_epochs = 5

warmup_scheduler  = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs
)
cosine_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs - warmup_epochs, eta_min=1e-7
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs],
)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_f1    = 0.0
best_epoch     = -1
os.makedirs("weights", exist_ok=True)

for epoch in trange(epochs, desc="training", unit="ep"):

    # ---- TRAIN ----
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == y).sum().item()
        train_loss    += loss.item()
        train_total   += y.size(0)

    train_loss /= len(train_loader)
    train_acc   = 100 * train_correct / train_total
    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # ---- VALIDATION ----
    model.eval()
    val_loss       = 0.0
    val_correct    = 0
    val_total      = 0
    all_probs      = []
    all_labels_raw = []
    patient_probs  = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for x, y, paths in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.nn.functional.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            val_correct += (preds == y).sum().item()
            val_total   += y.size(0)
            val_loss    += loss_fn(logits, y).item()

            all_probs.append(probs.cpu())
            all_labels_raw.append(y.cpu())

            for i in range(len(paths)):
                pid = paths[i].split("_")[1]
                patient_probs[pid].append(probs[i].cpu().numpy())
                patient_labels[pid] = y[i].cpu().item()

    val_loss /= len(test_loader)
    val_acc   = 100 * val_correct / val_total

    all_probs_np  = torch.cat(all_probs).numpy()
    all_labels_np = torch.cat(all_labels_raw).numpy()
    all_preds_np  = all_probs_np.argmax(axis=1)

    # Slice-level metrics
    val_f1   = f1_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)
    val_prec = precision_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)
    val_rec  = recall_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)

    # Very Mild Dementia per-class recall (class index 2)
    per_class_rec = recall_score(all_labels_np, all_preds_np, average=None, zero_division=0)
    vmd_recall    = per_class_rec[2] if len(per_class_rec) > 2 else 0.0

    # Patient-level AUROC
    final_probs, final_labels_p = [], []
    for pid in patient_probs:
        final_probs.append(np.mean(patient_probs[pid], axis=0))
        final_labels_p.append(patient_labels[pid])
    final_probs    = np.array(final_probs)
    final_labels_p = np.array(final_labels_p)

    try:
        slice_auc   = roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr")
        patient_auc = roc_auc_score(final_labels_p, final_probs,  multi_class="ovr")
    except ValueError:
        slice_auc = patient_auc = 0.0

    # ---- LOG ----
    print(
        f"Ep {epoch+1:02d}/{epochs} | "
        f"LR {current_lr:.2e} | "
        f"TrainLoss {train_loss:.4f} | TrainAcc {train_acc:.2f}% | "
        f"ValLoss {val_loss:.4f} | ValAcc {val_acc:.2f}% | "
        f"Prec {val_prec*100:.2f}% | Rec {val_rec*100:.2f}% | F1 {val_f1*100:.2f}% | "
        f"VMD-Recall {vmd_recall*100:.2f}% | "
        f"PatAUROC {patient_auc:.4f}"
    )

    # ---- CHECKPOINT: save best by val F1 ----
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch + 1
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     epoch + 1,
            "val_f1":    val_f1,
            "val_acc":   val_acc,
        }, "weights/best_finetune.pth")
        print(f"  *** New best saved  (F1={val_f1*100:.2f}%  Ep {epoch+1}) ***")

    # Save periodic checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch":     epoch + 1,
        }, f"weights/finetune_checkpoint_{epoch+1}.pth")

# ── Final save ────────────────────────────────────────────────────────────────
torch.save({
    "model":     model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch":     epochs,
}, "weights/finetune_final.pth")

print(f"\nTraining complete.")
print(f"Best model: Epoch {best_epoch}  |  Val F1 = {best_val_f1*100:.2f}%")
print("Saved to weights/best_finetune.pth")
