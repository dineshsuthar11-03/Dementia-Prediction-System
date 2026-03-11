# Dementia Prediction System

A deep learning–based MRI image classification system that automatically predicts the stage of dementia from brain MRI scans. The model is built on **DenseNet-121**, fine-tuned on the OASIS-1 dataset, and classifies patients into three categories: *Moderate Dementia*, *Non Demented*, and *Very Mild Dementia*.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Strategy](#training-strategy)
- [Fine-Tuning Improvements](#fine-tuning-improvements)
- [Evaluation Metrics](#evaluation-metrics)
- [How Prediction Works](#how-prediction-works)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Dependencies](#dependencies)

---

## Overview

Dementia is a progressive neurological condition that leads to cognitive decline. Early and accurate detection through MRI scans is critical for timely intervention. This system automates the classification of dementia severity directly from raw MRI slice images using transfer learning with DenseNet-121.

**Problem Type**: Multi-class image classification (3 classes)  
**Input**: Grayscale MRI brain scan images (JPEG)  
**Output**: Predicted dementia class with confidence score  

---

## Model Architecture

### Base Model — DenseNet-121

**DenseNet-121 (Densely Connected Convolutional Network)** is a state-of-the-art CNN architecture where each layer receives feature maps from *all* preceding layers. This dense connectivity pattern provides:

- **Feature reuse** — earlier features are reused throughout the network, leading to more compact and expressive representations.
- **Gradient flow** — dense skip connections mitigate the vanishing gradient problem, enabling effective training of deep networks.
- **Parameter efficiency** — achieves strong performance with fewer parameters compared to models like VGG or ResNet.

DenseNet-121 consists of 4 dense blocks separated by transition layers, with a final global average pooling and a fully connected classifier.

### Modifications for this Project

| Component | Original | Modified |
|-----------|----------|---------|
| Pre-training | ImageNet (1000 classes) | Loaded as initial weights |
| Classifier head | `Linear(1024, 1000)` | `Linear(1024, 3)` |
| Fine-tuned layers | All frozen | `denseblock4` + classifier unfrozen |
| Output activation | Softmax (1000) | Softmax (3 classes) |

- **Input Size**: 224×224 RGB images (grayscale MRI scans converted to 3-channel)
- **Output**: 3 class probabilities via softmax

### Classification Classes

| Label | Class Name | Description |
|-------|------------|-------------|
| 0 | Moderate Dementia | Patients with moderate cognitive impairment |
| 1 | Non Demented | Healthy control subjects with no cognitive decline |
| 2 | Very Mild Dementia | Patients with early-stage or mild cognitive impairment |

---

## Dataset

The project uses MRI brain scan images from the **OASIS-1 (Open Access Series of Imaging Studies)** dataset. Each patient session contains multiple MRI slices (typically 60–70 slices per patient), providing rich spatial coverage of the brain.

### Training Set — 36 patients

| Class | Patient Count | Patient IDs |
|-------|---------------|-------------|
| Moderate Dementia | 14 | 0028, 0031, 0052, 0053, 0056, 0067, 0073, 0122, 0134, 0137, 0184, 0185, 0308, 0351 |
| Non Demented | 11 | 0004, 0005, 0006, 0007, 0009, 0010, 0011, 0012, 0013, 0014, 0017 |
| Very Mild Dementia | 11 | 0021, 0022, 0023, 0039, 0041, 0042, 0046, 0060, 0066, 0082, 0084 |

### Test Set — 8 patients

| Class | Patient Count | Patient IDs |
|-------|---------------|-------------|
| Moderate Dementia | 3 | 0028, 0031, 0035 |
| Non Demented | 2 | 0001, 0002 |
| Very Mild Dementia | 3 | 0003, 0015, 0016 |

**Total**: 44 unique patients (36 training + 8 test)

---

## Training Strategy

### Data Preprocessing & Augmentation

All images undergo the following pipeline before being fed into the model:

| Step | Training | Evaluation/Test |
|------|----------|-----------------|
| Grayscale → 3-channel | ✔ | ✔ |
| Random horizontal flip | ✔ | ✘ |
| Random rotation (±10°) | ✔ | ✘ |
| Resize to 224×224 | ✔ | ✔ |
| Normalize (ImageNet stats) | ✔ | ✔ |

**ImageNet normalization**: mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]`

### Transfer Learning & Fine-Tuning

1. Load DenseNet-121 with ImageNet pre-trained weights.
2. Freeze all layers except the last dense block (`denseblock4`) and the classifier head.
3. Replace the original 1000-class classifier with a 3-class `Linear` layer.
4. Train only the unfrozen layers to preserve low-level feature representations while adapting high-level features to the MRI domain.

### Optimizer & Scheduler

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Batch size | 16 |
| Loss function | Cross-Entropy Loss |

**CosineAnnealingLR** gradually reduces the learning rate following a cosine curve, helping the model converge to a better minimum while avoiding oscillation at lower loss values.

### Validation Strategy

During training, **patient-level AUROC** is computed alongside slice-level accuracy. This is important because multiple slices from the same patient should collectively vote on the final prediction:

1. Softmax probabilities are computed for each slice.
2. Probabilities are averaged across all slices of a patient (ensemble averaging).
3. The averaged distribution is used to compute patient-level predictions and AUROC.

---

## Fine-Tuning Improvements

The baseline model (`dementia.py`) achieved **~59.8% accuracy** with poor Very Mild Dementia recall (0.42). The improved training script `dementia_finetune.py` applies the following targeted fixes:

### 1. Balanced Sampling (`WeightedRandomSampler`)
The root cause of low Very Mild Dementia recall was that the model trained on unbalanced batches. Each sample is now assigned a weight inversely proportional to its class frequency. Every batch drawn by the sampler is class-balanced, forcing the model to learn equally well across all three stages.

```python
class_weights  = {cls: 1.0 / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for _, label, _ in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
```

### 2. Weighted Cross-Entropy Loss + Label Smoothing
Two complementary mechanisms applied to the loss function:
- **Inverse-frequency class weights** increase the penalty for misclassifying rare classes, reinforcing the sampler's effect.
- **Label smoothing (0.1)** prevents the model from becoming overconfident on any single class, improving generalisation.

```python
loss_fn = nn.CrossEntropyLoss(weight=inv_weights.to(device), label_smoothing=0.1)
```

### 3. Deeper Unfreezing — denseblock3 + transition3
The baseline only unfroze `denseblock4`. The fine-tuned model additionally unfreezes `denseblock3` and the `transition3` layer, giving the model more capacity to re-learn MRI-specific intermediate features (tissue textures, boundary sharpness) that differ significantly from ImageNet patterns.

| Layer group | Baseline | Fine-tuned |
|-------------|----------|------------|
| denseblock1-2 | Frozen | Frozen |
| denseblock3 + transition3 | Frozen | **Unfrozen** |
| denseblock4 + norm5 | Unfrozen | Unfrozen |
| Classifier | Unfrozen | Unfrozen |

### 4. MRI-Specific Augmentation
Additional augmentations that simulate real MRI variability:

| Augmentation | Purpose |
|---|---|
| `RandomAffine` (translate ±5%, scale 95–105%) | Simulates patient head positioning variation |
| `RandomVerticalFlip` (p=0.1) | Axial slice orientation variation |
| `ColorJitter` (brightness 0.2, contrast 0.3) | MRI scanner gain/contrast variability |
| `GaussianBlur` (σ 0.1–1.5) | Simulates varying MRI slice thickness/noise |

### 5. Learning Rate Warmup + Cosine Annealing
The lower base LR (`5e-6`) avoids disrupting the pre-trained weights of the newly unfrozen `denseblock3`. A 5-epoch linear warmup ramps the LR from `1e-6` to `5e-6` before handing off to cosine annealing.

```
Epoch 1-5:  LR ramps  1e-6 → 5e-6   (warmup)
Epoch 6-50: LR cosine 5e-6 → 1e-7   (fine convergence)
```

### 6. Best-Model Checkpointing by F1
Instead of saving every-epoch checkpoints, the fine-tune run saves `weights/best_finetune.pth` only when validation **weighted F1** improves. This ensures the deployed model is the one that best balanced precision and recall, not the most-trained.

### 7. Per-epoch Diagnostic Logging
Each epoch prints Very Mild Dementia recall explicitly so degradation is caught early:
```
Ep 12/50 | LR 4.23e-06 | ... | F1 71.40% | VMD-Recall 68.30% | PatAUROC 0.9211
```

---

## Evaluation Metrics

Run the dedicated metrics script to compute all performance metrics on the test set:

```bash
python metrics.py
```

The script reports:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly classified MRI slices |
| **Precision** | Of all slices predicted as class X, how many actually belong to class X |
| **Recall** | Of all slices truly belonging to class X, how many were correctly identified |
| **F1-Score** | Harmonic mean of precision and recall — balances both metrics |

All metrics are reported both **per-class** and as a **weighted average** (weighted by class support). A confusion matrix is also displayed.

### Why These Metrics Matter

- **Accuracy** alone can be misleading on imbalanced datasets. In medical diagnosis, it is critical to also evaluate precision and recall.
- **Recall (Sensitivity)** is especially important in clinical settings — a missed dementia diagnosis (false negative) is more harmful than a false alarm.
- **F1-Score** provides a single balanced score useful when class distributions are unequal.

---

## How Prediction Works

### Slice-Level (`metrics.py` and training loop)

Each MRI slice image is individually classified. Predictions across the whole test set are aggregated to compute overall metrics.

### Patient-Level (`demeval.py`)

For a single patient, prediction is performed by **slice averaging**:

1. Load all MRI slices for the patient (e.g., 61 slices for OAS1_0031).
2. Preprocess each slice (grayscale → 3-channel, resize, normalize).
3. Forward pass through DenseNet-121 → `[num_slices, 3]` logits.
4. Apply softmax → per-slice class probabilities.
5. Average probabilities across all slices.
6. Final prediction = class with highest average probability.

```
Input:  61 MRI slices  →  torch.Size([61, 3, 224, 224])
Output: per-slice prob  →  torch.Size([61, 3])
Final:  avg. vote       →  "Moderate Dementia"
```

---

## Project Structure

```
Dementia-Prediction-System/
│
├── dementia.py            # Baseline training script (DenseNet-121, denseblock4 only)
├── dementia_finetune.py   # Fine-tuned training script (all improvements applied)
├── demeval.py             # Patient-level single-patient inference script
├── metrics.py             # Evaluation script: accuracy, precision, recall, F1-score
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── input/
│   ├── train/
│   │   ├── Moderate Dementia/
│   │   ├── Non Demented/
│   │   └── Very mild Dementia/
│   └── test/
│       ├── Moderate Dementia/
│       ├── Non Demented/
│       └── Very mild Dementia/
│
└── weights/
    ├── model_checkpoint_17.pth
    ├── model_checkpoint_21.pth
    ├── model_checkpoint_22.pth      ← baseline checkpoint
    ├── best_finetune.pth            ← best fine-tuned checkpoint (by val F1)
    └── finetune_checkpoint_<N>.pth  ← periodic checkpoints every 10 epochs
```

---

## Installation & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the baseline model

```bash
python dementia.py
```

Checkpoints saved as `model_checkpoint_<epoch>.pth` after each epoch.

### 3. Run the fine-tuned training (recommended)

```bash
python dementia_finetune.py
```

Applies all improvements (balanced sampling, weighted loss, deeper unfreezing, MRI augmentation, LR warmup). Saves the best checkpoint to `weights/best_finetune.pth` based on validation F1. Prints Precision, Recall, F1, and Very Mild Dementia recall every epoch.

### 4. Evaluate metrics on the test set

```bash
python metrics.py
```

Automatically uses `weights/best_finetune.pth` if it exists, otherwise falls back to `model_checkpoint_22.pth`. Prints accuracy, precision, recall, F1-score per class and overall, plus a confusion matrix.

### 4. Predict for a single patient

Edit the patient ID (`i`) in `demeval.py` and run:

```bash
python demeval.py
```

---

## Dependencies

Install Python 3.8 or newer (3.9+ recommended).

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning framework (model, training loop, inference) |
| `torchvision` | Pre-trained DenseNet-121, image transforms, `ImageFolder` dataset |
| `numpy` | Numerical operations and array handling |
| `scikit-learn` | Metrics: accuracy, precision, recall, F1-score, AUROC, confusion matrix |
| `pandas` | Data manipulation and logging |
| `tqdm` | Training progress bar |
| `Pillow` | Image loading and processing |

All dependencies are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.
- `pandas` – imported but not heavily used; safe to keep.
- `scikit-learn` – for `roc_auc_score` (AUROC metrics).
- `tqdm` – progress bars during training epochs.
- `Pillow` – image backend used by `torchvision` transforms and `ImageFolder` (usually pulled as a dependency but listed explicitly here).

You will also need:

- A CUDA-capable GPU + compatible CUDA toolkit and GPU-enabled PyTorch build **or** CPU-only PyTorch.
- Image data organized as described below.

The script automatically selects the device with:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

So if CUDA is available, it will train on GPU; otherwise it runs on CPU (slower but still functional).

---

## Dataset Layout and Naming Assumptions

The script expects images in the following directory structure, **relative to the project root**:

```text
input/
  train/
    Moderate Dementia/
    Non Demented/
    Very mild Dementia/
  test/
    Moderate Dementia/
    Non Demented/
    Very mild Dementia/
```

Each subfolder under `train/` and `test/` corresponds to a class label. `torchvision.datasets.ImageFolder` will map classes in **alphabetical order** to numeric labels:

- `Moderate Dementia`  → label 0 (if alphabetically first)
- `Non Demented`       → label 1
- `Very mild Dementia` → label 2

The exact numeric mapping can be printed from `train_dataset.class_to_idx` if needed.

### Patient ID expectation

In the validation code, patient-level AUROC is computed by grouping slices/images by a **patient ID parsed from the filename**:

```python
filename = paths[i]
patient_id = filename.split("_")[1]
```

This assumes that each image file name contains a patient identifier as the second element when split by `_`, for example:

- `scan_0052_slice1.png`
- `MRI_1234_001.jpg`

If your filenames do not follow this pattern, the patient-level grouping will either fail or produce meaningless groupings. In that case you should adjust the `patient_id` extraction in `dementia.py`.

---

## Setting Up the Environment

1. **Clone or copy the project folder** to your machine (for example `e:\Miniproject-Dementia`).

2. **Create and activate a virtual environment** (recommended):

   ```powershell
   cd e:\Miniproject-Dementia
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install the required packages**:

   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install the correct PyTorch build for your hardware** (if needed):

   The generic `torch` / `torchvision` installation from `pip` works for CPU-only or some GPU setups, but for best results follow the official PyTorch installation instructions:

   - Visit https://pytorch.org
   - Select your OS, package manager (pip/conda), language (Python) and compute platform (CUDA version or CPU).
   - Run the provided install command instead of (or in addition to) plain `pip install torch torchvision`.

---

## How to Run Training

The main script is `dementia.py`. It:

- Loads a pretrained `DenseNet121` from `torchvision.models`.
- Freezes most layers and unfreezes the last dense block (`denseblock4`).
- Replaces the classifier with a new `nn.Linear` for 3 output classes.
- Applies data augmentation and normalization using `torchvision.transforms`.
- Trains for 30 epochs using `Adam` optimizer and `CosineAnnealingLR` scheduler.
- Evaluates on the `input/test` set each epoch, computing AUROC and patient-level AUROC.
- Saves a checkpoint after every epoch as `model_checkpoint_<epoch>.pth` and finally `model_checkpoint_final.pth`.

### Basic run command (Windows / PowerShell)

From the project root:

```powershell
cd e:\Miniproject-Dementia
# (optional but recommended)
.\.venv\Scripts\activate

python dementia.py
```

You should see the script print the detected device (`cuda` or `cpu`) and progress bars for training epochs (via `tqdm`). Checkpoints will be created in the project root.

### Customizing training (high-level notes)

Currently, key hyperparameters are hard-coded inside `dementia.py`:

- Number of epochs: `epochs = 30`
- Batch size: `batch_size=16` in the `DataLoader`.
- Learning rate: `lr=1e-5` in the `Adam` optimizer.
- Image size: `224x224`.

To change any of these, edit the corresponding lines in `dementia.py` before running.

---

## Using Existing Checkpoints

The folder `weights/` contains pre-trained checkpoints:

- `weights/model_checkpoint_17.pth`
- `weights/model_checkpoint_21.pth`
- `weights/model_checkpoint_22.pth`

And there is a directory `best_model18.pth/` that appears to contain the unpacked data of a `.pth` file.

The current `dementia.py` script **always starts from the ImageNet-pretrained DenseNet121** and does **not** automatically load any of these project-specific checkpoints. To use a saved checkpoint, you could (manually) modify `dementia.py` after creating the model:

```python
checkpoint = torch.load("weights/model_checkpoint_21.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
```

Place this code after the model (and classifier) are defined, and before training begins, if you want to resume from an existing checkpoint instead of starting from scratch.

> Note: This modification is not currently in the repository; it is just a suggested extension.

---

## Evaluation / Inference

The file `demeval.py` is currently empty, so there is no standalone evaluation or inference script provided.

To evaluate a trained model on a new dataset or on the test set outside the training loop, you can either:

- Use the validation logic already in `dementia.py` (it runs after each epoch), or
- Create a new script `demeval.py` that:
  - Recreates the same model architecture and transforms.
  - Loads a checkpoint with `torch.load` and `model.load_state_dict`.
  - Runs `model.eval()` and loops over images in a `DataLoader`.

If you want, I can help you implement such an evaluation script as a next step.

---

## Troubleshooting

- **CUDA not found / GPU not used**: Ensure you installed a CUDA-enabled PyTorch build and that your NVIDIA drivers and CUDA toolkit versions are compatible.
- **Out of memory errors (CUDA or CPU)**: Try lowering `batch_size` in the `DataLoader` (e.g., from 16 to 8 or 4).
- **File or directory not found**: Make sure the `input/train` and `input/test` folders exist and match the expected names exactly (case-sensitive on some systems).
- **Patient ID parsing errors**: If filenames do not contain `_` or do not follow the assumed pattern, adjust the `patient_id = filename.split("_")[1]` line in `dementia.py`.
