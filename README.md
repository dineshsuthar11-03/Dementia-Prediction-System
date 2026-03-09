# Dementia Classification (DenseNet121)

This project trains a DenseNet121-based image classifier to distinguish between different dementia-related classes using MRI/brain scan images from the OASIS (Open Access Series of Imaging Studies) dataset.

---

## Model Architecture

- **Base Model**: DenseNet-121 (pretrained on ImageNet)
- **Modification**: Final classifier layer replaced with `nn.Linear(1024, 3)` for 3-class classification
- **Input Size**: 224×224 RGB images (grayscale MRI scans converted to 3-channel)
- **Output**: 3 class probabilities (softmax)

### Classification Classes
| Label | Class Name | Description |
|-------|------------|-------------|
| 0 | Moderate Dementia | Patients with moderate cognitive impairment |
| 1 | Non Demented | Healthy control subjects |
| 2 | Very mild Dementia | Patients with early-stage/mild cognitive impairment |

---

## Dataset Statistics

The dataset uses MRI brain scan images from the **OASIS-1** dataset. Each patient has multiple MRI slices (typically 60-70 slices per patient).

### Training Set (36 patients total)
| Class | Patient Count | Patient IDs |
|-------|---------------|-------------|
| Moderate Dementia | 14 | 0028, 0031, 0052, 0053, 0056, 0067, 0073, 0122, 0134, 0137, 0184, 0185, 0308, 0351 |
| Non Demented | 11 | 0004, 0005, 0006, 0007, 0009, 0010, 0011, 0012, 0013, 0014, 0017 |
| Very mild Dementia | 11 | 0021, 0022, 0023, 0039, 0041, 0042, 0046, 0060, 0066, 0082, 0084 |

### Test Set (8 patients total)
| Class | Patient Count | Patient IDs |
|-------|---------------|-------------|
| Moderate Dementia | 3 | 0028, 0031, 0035 |
| Non Demented | 2 | 0001, 0002 |
| Very mild Dementia | 3 | 0003, 0015, 0016 |

**Total Patients**: 44 unique patients (36 training + 8 test)

---

## How Prediction Works

The evaluation script (`demeval.py`) performs **patient-level prediction** using slice averaging:

1. **Load all MRI slices** for a single patient (e.g., 61 slices for patient OAS1_0031)
2. **Preprocess each slice**:
   - Convert grayscale to 3-channel RGB
   - Resize to 224×224 pixels
   - Normalize using ImageNet mean/std
3. **Forward pass** through DenseNet-121 → outputs `[num_slices, 3]` logits
4. **Apply softmax** to get probabilities for each slice
5. **Average probabilities** across all slices (ensemble voting)
6. **Final prediction**: Class with highest average probability

### Example Output
```
torch.Size([61, 3, 224, 224])  # 61 slices, 3 channels, 224x224
torch.Size([61, 3])            # 61 predictions, 3 classes each
Moderate Dementia              # Final averaged prediction
```

---

## Project Structure

- `dementia.py` – Training and validation loop using DenseNet121
- `demeval.py` – Patient-level inference/evaluation script
- `input/`
  - `train/`
    - `Moderate Dementia/`
    - `Non Demented/`
    - `Very mild Dementia/`
  - `test/`
    - `Moderate Dementia/`
    - `Non Demented/`
    - `Very mild Dementia/`
- `weights/`
  - `model_checkpoint_17.pth`
  - `model_checkpoint_21.pth`
  - `model_checkpoint_22.pth`
- `best_model18.pth/` – Unpacked PyTorch checkpoint directory
- `requirements.txt` – Python dependencies

The training script saves checkpoints as `model_checkpoint_<epoch>.pth` plus a final `model_checkpoint_final.pth`.

---

## Python and Library Requirements

Install Python 3.8 or newer (3.9+ recommended).

Core Python dependencies (also listed in `requirements.txt`):

- `torch` – PyTorch for deep learning.
- `torchvision` – pretrained DenseNet121 model and image transforms/datasets.
- `numpy` – numerical operations.
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
