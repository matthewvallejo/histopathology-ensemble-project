# Histopathology Ensemble Classifier

A desktop application (GUI) for classifying **breast cancer histology images** into four
tissue categories — **Normal, Benign, In Situ carcinoma, and Invasive carcinoma** — using
deep learning. It lets you train your own models, evaluate them, and run predictions on new
images without writing any code.

The app is built with Python's standard `tkinter` toolkit and uses TensorFlow/Keras under
the hood. A fully trained example model and its training data are included so you can try
everything immediately.

> ⚠️ **Disclaimer:** This is a portfolio / educational project. It is **not** a medical
> device and must **not** be used for diagnosis or any clinical decision-making.

---

## Table of contents

- [Dataset & credit](#dataset--credit)
- [What the app does](#what-the-app-does)
- [Installation](#installation)
- [Download the trained model](#download-the-trained-model)
- [Running the app](#running-the-app)
- [The three tabs](#the-three-tabs)
- [Try it: running the included example](#try-it-running-the-included-example)
- [How the model is trained](#how-the-model-is-trained)
- [Output files](#output-files)
- [Project structure](#project-structure)
- [Glossary (for non-AI folks)](#glossary-for-non-ai-folks)

---

## Dataset & credit

The model in `Example/` was trained on the **BACH (BreAst Cancer Histology)** dataset,
obtained from Kaggle:

> **BACH Breast Cancer Histology Images**
> https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images
> (uploaded by Kaggle user *truthisneverlinear*)

Full credit for the images goes to the dataset authors and the original
[ICIAR 2018 BACH Grand Challenge](https://iciar2018-challenge.grand-challenge.org/). If you
use this data, please cite the original work:

> G. Aresta et al., *"BACH: Grand challenge on breast cancer histology images,"*
> Medical Image Analysis, vol. 56, pp. 122–139, 2019.

The dataset contains **400 hematoxylin & eosin (H&E) stained microscopy images** —
100 per class — across the four categories listed above.

> **The images themselves are not included in this repository** (they are ~6.9 GB and belong
> to the dataset authors). Download them from the Kaggle link above and drop them into
> `Example/Photos/<class>/` — see [`Example/Photos/README.md`](Example/Photos/README.md) for
> the exact folder layout. The **trained model is distributed via
> [GitHub Releases](../../releases)** (it is ~236 MB) — download it once and place it in
> `Example/` to use the **Predict** tab. You only need the Kaggle dataset to re-train or run a
> full evaluation. See [Download the trained model](#download-the-trained-model) below.

---

## What the app does

The classifier looks at a microscopy image of breast tissue and predicts which of four
categories it belongs to:

| Class      | Meaning                                              |
|------------|------------------------------------------------------|
| `Normal`   | Healthy tissue                                       |
| `Benign`   | Non-cancerous abnormality                            |
| `InSitu`   | In situ carcinoma (cancer that is not spreading)        |
| `Invasive` | Invasive carcinoma (cancer that is spreading)          |

The app supports three workflows, one per tab: **train** new models, **evaluate** how good
they are, and **predict** on brand-new images.

---

## Installation

### Clone the repo

```bash
git clone <your-repo-url>
```

The repo is small — it contains only code, config, and metadata. The trained model is
downloaded separately (next section).

### Download the trained model

The trained model (`resnet50v2_seed42.keras`, ~236 MB) is **not** stored in the git repo;
it is published as a **[GitHub Release](../../releases)** asset. To use the **Predict** tab
with the included example:

1. Go to the [Releases page](../../releases) and download `resnet50v2_seed42.keras` from the
   latest release.
2. Place the file in the **`Example/`** folder, next to `ensemble_metadata.json`.

That's it — the app finds the model by scanning that folder. (You only need this for the
example model; models you train yourself are saved locally by the app.)

### Environment setup

You need **Python 3.12** (the version the project was built and tested with). Two setup
options are provided.

### Option A — pip + virtual environment (recommended on Windows)

```powershell
# Create the venv in your user folder (outside OneDrive) — the launchers find it here.
py -3.12 -m venv "$env:USERPROFILE\venvs\histopathology"
& "$env:USERPROFILE\venvs\histopathology\Scripts\python.exe" -m pip install -r requirements.txt
```

`requirements.txt` contains exact, tested versions of every package (TensorFlow 2.21,
Keras 3.14, NumPy 2.4, scikit-learn, OpenCV, albumentations, Pillow).

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate histopathology-slides
```

`environment.yml` installs Python 3.12 and then pulls the same pinned packages from
`requirements.txt`, so both setups produce an identical environment.

> **Note on the venv path:** `run_gui.bat` and `run_gui.ps1` locate the virtual environment
> automatically, in this order:
> 1. the `HISTO_VENV` environment variable (point it at any venv: `$env:HISTO_VENV = "C:\path\to\venv"`),
> 2. a `.venv` folder inside the project, then
> 3. the default `%USERPROFILE%\venvs\histopathology`.
>
> So no script editing is needed — just create the venv in one of those locations. It is kept
> **outside** OneDrive by default to avoid Windows long-path/sync issues. You can also skip the
> launchers entirely and run `python ensemble_gui.py` from any environment that has the
> requirements installed.

---

## Running the app

Any one of these works:

```powershell
# 1. Double-click run_gui.bat  (no console window)

# 2. From PowerShell
.\run_gui.ps1

# 3. Directly, using any Python that has the requirements installed
python ensemble_gui.py
```

A window titled **"Histopathology Ensemble Classifier"** opens with three tabs.

---

## The three tabs

### 1. Train Ensemble

Configure and launch training. Key settings:

- **Data Directory** — a folder containing one subfolder per class
  (`Normal/`, `Benign/`, `InSitu/`, `Invasive/`), each filled with images.
- **Output Directory** — where trained models and metadata are saved.
- **Model** — the network architecture (7 choices, see below).
- **Seeds** — comma-separated random seeds. **Each seed trains one model**; together they
  form the *ensemble* (e.g. `42, 123, 456` trains three models).
- **Loss Type**, **Class Weight**, **Stain Normalization**, **Layers to Unfreeze**, and the
  two **epoch** counts — all explained in [How the model is trained](#how-the-model-is-trained).

Training progress (per-epoch accuracy, loss, etc.) streams live into the log panel. When
finished, the app automatically evaluates the ensemble on a held-out test set and prints a
full report.

### 2. Evaluate Models

Point it at a folder containing trained models + `ensemble_metadata.json` and at a data
directory. It re-creates the same held-out test split used during training and reports
**precision, recall, F1-score, a confusion matrix, and per-class accuracy** for each model.

### 3. Predict

Classify new, unlabeled images:

- **Model Directory** — the folder with your trained `.keras` model(s).
- **Use Ensemble** — average the predictions of all models (more accurate), or pick a
  single model from the dropdown.
- **Test-Time Augmentation (TTA)** — runs each image through the model several times with
  small flips/rotations and averages the result (slower, slightly more robust).
- **Apply Stain Normalization** — should match how the model was trained (auto-filled from
  metadata).

Pick one or more images, click **Run**, and the app shows each prediction with a confidence
score and the full probability breakdown across all four classes. Results can be exported to
**CSV**.

---

## Try it: running the included example

The `Example/` folder ships with a **ready-to-use trained model**, so you can test
predictions without training anything.

```
Example/
├── Photos/                      # class folders for the BACH images (download from Kaggle)
│   ├── Normal/  Benign/  InSitu/  Invasive/
├── resnet50v2_seed42.keras      # trained ResNet50V2 model (seed 42) — download from Releases
└── ensemble_metadata.json       # describes how that model was trained
```

> The trained model isn't in the repo — download it from the
> [Releases page](../../releases) into `Example/` first (see
> [Download the trained model](#download-the-trained-model)).
> The dataset images aren't in the repo either. To run the steps that need images, first download
> the BACH images into `Example/Photos/<class>/` (see
> [`Example/Photos/README.md`](Example/Photos/README.md)). Any H&E histology image works for
> a quick Predict test — it doesn't have to be from BACH.

**To make a prediction:**

1. Open the app and go to the **Predict** tab.
2. Set **Model Directory** to the `Example` folder.
3. Click **Add Images** and pick one or more histology images (e.g. `.tif` files you
   downloaded into `Example/Photos/Invasive/`).
4. Click **Run Prediction**. You'll see each image classified with a confidence score.

**To evaluate the example model** (requires the downloaded dataset):

1. Go to the **Evaluate Models** tab.
2. Set the model directory to `Example` and the data directory to `Example/Photos`.
3. Run it to see the accuracy, confusion matrix, and per-class breakdown.

> The example is a *single-seed* model (seed 42), so its "ensemble" is just one network.
> Train with multiple seeds to see the ensemble effect.

---

## How the model is trained

This section is written for someone comfortable with Python but new to machine learning.
Here is the full pipeline, step by step.

### 1. Loading and cleaning the images

Every image is read, converted to RGB, and resized to the size the chosen network expects
(384×384 pixels for most models). Optionally, **stain normalization** is applied first:
H&E-stained slides vary a lot in color and brightness depending on the lab and scanner. The
app converts each image to the LAB color space and equalizes the **lightness** channel, which
evens out brightness/contrast differences so the model focuses on tissue structure rather
than staining quirks.

### 2. Splitting the data

The images are split into three groups, **stratified** so each group keeps the same class
balance:

- **Training set (70%)** — what the model learns from.
- **Validation set (15%)** — checked after every epoch to monitor progress and decide when
  to stop. The model never learns directly from this.
- **Test set (15%)** — locked away until the very end to measure true performance on data
  the model has never seen.

The split is computed once (using the first seed) so that **every model in the ensemble sees
exactly the same train/val/test data** — only their random initialization differs.

### 3. Transfer learning (standing on giants' shoulders)

Rather than learning to see from scratch, the app starts from a network **pre-trained on
ImageNet** (1.2 million everyday photos). That network already knows how to detect edges,
textures, and shapes. We keep that visual "backbone" and attach a small custom
**classification head** on top:

```
Pre-trained backbone (ResNet50V2, etc.)
        │
GlobalAveragePooling  →  Dense(512) → BatchNorm → ReLU → Dropout
                      →  Dense(256) → BatchNorm → ReLU → Dropout
                      →  Dense(4, softmax)   ← outputs 4 probabilities
```

The final `softmax` layer outputs four numbers that sum to 1 — the model's confidence in
each class.

**Available architectures:** ResNet50V2, ResNet101V2, EfficientNetV2S, EfficientNetV2M,
DenseNet201 (all 384×384), plus InceptionResNetV2 and Xception (299×299).

### 4. Two-stage training

Training happens in two stages:

- **Stage 1 — Frozen backbone.** The pre-trained layers are "frozen" (locked), and only the
  new head learns, at a relatively fast learning rate (`1e-3`). This teaches the head to use
  the existing features. Runs for up to *Epochs (frozen)* — default **50**.
- **Stage 2 — Fine-tuning.** The last *N* layers of the backbone (default **30**) are
  "unfrozen" and trained together with the head at a much slower learning rate (`1e-5`), so
  the network gently adapts its high-level features to histology images. Runs for up to
  *Epochs (fine-tune)* — default **10**.

Two safety mechanisms run in both stages:

- **Early stopping** — if validation accuracy stops improving, training halts and the best
  weights are restored (so it never overfits past its peak).
- **Learning-rate reduction** — if progress plateaus, the learning rate is automatically
  cut, allowing finer adjustments.

### 5. Handling class imbalance

Some classes can be harder or rarer than others. Two tools help:

- **Class weights** (`balanced`, `sqrt_balanced`, or `none`) make under-represented classes
  count more during training.
- **Focal loss** (the default, vs. plain cross-entropy) automatically focuses the model's
  attention on the hard, frequently-misclassified examples instead of the easy ones.

### 6. Data augmentation

To prevent the model from simply memorizing the (small) training set, each training image is
randomly transformed every time it's shown — flips, 90° rotations, shifts/zooms, mild
noise/blur, elastic and grid distortions, hue/brightness changes, and small random cut-outs
(via the `albumentations` library). The model effectively sees a slightly different image
each epoch, which improves generalization. The validation and test sets are **never**
augmented.

### 7. Building the ensemble

Each seed produces an independently trained model. At prediction time the app **averages the
softmax probabilities** of all models. Because each model makes slightly different mistakes,
averaging cancels out individual errors and usually yields higher, more stable accuracy than
any single model — this is the **ensemble** effect.

### 8. Evaluation

After training, the ensemble is measured on the untouched test set, reporting:

- **Precision / recall / F1-score** per class (via scikit-learn's `classification_report`),
- a **confusion matrix** (which classes get mistaken for which),
- **per-class accuracy**, and
- a **single-model vs. ensemble** comparison so you can see the improvement.

---

## Output files

After training, the **Output Directory** contains:

- `{ModelName}_seed{seed}.keras` — one saved model per seed (e.g. `ResNet50V2_seed42.keras`).
- `ensemble_metadata.json` — records the model name, seeds, image size, class names, and
  every training setting (stain normalization, loss type, epochs, etc.). The Evaluate and
  Predict tabs read this file to reload everything correctly.

---

## Project structure

```
Histopathology_Project/
├── ensemble_gui.py        # the entire application (GUI + training/evaluation/prediction)
├── requirements.txt       # exact pinned Python dependencies
├── environment.yml        # Conda environment (Python 3.12 + requirements.txt)
├── run_gui.bat            # double-click launcher (Windows, no console)
├── run_gui.ps1            # PowerShell launcher (shows console output)
├── .gitattributes        # line-ending normalization config
├── .gitignore
├── Example/               # example model metadata (model + dataset downloaded separately)
│   ├── Photos/            #   class folders; images come from Kaggle (see README)
│   ├── resnet50v2_seed42.keras   # download from GitHub Releases into this folder
│   └── ensemble_metadata.json
└── Output/                # default destination for newly trained models
```

---

## Glossary (for non-AI folks)

| Term | Plain-English meaning |
|------|----------------------|
| **Epoch** | One full pass over the training data. |
| **Transfer learning** | Reusing a network already trained on a huge image set as a starting point. |
| **Fine-tuning** | Gently re-training part of that reused network on your specific data. |
| **Backbone / head** | The reused feature-extractor vs. the small new layers that make the final decision. |
| **Softmax** | The final layer; turns raw scores into probabilities that sum to 100%. |
| **Augmentation** | Randomly altering training images so the model generalizes instead of memorizing. |
| **Ensemble** | Several models voting together; their averaged answer beats any single model. |
| **Validation / test set** | Data held back to check progress (validation) and to grade the final model (test). |
| **Confusion matrix** | A table showing which classes the model confuses with which. |
| **Seed** | A number that fixes the randomness, making a training run reproducible. |
| **TTA (Test-Time Augmentation)** | Predicting on several altered copies of one image and averaging, for robustness. |

---

*Built as a portfolio project. Histology images courtesy of the
[BACH dataset on Kaggle](https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images).*
