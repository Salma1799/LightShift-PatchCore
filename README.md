# LightShift-PatchCore

### Evaluating PatchCore Robustness Under Lighting Distribution Shifts

## 1. Project Overview

This project investigates the robustness of **PatchCore**, a visual anomaly detection method, under changes in lighting conditions.

In industrial environments, variations in illumination can affect the visual appearance of objects and potentially reduce the performance of anomaly detection systems.

The main objective is to evaluate how different photometric transformations influence PatchCore's ability to detect and localize anomalies.

The experimental study uses the **MVTec Anomaly Detection (MVTec AD)** dataset and focuses on three industrial object categories.

The project establishes a PatchCore baseline trained on original normal images and evaluates its performance on test images affected by synthetic lighting variations.

## 2. Dataset

The experiments are conducted using the **MVTec Anomaly Detection (MVTec AD)** dataset, a benchmark for unsupervised industrial anomaly detection.

The following categories are selected:

- `metal_nut`
- `screw`
- `transistor`

Each category contains:

- **Training images:** normal images located in `train/good`.
- **Test images:** normal and anomalous images located in `test`.
- **Ground-truth masks:** pixel-level annotations for anomalous test images, located in `ground_truth`.

### Dataset Organization

The original dataset is stored in:

```text
data/raw/mvtec_anomaly_detection/
├── metal_nut/
├── screw/
└── transistor/
```

The original data is preserved separately from the generated images to maintain a clear distinction between the original and processed datasets.

Dataset source: [MVTec Anomaly Detection Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

## 3. Data Preparation

The data preparation process includes:

1. Loading the three selected MVTec AD categories.
2. Organizing the original training and test images.
3. Generating synthetic lighting variations.
4. Associating the corresponding ground-truth masks with transformed test images.

The original dataset is retained in `data/raw/`, while the generated images and masks are stored in `data/processed/`.

### Ground-Truth Masks

For anomalous images, the original MVTec AD masks are reused because the photometric transformations do not modify the spatial location of defects.

For normal images (`good`), completely black masks are generated to indicate the absence of anomalies.

The processed masks follow the same category, transformation, severity and defect-folder organization as the corresponding test images.

## 4. Lighting Shift Generation

Five photometric transformations are implemented using **OpenCV** and **NumPy**.

| Transformation | Description |
|---|---|
| Brightness | Changes the overall image brightness. |
| Contrast | Modifies the difference between light and dark regions. |
| Gamma | Applies gamma correction to image intensities. |
| Color Temperature | Changes the color balance of the image. |
| Shading | Introduces a spatial illumination gradient. |

Each transformation is generated at three severity levels:

- **Severity 1:** low intensity.
- **Severity 2:** medium intensity.
- **Severity 3:** high intensity.

The transformations are applied to both training and test images to build the processed dataset.

The baseline evaluation uses the original normal training images as the reference and the transformed test images as evaluation inputs.

This setup makes it possible to investigate how lighting distribution shifts affect a model trained under original illumination conditions.

### Processed Data Organization

The transformed images are stored in:

```text
data/processed/
└── <category>/
    ├── train/
    │   └── <transformation>/
    │       └── severity_<level>/
    │
    ├── test/
    │   └── <transformation>/
    │       └── severity_<level>/
    │           ├── good/
    │           └── <defect_type>/
    │
    └── masks/
        └── <transformation>/
            └── severity_<level>/
                ├── good/
                └── <defect_type>/
```

## 5. PatchCore Baseline

### Model Description

PatchCore is an unsupervised visual anomaly detection method.

It extracts local feature representations from normal training images and builds a memory bank containing representative normal features.

During inference, features extracted from a test image are compared with the normal memory bank. Large distances from normal feature representations indicate potentially anomalous regions.

This approach provides both image-level anomaly detection and pixel-level anomaly localization.

### Model Configuration

The baseline uses the following configuration:

| Parameter | Value |
|---|---|
| Model | PatchCore |
| Backbone | Wide ResNet50-2 |
| Feature layers | `layer2`, `layer3` |
| Pretrained weights | Enabled |
| Coreset sampling ratio | 0.1 |
| Number of neighbors | 9 |
| Execution device | CPU |

A separate PatchCore model is trained for each selected category:

- `metal_nut`
- `screw`
- `transistor`

The models are trained using the original normal training images.

The trained model checkpoints are stored in:

```text
results/Patchcore/
```

## 6. Experimental Evaluation

After training, the baseline models are evaluated under different lighting conditions.

The experimental protocol covers:

- **3 object categories**
- **5 lighting transformations**
- **3 severity levels**

This gives a total of:

**3 × 5 × 3 = 45 evaluation configurations.**

For each configuration, the corresponding trained PatchCore model is evaluated on the transformed test images.

The original normal training images are retained as the reference for the model's normal memory bank.

The corresponding ground-truth masks are used to evaluate anomaly localization at the pixel level.

### Evaluation Metrics

Four metrics are collected:

| Metric | Purpose |
|---|---|
| Image AUROC | Measures the ability to distinguish normal and anomalous images. |
| Image F1-Score | Measures the balance between precision and recall at image level. |
| Pixel AUROC | Measures the ability to distinguish normal and anomalous pixels. |
| Pixel F1-Score | Measures the balance between precision and recall at pixel level. |

The F1-Score depends on the decision threshold used during evaluation, while AUROC evaluates discrimination across different thresholds.

### Results Storage

The evaluation results are progressively saved in:

```text
results/lighting_evaluation.csv
```

This file contains the metrics associated with each category, lighting transformation and severity level.

The results will be analyzed to identify the lighting conditions that have the greatest impact on PatchCore performance.

## 7. Project Structure

The main project organization is presented below:

```text
LightShift-PatchCore/
│
├── data/
│   ├── raw/
│   │   └── mvtec_anomaly_detection/
│   │
│   ├── processed/
│   │
│   └── splits/
│
├── src/
│   ├── augmentation/
│   │   └── lighting_shifts.py
│   │
│   ├── data/
│   │   ├── Load_dataset.py
│   │   └── copy_masks.py
│   │
│   ├── evaluation/
│   │   └── evaluate_lighting.py
│   │
│   └── models/
│       ├── patchcore_model.py
│       └── train_patchcore.py
│
├── configs/
│
├── notebooks/
│
├── results/
│   ├── Patchcore/
│   └── lighting_evaluation.csv
│
├── README.md
└── .gitignore
```

**Note:** The dataset, processed images, model checkpoints and virtual environment are not included in this simplified project structure.

## 8. Installation and Execution

### Requirements

- Python 3.11
- PyTorch
- Torchvision
- Anomalib
- OpenCV
- NumPy
- Pandas

### Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```cmd
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install torch torchvision
pip install anomalib opencv-python numpy pandas
```

For PyTorch installation, use a compatible build for the target hardware and operating system.

### Execution Order

All commands should be executed from the project root directory.

**1. Load the dataset**

```bash
python src/data/Load_dataset.py
```

**2. Generate lighting variations**

```bash
python src/augmentation/lighting_shifts.py
```

**3. Generate the corresponding masks**

```bash
python src/data/copy_masks.py
```

**4. Configure the PatchCore model**

```bash
python src/models/patchcore_model.py
```

**5. Train the PatchCore models**

```bash
python src/models/train_patchcore.py
```

**6. Evaluate the models under lighting shifts**

```bash
python src/evaluation/evaluate_lighting.py
```

**Note:** The training step requires the original MVTec AD dataset. The evaluation step requires the trained model checkpoints and the processed test images and masks.

The scripts use paths relative to the project root directory.

## 9. Experimental Results and Analysis

This section will be completed after the evaluation of all 45 configurations.

The analysis will focus on the following aspects:

- Performance degradation under different lighting transformations.
- The influence of transformation severity on anomaly detection.
- Differences in robustness across the three object categories.
- Comparison between original test conditions and transformed test conditions.
- Analysis of image-level detection and pixel-level localization performance.

Tables and graphical visualizations will be added to summarize the experimental findings.

## 10. Further Experimental Comparisons

The PatchCore baseline provides a reference for evaluating potential improvements in robustness to lighting distribution shifts.

Further experimental investigations within the project include:

- **Memory Bank Augmentation:** enriching the normal memory bank with feature representations from images captured under different lighting conditions.
- **Feature Alignment:** adapting feature representations to reduce differences between source and target lighting conditions.
- **Threshold Calibration:** adjusting the anomaly detection threshold using target-condition calibration data.

These approaches will be compared with the baseline to investigate their effectiveness under lighting distribution shifts.

## 11. References

- [MVTec Anomaly Detection Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [Anomalib Documentation](https://anomalib.readthedocs.io/)
- [PatchCore — Anomalib Implementation](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/patchcore/README.md)

---

**Project:** LightShift-PatchCore  
**Research Focus:** Industrial Visual Anomaly Detection and Robustness to Lighting Distribution Shifts