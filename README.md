# MAGIC Gamma Telescope — Binary Classification at Low False Positive Rate (pAUC)

Binary classification project on Monte Carlo simulated events from the MAGIC atmospheric Cherenkov gamma telescope.
Goal: discriminate **gamma (signal)** vs **hadron (background)** under **physics-driven constraints**, focusing on *very low false positive rates*.

## Why this project
In this domain, misclassifying background as signal is particularly costly.
Therefore, plain accuracy is not meaningful: model selection is driven by **ROC analysis** and **partial AUC (pAUC)** at small FPR thresholds.

## Dataset
- Source: MAGIC Gamma Telescope dataset (Kaggle)
- Instances: **19,020**
- Features: **10** continuous Hillas-like parameters (after dropping `ID`)
- Target:
  - `g` = gamma (signal) → mapped to 1
  - `h` = hadron (background) → mapped to 0
- Class distribution (approx.): g=12,332, h=6,688

Dataset link: https://www.kaggle.com/datasets/ppb00x/find-gamma-particles-in-magic-telescope

## Key idea: optimize where it matters (low-FPR regime)
Physics experiments often require operating points such as:
- FPR ≤ 0.01, 0.02, 0.05, 0.10

I define custom scorers using:
`roc_auc_score(max_fpr=...)`  
to optimize performance **only** in the low-background region of the ROC curve.

Metrics tracked:
- pAUC@0.01, pAUC@0.02, pAUC@0.05, pAUC@0.10

## Method overview
### 1) Preprocessing (leakage-safe)
All features are continuous but with different statistical behaviors:
- Heavy tails/outliers → `RobustScaler`
- More regular distributions → `StandardScaler`

A `ColumnTransformer` applies the transformations inside pipelines, preventing data leakage during CV.

### 2) Baselines (model family selection)
Compared baseline pipelines (same preprocessing, different classifiers):
- Logistic Regression (balanced)
- Perceptron (balanced)
- Random Forest
- XGBoost

Result: tree ensembles (Random Forest / XGBoost) dominate in the low-FPR region.

### 3) Modular pipeline exploration
Built a flexible `imblearn` pipeline with interchangeable components:
- resampling: passthrough / SMOTE / RandomOverSampler
- dimensionality reduction: passthrough / PCA
- classifier: Perceptron / Logistic Regression / Random Forest

### 4) Nested cross-validation (unbiased selection)
Used nested CV:
- Inner loop: `RandomizedSearchCV` optimizing pAUC@0.02
- Outer loop: `cross_validate` to estimate generalization without selection bias

### 5) Overfitting diagnosis + targeted regularization
Observed strong train-vs-test gap for Random Forest → variance-driven overfitting.
Performed a focused hyperparameter search (depth, min samples leaf/split, class weight, #trees)
to improve stability in the low-FPR regime.

### 6) Threshold selection at fixed FPR
Instead of using a default 0.5 threshold, I compute decision thresholds to satisfy physics constraints:
- choose threshold using OOF predictions (train) such that FPR ≤ 0.02
- evaluate confusion matrices on the held-out test set
- report counts + row-normalized percentages (signal efficiency vs background rejection)

## Results 
- Best model: **see notebook** 
- Best CV pAUC@0.02: `see notebook`
- Test pAUC@0.02: `see notebook`
- Example operating point (FPR ≤ 0.02):
  - background rejection / signal efficiency: `TODO` (from confusion matrix)

> For complete experiments, plots, and details, see: `notebooks/`.

