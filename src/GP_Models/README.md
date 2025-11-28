# GP-Tc Models Workflow

This directory contains the complete workflow for training and deploying Gaussian Process (GP) models for superconductivity prediction within the GP-Tc repository. The workflow consists of two main stages:
1.  **Training**: Using the tools in `sc_train_gp-main` to train GP models on histogram and symmetry features.
2.  **Prediction**: Using the trained models to make predictions on new data via `GP_Regression_pred.py` and `GP_Classification_pred.py`.

## 1. Training Models

The training logic is encapsulated in the `sc_train_gp-main` submodule. Please refer to [sc_train_gp-main/README.md](sc_train_gp-main/README.md) for detailed documentation on the training scripts and arguments.

To train models that are compatible with the prediction scripts in this directory, you should save the outputs to the `Trained Models` directory.

### Example: Training a Regressor
To train a regression model (e.g., for Critical Temperature $T_c$) and save it for later use:

```bash
cd sc_train_gp-main/scripts

python general_example_gp_regression.py \
    --save_data_dir "../../Trained Models/Regressor_4-2odr_all-sym" \
    --path_to_data_file "../data/regression_data_histogram&symmetry.pkl" \
    --hist_features_key "histogram_features" \
    --labels_key "Tc" \
    --symm_features_key "symmetry_features" \
    --list_of_symm_features_to_use 0 1 2 3 \
    --list_of_hist_features_to_use 13 18 26 30 \
    --n_epochs 32
```

### Example: Training a Classifier
To train a classification model (e.g., for Superconductor vs. Non-Superconductor) and save it:

```bash
cd sc_train_gp-main/scripts

python general_example_gp_classification.py \
    --save_data_dir "../../Trained Models/Classifier_2odr_all-sym" \
    --path_to_data_file "../data/classification_data_3DSCnonsc_labeled.pkl" \
    --hist_features_key "X_all" \
    --labels_key "label" \
    --symm_features_key "symm_features" \
    --n_epochs 32 \
    --use_oversampling True
```

> **Note**: The directory names `Regressor_4-2odr_all-sym` and `Classifier_2odr_all-sym` match the default `Model_Path` expected by the prediction scripts. If you choose different names, you will need to update the paths in the prediction scripts accordingly.

## 2. Making Predictions

Once the models are trained and saved in the `Trained Models` directory, you can use the prediction scripts to generate predictions on new batches of data.

### Regression Predictions
**Script**: `GP_Regression_pred.py`

This script loads the trained regression model and scaler, then processes batches of feature files to predict target values (e.g., $T_c$).

**Usage**:
The script is set up to run as a standalone module. It will:
1.  Load the model from `Trained Models/Regressor_4-2odr_all-sym` (default).
2.  Load feature files from `../ICSD_Features/Histogram_Features` (default).
3.  Generate predictions and save them to `../gp_regression_preds.pkl`.

```bash
python GP_Regression_pred.py
```

### Classification Predictions
**Script**: `GP_Classification_pred.py`

This script loads the trained classification model and scaler to predict probabilities (e.g., probability of being a superconductor).

**Usage**:
Similar to the regression script, it will:
1.  Load the model from `Trained Models/Classifier_2odr_all-sym` (default).
2.  Load feature files from `../ICSD_Features/Histogram_Features` (default).
3.  Generate predictions and save them to `../gp_classification_preds.pkl`.

```bash
python GP_Classification_pred.py
```

## Data Dependencies

The workflow relies on specific data locations for both training and prediction.

### Training Data
The training scripts expect data files (pickles) containing the training set (features and labels).
*   **Location**: `sc_train_gp-main/data/`
*   **Files**:
    *   `regression_data_histogram&symmetry.pkl` (for regression)
    *   `classification_data_3DSCnonsc_labeled.pkl` (for classification)

### Prediction Input Data
The prediction scripts (`GP_Regression_pred.py` and `GP_Classification_pred.py`) generate predictions for new materials. They expect individual feature files for each material.
*   **Location**: `../ICSD_Features/Histogram_Features` (Relative to `GP_Models/`)
*   **Format**: The directory should contain `.pkl` files for each material (e.g., `icsd_12345_histogram.pkl`).
*   **Content**: Each pickle file must contain a dictionary with keys matching those expected by the predictor (e.g., `reg_histograms`, `symmetry_feature`, `reduced_formula`).

## Directory Structure

```
GP_Models/
├── GP_Regression_pred.py       # Script for regression predictions
├── GP_Classification_pred.py   # Script for classification predictions
├── Trained Models/             # Directory where trained models are saved
│   ├── Regressor_.../          # Saved regression model files
│   └── Classifier_.../         # Saved classification model files
└── sc_train_gp-main/           # Submodule for model training
    ├── scripts/                # Training scripts
    └── data/                   # Training data
```

## Prerequisites

Ensure you have the following Python packages installed:
*   `torch`
*   `gpytorch`
*   `numpy`
*   `scikit-learn`

**Note**: A GPU is recommended for faster training and prediction, especially with large datasets or batch sizes.

## Critical Note: Feature Consistency

**Crucial**: The features used to train the model **MUST** match the features used during prediction.

1.  **Regression**:
    *   The `GP_Regression_pred.py` script hardcodes `Hist_Keep_Idxs = [13, 18, 26, 30]`.
    *   When training your regression model, you **must** use these same indices via the `--list_of_hist_features_to_use 13 18 26 30` argument.
    *   If you change the features during training, you must manually update `Hist_Keep_Idxs` in `GP_Regression_pred.py` to match.

2.  **Classification**:
    *   The `GP_Classification_pred.py` script extracts a specific slice of histogram features (indices `10:31`).
    *   Ensure your trained classification model expects this specific subset of features.

## Alternative: Neural Network Regression

In addition to GP models, this directory includes a neural network-based regression approach.

**Script**: `nn_regression.py`

This script implements a convolutional neural network (CNN) for predicting critical temperatures from histogram and symmetry features. It provides an alternative to GP models and may be useful for:
- Comparison with GP predictions
- Faster inference on large datasets
- Exploring different model architectures

**Key Features**:
- 1D convolutions over histogram bins
- Concatenation of symmetry features
- MSE loss with R² tracking
- Visualization of training progress

**Usage**:
The script is configured as a standalone training script. Modify the data path and hyperparameters directly in the file, then run:

```bash
python nn_regression.py
```

**Note**: This script requires modification of hardcoded paths and is provided as a research tool rather than a production pipeline component.

**Author**: Omri Lesser

**Authors**: Natalie Maus, Aaditya Panigrahi, Yanjun Liu, Omri Lesser
