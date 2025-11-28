# GP-Tc Gaussian Process Regression & Classification
**Author:** Natalie Maus

This repository provides tools for **Gaussian Process (GP)**–based modeling for both **regression** and **classification** tasks using **structured scientific features** such as symmetry‑based descriptors and histogram (Earth Mover’s Distance, EMD) representations. It serves as the training submodule for the GP-Tc pipeline.  
It includes utilities for:

- Data loading and preprocessing  
- GP model construction and training (via [GPyTorch](https://gpytorch.ai))  
- Performance evaluation (R², MAE, accuracy, etc.)  
- Visualization and saving of model outputs  

------------------------------------------------------

# Regression Script

**File:** `scripts/general_example_gp_regression.py`

This script performs **exact Gaussian Process regression** using a dataset that includes symmetry and/or histogram features.  
It automatically handles feature scaling, data splitting, model training, and saves results including learned kernel parameters, predictions, and performance plots.

### Key Features

- Flexible use of **symmetry**, **histogram**, or **combined** feature sets  
- Automatic scaling and reshaping for histogram features  
- Exact GP regression using a Gaussian likelihood  
- Full training pipeline (fit, evaluate, visualize, save)  
- Command‑line interface via `argparse`  

---

## Available Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--save_data_dir` | `str` | Directory where results, predictions, model weights, and plots will be saved. | **(required)** |
| `--path_to_data_file` | `str` | Path to the `.pkl` dataset file (e.g., `regression_data_histogram&symmetry.pkl`). | **(required)** |
| `--hist_features_key` | `str` | Key for accessing histogram features in the `.pkl` file. | `'histogram_features'` |
| `--labels_key` | `str` | Key for accessing target labels in the `.pkl` file. | `'Tc'` |
| `--symm_features_key` | `str` | Key for accessing symmetry features in the `.pkl` file. | `'symmetry_features'` |
| `--list_of_symm_features_to_use` | `list[int]` | Indices of symmetry features to use (e.g., `0 1 2`). | `[]` |
| `--list_of_hist_features_to_use` | `list[int]` | Indices of histogram features to use (e.g., `0 1 2`). | `[]` |
| `--random_split_seed` | `int` | Random seed for reproducibility in train/test splitting. | `2` |
| `--test_set_size` | `float` | Fraction of data to reserve for the test set. Must be in (0, 1). | `0.2` |
| `--n_epochs` | `int` | Number of training epochs for the GP model. | `32` |
| `--n_batches_emd_kernel` | `int` | Number of batches used when computing the EMD kernel. | `10` |
| `--lr` | `float` | Learning rate for the Adam optimizer. | `0.1` |

## Example Command

```bash
cd scripts 

python general_example_gp_regression.py \
    --save_data_dir ../save_model_data/gp_regression_outputs_v1 \
    --path_to_data_file "../data/regression_data_histogram&symmetry.pkl" \
    --n_epochs 32 \
    --list_of_symm_features_to_use 0 1 2 3 \
    --list_of_hist_features_to_use 0 1 2 
```

## More Example Testing Commands 
(for quickly testing to see that everything works correctly)

### 1. Regression - ONLY symmetry features
```bash
python general_example_gp_regression.py \
    --save_data_dir ../save_model_data/test_regression_symm_only \
    --path_to_data_file "../data/regression_data_histogram&symmetry.pkl" \
    --list_of_symm_features_to_use 0 1 5 \
    --n_epochs 2
```

### 2. Regression - ONLY histogram features
```bash
python general_example_gp_regression.py \
    --save_data_dir ../save_model_data/test_regression_hist_only \
    --path_to_data_file "../data/regression_data_histogram&symmetry.pkl" \
    --list_of_hist_features_to_use 2 4 \
    --n_epochs 2
```

### 3. Regression - BOTH symmetry + histogram features
```bash
python general_example_gp_regression.py \
    --save_data_dir ../save_model_data/test_regression_both \
    --path_to_data_file "../data/regression_data_histogram&symmetry.pkl" \
    --list_of_symm_features_to_use 0 1 8 \
    --list_of_hist_features_to_use 0 11 24 \
    --n_epochs 2
```


## Output Files

| File | Description |
|------|-------------|
| `performance_stats.json` | Contains regression metrics such as R² and MAE for both training and test sets. |
| `train_preds.npy` | Predicted values for the training set (NumPy array). |
| `test_preds.npy` | Predicted values for the test set (NumPy array). |
| `model_state.pt` | Saved PyTorch state dictionary for the trained GP model. |
| `likelihood_state.pt` | Saved state dictionary for the GPyTorch likelihood module. |
| `test_result.png` | Scatter plot of predicted vs. true test set values (R² and MAE in title). |
| `train_result.png` | Scatter plot of predicted vs. true training set values (R² and MAE in title). |
| `emd_kernel_weights.npy` | Learned weights for the histogram EMD kernel (if histogram features used). |
| `emd_kernel_lengthscales.npy` | Learned lengthscales for the histogram EMD kernel. |
| `sg_kernel_lengthscale.npy` | Learned lengthscales for the symmetry feature kernel (if symmetry features used). |



------------------------------------------------------

# Classification Script 

**File:** `scripts/general_example_gp_classification.py`

This script performs **Gaussian Process classification** using a dataset that includes symmetry and/or histogram features.  
It supports **variational inference with inducing points**, automatic oversampling for imbalanced classes, and batch training for large datasets.

### Key Features

- Supports **binary classification** using GP with Bernoulli likelihood  
- Flexible use of **symmetry**, **histogram**, or **combined** feature sets  
- Automatic **oversampling** of minority class for balanced training  
- Mini-batch training for scalability  
- Saves predictions, kernel parameters, and classification metrics  
- Command-line interface via `argparse`  

---

## Available Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--save_data_dir` | `str` | Directory where results, model weights, and predictions will be saved. | **(required)** |
| `--path_to_data_file` | `str` | Path to the `.pkl` dataset file. | **(required)** |
| `--hist_features_key` | `str` | Key for accessing histogram features in the `.pkl` file. | `'histogram_features'` |
| `--labels_key` | `str` | Key for accessing binary class labels in the `.pkl` file. | `'label'` |
| `--symm_features_key` | `str` | Key for accessing symmetry features in the `.pkl` file. | `'symmetry_features'` |
| `--list_of_symm_features_to_use` | `list[int]` | Indices of symmetry features to use (e.g., `0 1 2`). | `[]` |
| `--list_of_hist_features_to_use` | `list[int]` | Indices of histogram features to use (e.g., `0 1 2`). | `[]` |
| `--random_split_seed` | `int` | Random seed for reproducibility in train/test splitting. | `2` |
| `--test_set_size` | `float` | Fraction of data to reserve for the test set. Must be in (0, 1). | `0.2` |
| `--n_epochs` | `int` | Number of training epochs. | `32` |
| `--n_batches_emd_kernel` | `int` | Number of batches used when computing the EMD kernel. | `10` |
| `--lr` | `float` | Learning rate for the Adam optimizer. | `0.05` |
| `--n_inducing_pts` | `int` | Number of inducing points used in the variational GP. | `1024` |
| `--use_oversampling` | `bool` | Whether to balance class distribution by oversampling the minority class. | `True` |
| `--mini_batch_size` | `int` | Batch size for training. Reduce if running into GPU OOM issues. | `1024` |

---

## Example Command

```bash
cd scripts

python general_example_gp_classification.py \
    --save_data_dir ../save_model_data/gp_classification_outputs_v1 \
    --path_to_data_file "../data/classification_data_histogram&symmetry.pkl" \
    --list_of_symm_features_to_use 0 1 8 \
    --list_of_hist_features_to_use 10 11 12 16 \
    --n_epochs 32 \
    --use_oversampling True \
    --mini_batch_size 512 
```

## Output Files

| File | Description |
|------|-------------|
| `performance.json` | Classification metrics (e.g., accuracy, precision, recall) for both training and test sets. |
| `train_preds.npy` | Predicted class probabilities or binary labels for the training set (NumPy array). |
| `test_preds.npy` | Predicted class probabilities or binary labels for the test set (NumPy array). |
| `model_state.pt` | Saved PyTorch state dictionary for the trained GP classification model. |
| `likelihood_state.pt` | Saved GPyTorch likelihood state dictionary (BernoulliLikelihood). |
| `emd_kernel_weights.npy` | Learned weights for the histogram EMD kernel (if histogram features used). |
| `emd_kernel_lengthscales.npy` | Learned lengthscales for the histogram EMD kernel (if used). |
| `sg_kernel_lengthscale.npy` | Learned lengthscales for the symmetry feature kernel (if symmetry features used). |
