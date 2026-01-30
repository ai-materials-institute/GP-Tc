# GP-Tc: Superconductivity Prediction Pipeline

GP-Tc provides a complete end-to-end pipeline for predicting superconductivity properties of crystalline materials using Gaussian Process (GP) models with graphlet-based structural features and symmetry descriptors.

## Overview

The pipeline consists of three main stages:

1. **Feature Generation** (`main/`): Convert CIF files into ML-ready features
2. **Model Training** (`src/GP_Models/sc_train_gp-main/`): Train GP models on labeled data
3. **Prediction** (`src/GP_Models/`): Apply trained models to new materials

## Quick Start

### 1. Generate Features from CIF Files

```bash
cd main

# Set up paths
export MODE="CIF"
export CIF_DIR="/path/to/your/cifs"
export OUT_ROOT="../ICSD_Features"

# Run feature generation
python Feature_Maker.py
```

**Requirements:**
- CIF filenames must contain `icsd_{id}.cif` (e.g., `compound_icsd_12345.cif`)
- Outputs: `Histogram_Features/` and `Graphlet_Data/` directories

See [`main/README.md`](main/README.md) for detailed usage.

### 2. Train GP Models

```bash
cd src/GP_Models/sc_train_gp-main/scripts

# Train regression model (e.g., for Tc prediction)
python general_example_gp_regression.py \
    --save_data_dir "../../Trained Models/Regressor_4-2odr_all-sym" \
    --path_to_data_file "../../../data/regression_data_histogram&symmetry.pkl" \
    --list_of_hist_features_to_use 13 18 26 30 \
    --n_epochs 32

# Train classification model (superconductor vs. non-superconductor)
python general_example_gp_classification.py \
    --save_data_dir "../../Trained Models/Classifier_2odr_all-sym" \
    --path_to_data_file "../../../data/classification_data_3DSCnonsc_labeled.pkl" \
    --n_epochs 32 \
    --use_oversampling True
```

See [`src/GP_Models/sc_train_gp-main/README.md`](src/GP_Models/sc_train_gp-main/README.md) for all training options.

### 3. Make Predictions

```bash
cd src/GP_Models

# Regression predictions (e.g., Tc values)
python GP_Regression_pred.py

# Classification predictions (superconductor probability)
python GP_Classification_pred.py
```

Outputs are saved as `gp_regression_preds.pkl` and `gp_classification_preds.pkl`.

See [`src/GP_Models/README.md`](src/GP_Models/README.md) for the complete workflow.

## Repository Structure

```
.
├── config/                          # Configuration files
│   ├── atomic_radii.json           # Atomic radii for graphlet construction
│   ├── Filtered_atomic_features.json  # Element features
│   ├── Space_group.xls             # Symmetry group mappings
│   ├── bin_centers_classification.pkl  # Histogram bins for classification
│   └── bin_centers_regression.pkl      # Histogram bins for regression
│
├── main/                            # Feature generation scripts
│   ├── Feature_Maker.py            # Main feature generation script
│   └── README.md                   # Feature generation documentation
│
├── src/                             # Source code
│   ├── BatchGraphletSymmetryProcessor.py   # Batch processing utilities
│   ├── GraphletSymmetryProcessor.py        # Core graphlet processor
│   ├── SplitGraphletSymmetryProcessor.py   # Split output processor
│   ├── PYGraphlets.py              # Graphlet construction library
│   ├── ParallelRunner.py           # Parallel processing utilities
│   ├── ProgressBar.py              # Progress tracking
│   │
│   └── GP_Models/                  # GP model training & prediction
│       ├── GP_Regression_pred.py   # Regression prediction script
│       ├── GP_Classification_pred.py  # Classification prediction script
│       ├── Trained Models/         # Saved model checkpoints
│       ├── sc_train_gp-main/       # Training framework
│       └── README.md               # GP workflow documentation
│
└── data/                            # Data directory
    ├── classification_data_3DSCnonsc_labeled.pkl  # Training data for classification
    ├── regression_data_histogram&symmetry.pkl     # Training data for regression
    ├── ICSD/
    │   └── CIFS/                   # Input CIF files
    └── Pickled_ICSD_Histograms/    # Pre-computed graphlets (optional)
```

## Dependencies

### Core Requirements
- Python 3.8+
- `numpy`
- `pandas`
- `pymatgen`
- `spglib`
- `torch`
- `gpytorch`
- `scikit-learn`

### Installation
```bash
pip install -r requirements.txt
```

## Key Features

### Graphlet-Based Structural Representation
- **1-site graphlets**: Atomic composition and local symmetry
- **2-site graphlets**: Bond lengths and pair features
- **3-site graphlets**: Triplet angles and geometric descriptors
- **Histogram features**: Fixed-bin 2D histograms using Earth Mover's Distance (EMD)

### Symmetry Features
- Point group symmetry descriptors
- Site-specific symmetry operations
- Averaged symmetry vectors per structure

### Gaussian Process Models
- **Regression**: Predict continuous properties (e.g., critical temperature Tc)
- **Classification**: Binary classification (superconductor vs. non-superconductor)
- **Kernels**: Combined EMD kernel for histograms + RBF kernel for symmetry features
- **Scalability**: Variational inference with inducing points for large datasets

## Environment Variables

The pipeline supports configuration via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODE` | Feature generation mode: `"CIF"` or `"PICKLE"` | `"PICKLE"` |
| `CIF_DIR` | Directory containing input CIF files | `<project_root>/data/ICSD/CIFS` |
| `GRAPHLET_DIR` | Directory containing pre-computed graphlet pickles | `<project_root>/data/Pickled_ICSD_Histograms` |
| `OUT_ROOT` | Output directory for generated features | `<project_root>/ICSD_Features` |

## Authors

- **Aaditya Panigrahi** - Feature generation pipeline, graphlet processing
- **Yanjun Liu** - Graphlet symmetry processing
- **Natalie Maus** - GP model training framework
- **Krishnanand Mallayya** - PYGraphlets library
- **Omri Lesser** - Neural network regression models

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lesser2025learning,
  title={Learning to predict superconductivity},
  author={Lesser, Omri and Liu, Yanjun and Maus, Natalie and Panigrahi, Aaditya and Mallayya, Krishnanand and Schoop, Leslie M. and Gardner, Jacob R. and Kim, Eun-Ah},
  journal={arXiv preprint arXiv:2510.07373},
  year={2025}
}
```

**Paper**: [arXiv:2510.07373](https://arxiv.org/abs/2510.07373)

## License

[Add your license information here]

## Support

For questions or issues:
- Check the individual README files in each directory
- Review the docstrings in the Python modules
- Open an issue on the repository

## Workflow Summary

```mermaid
graph LR
    A[CIF Files] --> B[Feature_Maker.py]
    B --> C[Histogram Features]
    B --> D[Graphlet Data]
    C --> E[Train GP Models]
    D --> E
    E --> F[Trained Models]
    F --> G[GP Prediction Scripts]
    C --> G
    G --> H[Predictions]
```

1. **Input**: Crystallographic Information Files (CIF)
2. **Feature Extraction**: Graphlets + Symmetry → Histograms
3. **Training**: GP models learn from labeled data
4. **Prediction**: Apply models to new materials
5. **Output**: Predicted properties (Tc, superconductivity probability)
