# Code Base Preparation

This document is a faithful in-repo representation of the original `code base preparation/README.md`, adapted only where necessary so that the paths and commands match the integrated `GP-Tc` repository layout.

This package contains the code and data files used for the 3DSC preprocessing step and the Gaussian process (GP) experiments associated with Fig. 2(c), Fig. 2(d), and Fig. 2(e).

## Package contents

- `replication/scripts/`
  - `3DSC_subsetcsv.py`: parses raw ICSD CIF files and writes `ICSD_subset.csv` for the 3DSC ICSD pipeline.
  - `train_gp_09_26_25.py`: runs GP training for the `ord1` and `ordall` histogram experiments, with or without symmetry features.
  - `train_models_10_06_25_part2.py`: runs the `ord1+2, no symm` experiment.
- `replication/data_loading_scripts/`: dataset-specific loaders used by the replication-specific scripts above.
- `scripts/train_gp_new_sg_featurization.py`: runs GP training for the new symmetry-feature experiments, including the `ord2 + symm` setting.
- `scripts/manually_removing_features.py`: runs the greedy backward feature-removal procedure used for Fig. 2(e).
- `models/`: GP model definitions.
- `kernels/`: EMD and related kernels.
- `utils/`: scaling, metrics, prediction, logging, and helper utilities.
- `constants.py`: dataset-name mappings and feature-index constants.
- `data/replication/`: bundled replication data files expected by the included scripts.

## Environment

The GP scripts were written in Python and use `torch`, `gpytorch`, `numpy`, `matplotlib`, `scikit-learn`, `pandas`, and `wandb`.

A minimal setup is:

```bash
python -m venv venv
source venv/bin/activate
pip install torch gpytorch numpy matplotlib scikit-learn pandas wandb pymatgen
```

All GP scripts auto-select CUDA if available and otherwise fall back to CPU. The GP training runs were performed on an `NVIDIA RTX A5000 GPU`.

## 3DSC preprocessing

### Important note

This repository does **not** vendor the official 3DSC repository. To run the ICSD artificial-doping pipeline, the reader should install the official 3DSC repo separately, then use the script in this package to prepare the ICSD metadata file expected by 3DSC.

### Recommended 3DSC installation

```bash
git clone https://github.com/aimat-lab/3DSC.git
cd 3DSC
pip install .
```

If installation fails because of `setuptools>58`, use the older setuptools environment recommended by the 3DSC repository:

```bash
conda create --name 3DSC python=3.9 setuptools=58 pip
conda activate 3DSC
pip install .
```

### Preparing `ICSD_subset.csv`

`replication/scripts/3DSC_subsetcsv.py` expects the raw ICSD CIF files to be placed at:

```text
/3DSC/superconductors_3D/data/source/ICSD/raw/cifs/
```

and it writes the extracted metadata table to:

```text
/3DSC/superconductors_3D/data/source/ICSD/raw/ICSD_subset.csv
```

Run:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts
python 3DSC_subsetcsv.py
```

This script extracts the ICSD metadata fields required by the 3DSC ICSD workflow and writes a CSV with the expected column names.

### Running the 3DSC ICSD pipeline

After `ICSD_subset.csv` has been created and the raw CIF directory is in place, run the official 3DSC ICSD pipeline from the 3DSC repository:

```bash
make_3DSC -d ICSD -n 1 -dd superconductors_3D/data
```

This is the step that performs the standard 3DSC cleaning, matching, and artificial-doping workflow.

## Local data layout

The included scripts expect the following data layout relative to the repository root:

```text
GP-Tc/
└── data/
    └── replication/
        ├── data/
        │   └── good_data_ord2_thre1%_with_symm.pkl
        ├── data_09_26_2025/
        │   ├── good_data_ord1_20250926.pkl
        │   └── good_data_ordall_20250926.pkl
        ├── data_10_6_25/
        │   └── regression_data_histogram&symmetry.pkl
        └── data_new_sg_featurization/
            └── good_data_symm_thre1%.pkl
```

The training scripts use these paths through the loaders in `data_loading_scripts/` and `replication/data_loading_scripts/`.

Dataset `14` (`good_data_symm_33d_thre1%`) remains optional and is not bundled in this repository.

## How to run the GP experiments

Run the replication-specific GP scripts from `replication/scripts/` so that the relative import paths resolve correctly.

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts
```

### Fig. 2(c) and part of Fig. 2(d): `train_gp_09_26_25.py`

This script runs a single experiment per invocation. It supports:
- `--hist_features_ord_id ord1`
- `--hist_features_ord_id ordall`
- `--include_sg_features True/False`

Example commands:

```bash
python train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features False
python train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features True
python train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features False
python train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features True
```

Outputs are written under:

```text
src/GP_Models/sc_train_gp-main/replication/save_model_data/sc-train-gp-09-26-25/
```

Each run saves model weights, likelihood state, train/test predictions, learned kernel parameters, and result plots.

### `ord2 + symm`: `train_gp_new_sg_featurization.py`

This script uses the new symmetry-feature datasets and supports dataset IDs:
- `14` -> `good_data_symm_33d_thre1%`
- `15` -> `good_data_symm_thre1%`

It also supports:
- `--use_hist_features_too True/False`
- `--only_4_best_hist_features True/False`

For the `ord2 + symm` setting, run one of the following depending on which symmetry-feature dataset is being used:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/scripts
python train_gp_new_sg_featurization.py --dataset_id 14 --use_hist_features_too True --only_4_best_hist_features False
python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too True --only_4_best_hist_features False
```

For the compressed `4 best ord2 features + symm` setting:

```bash
python train_gp_new_sg_featurization.py --dataset_id 14 --use_hist_features_too True --only_4_best_hist_features True
python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too True --only_4_best_hist_features True
```

For symmetry-only runs:

```bash
python train_gp_new_sg_featurization.py --dataset_id 14 --use_hist_features_too False --only_4_best_hist_features False
python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too False --only_4_best_hist_features False
```

Outputs are written under:

```text
src/GP_Models/sc_train_gp-main/save_model_data/sc-train-gp-new-sg-fts/
```

### `ord1+2, no symm`: `train_models_10_06_25_part2.py`

This script uses the regression dataset that combines histogram and symmetry data, and it keeps only the specified histogram features in the loader used by the script.

Run:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts
python train_models_10_06_25_part2.py --dataset_id 21 --feature_subset_id 1
python train_models_10_06_25_part2.py --dataset_id 21 --feature_subset_id 2
```

This script reads:

```text
data/replication/data_10_6_25/regression_data_histogram&symmetry.pkl
```

Outputs are written under the save directory configured in the script.

### Fig. 2(e): greedy feature removal

`manually_removing_features.py` performs greedy backward elimination on the 32-feature `good_data_ord2_thre1%_with_symm` dataset. At each step, it removes the feature whose exclusion gives the highest test-set R2, then repeats.

Run:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/scripts
python manually_removing_features.py --dataset_id 8 --n_epochs 32
```

This script reads:

```text
data/replication/data/good_data_ord2_thre1%_with_symm.pkl
```

Outputs are written under:

```text
src/GP_Models/sc_train_gp-main/save_model_data/removing-features-exact-gp/
```

The saved outputs include per-step train/test predictions, learned kernel parameters, summary CSV files, and diagnostic plots.

## Notes on outputs

Across the GP scripts, the output directories typically contain:
- `model_state.pt`
- `likelihood_state.pt`
- `train_preds.npy`
- `test_preds.npy`
- learned kernel weights and lengthscales
- train/test scatter plots
- run-specific summary files

## Running order summary

A reader who wants to reproduce the materials in this package can use the following order:

1. Install the separate official 3DSC repository.
2. Place the ICSD raw CIF files in the 3DSC ICSD raw directory.
3. Run `replication/scripts/3DSC_subsetcsv.py` to create `ICSD_subset.csv`.
4. Run the official 3DSC ICSD pipeline with `make_3DSC -d ICSD -n 1 -dd superconductors_3D/data`.
5. Run the GP experiment scripts using the commands above.
