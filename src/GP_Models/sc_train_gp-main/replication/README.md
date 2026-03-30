# Replication Guide

This directory contains the replication-specific code for the Figure 2 GP regression experiments, while reusing the shared GP implementation already present in `sc_train_gp-main`.

For a faithful in-repo representation of the original `code base preparation/README.md`, see `CODE_BASE_PREPARATION.md`.

## Design

To keep the repository clean and avoid redundancy:

- shared code remains in the parent module:
  - `models/`
  - `kernels/`
  - `utils/`
  - `constants.py`
  - existing reusable scripts such as `scripts/train_gp_new_sg_featurization.py` and `scripts/manually_removing_features.py`
- replication-specific additions live only in this subtree:
  - `replication/scripts/`
  - `replication/data_loading_scripts/`

This means the old `code base preparation/` folder can be represented in-repo without duplicating the shared GP stack.

## Hardware

The GP training runs were performed on an `NVIDIA RTX A5000 GPU`.

## Environment

From the repository root:

```bash
pip install -r requirements.txt
```

For the GP training code, also make sure the packages used by `sc_train_gp-main` are installed, including:

```bash
pip install torch gpytorch numpy matplotlib scikit-learn pandas wandb
```

## Data Layout

The bundled replication datasets live under the repository data directory:

```text
data/replication/
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

Outputs will be written under:

```text
GP-Tc/src/GP_Models/sc_train_gp-main/replication/save_model_data/
```

## 3DSC Preprocessing Step

To generate `ICSD_subset.csv` from a directory of raw ICSD CIF files:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts

python 3DSC_subsetcsv.py \
  --cif-dir /3DSC/superconductors_3D/data/source/ICSD/raw/cifs \
  --output /3DSC/superconductors_3D/data/source/ICSD/raw/ICSD_subset.csv
```

Then run the official 3DSC ICSD workflow from the 3DSC repository:

```bash
make_3DSC -d ICSD -n 1 -dd superconductors_3D/data
```

## GP Replication Runs

Run the replication-only scripts from `replication/scripts/`.

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts
```

### Figure 2: `ord1` and `ordall` histogram regression

Use `train_gp_09_26_25.py` for the `ord1` and `ordall` histogram experiments, with and without symmetry features:

```bash
python train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features False
python train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features True
python train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features False
python train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features True
```

Outputs are written under:

```text
GP-Tc/src/GP_Models/sc_train_gp-main/replication/save_model_data/sc-train-gp-09-26-25/
```

### Figure 2: `ord2 + symm` regression

Use the existing parent script:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/scripts

python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too True --only_4_best_hist_features False
```

The bundled replication package includes dataset `15` (`good_data_symm_thre1%`). Dataset `14` (`good_data_symm_33d_thre1%`) remains optional and is not bundled here.

For the compressed `4 best ord2 features + symm` setting:

```bash
python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too True --only_4_best_hist_features True
```

For symmetry-only runs:

```bash
python train_gp_new_sg_featurization.py --dataset_id 15 --use_hist_features_too False --only_4_best_hist_features False
```

### Figure 2: `ord1+2, no symm` regression

Use the replication-only script:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/replication/scripts

python train_models_10_06_25_part2.py --dataset_id 21 --feature_subset_id 1 --n_epochs 32
python train_models_10_06_25_part2.py --dataset_id 21 --feature_subset_id 2 --n_epochs 32
```

`feature_subset_id` maps to:

- `1` → histogram features `0-30`
- `2` → histogram features `10-30`

### Figure 2: greedy feature removal

Use the existing parent script:

```bash
cd GP-Tc/src/GP_Models/sc_train_gp-main/scripts

python manually_removing_features.py --dataset_id 8 --n_epochs 32
```

This existing script now resolves the dataset from:

```text
data/replication/data/good_data_ord2_thre1%_with_symm.pkl
```

## Outputs

Depending on the run, the scripts save:

- `model_state.pt`
- `likelihood_state.pt`
- `train_preds.npy`
- `test_preds.npy`
- `performance_stats.json` or similar metrics files
- learned kernel parameters such as `emd_kernel_weights.npy`, `emd_kernel_lengthscales.npy`, and optionally `sg_kernel_lengthscale.npy`
- `train_result.png` and `test_result.png`

## Notes

- The replication subtree contains only the experiment-specific code that is not already present elsewhere in `sc_train_gp-main`.
- Shared GP implementations are intentionally reused from the parent module.
- Bundled replication datasets are stored in `data/replication/`.
- Local datasets and run outputs are ignored by `replication/.gitignore` to keep the repository clean.

## Compatibility Checks

The full GP training runs are GPU-oriented and were not executed end-to-end here.

The following CPU-safe checks were completed instead:

- Python syntax checks for the replication scripts and loaders
- Loader smoke tests against the bundled datasets in `data/replication/`
- Path-resolution checks for legacy loaders reused by `train_gp_new_sg_featurization.py` and `manually_removing_features.py`

This verifies import/path compatibility and dataset wiring, while final performance replication should still be run on a CUDA-enabled machine.
