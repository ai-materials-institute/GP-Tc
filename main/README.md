# Feature Maker
Part of the GP-Tc pipeline. This script generates feature files for machine learning models from either existing graphlet pickles or raw CIF files.

## Usage

The script `Feature_Maker.py` supports two modes of operation, controlled by the `MODE` environment variable.

### 1. Generate from CIF Files (Recommended for new data)
To generate features from a folder of CIF files:

```bash
export MODE="CIF"
export CIF_DIR="/path/to/your/cifs"
export OUT_ROOT="/path/to/output/directory"

python Feature_Maker.py
```

**Requirements:**
- CIF filenames should contain the ICSD ID in the format `icsd_{id}.cif` (e.g., `compound_icsd_12345.cif`).
- The script will generate graphlets from scratch and then compute histograms.

### 2. Regenerate from Graphlet Pickles
To regenerate features from existing graphlet pickle files (e.g., to update histogram binning):

```bash
export MODE="PICKLE"  # Default
export GRAPHLET_DIR="/path/to/graphlet/pickles"
export OUT_ROOT="/path/to/output/directory"

python Feature_Maker.py
```

## Output
The script creates two subdirectories in `OUT_ROOT`:
- `Histogram_Features/`: Contains pickle files with histogram features and reduced formulas.
- `Graphlet_Data/`: Contains pickle files with the full graphlet objects.

A manifest file (`manifest_cif.pkl` or `manifest_pickle.pkl`) is also saved in `OUT_ROOT` summarizing the processing results.
