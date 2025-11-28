#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Maker
=============

This script processes a directory of existing graphlet pickle files to regenerate
and split their features into separate histogram and graphlet data files. It utilizes
the ``ICSDGraphletProcessor`` to recompute histograms based on current configuration
bins and saves the results in a structured output directory.

Key Features
------------
- Extracts ICSD IDs from filenames.
- Loads existing graphlet objects.
- Recomputes classification and regression histograms.
- Saves outputs to ``Histogram_Features`` and ``Graphlet_Data`` subdirectories.
- Uses parallel processing for efficiency.

Author
------
Aaditya Panigrahi
"""

import os
import sys
import re
import pickle
import warnings

# --- Configuration & Path Setup ---
# Determine absolute paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Assumes script is in <root>/main
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src to sys.path to import modules
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Define data directories (override with env vars if needed)
# Defaulting to a 'data' directory in the project root
CIF_DIR = os.getenv("CIF_DIR", os.path.join(PROJECT_ROOT, "data", "ICSD", "CIFS"))
GRAPHLET_DIR = os.getenv("GRAPHLET_DIR", os.path.join(PROJECT_ROOT, "data", "Pickled_ICSD_Histograms"))
OUT_ROOT = os.getenv("OUT_ROOT", os.path.join(PROJECT_ROOT, "ICSD_Features"))
MODE = os.getenv("MODE", "PICKLE")  # Options: "PICKLE" (default), "CIF"

# Import project modules
from SplitGraphletSymmetryProcessor import ICSDGraphletProcessor, load_bins_from_config  # noqa: E402
from ParallelRunner import ParallelRunner  # noqa: E402

warnings.filterwarnings("ignore")


def extract_icsd_id(filename: str) -> str:
    """
    Extract the ICSD ID (string) from filenames.
    Expects format ending in 'icsd_{id}.cif' or 'icsd_{id}' segments.
    """
    # Try matching end of string first for "icsd_{id}.cif"
    match = re.search(r"icsd_(\d+)\.cif$", filename)
    if match:
        return match.group(1)
    
    # Fallback to searching for any "icsd_{id}" segment
    match = re.search(r"icsd_(\d+)", filename)
    if match:
        return match.group(1)
        
    raise ValueError(f"No ICSD ID found in '{filename}'")


def BuildFromGraphlet(graph_path, bin_clas, bin_reg, feat_names, out_root=OUT_ROOT):
    """
    For a single graphlet pickle:
      - Extract ICSD ID from filename
      - Load graphlet object
      - Load corresponding CIF
      - Recompute histograms & save split outputs
    """
    try:
        # --- Extract ICSD ID and paths ---
        icsd_id = extract_icsd_id(os.path.basename(graph_path))
        cif_path = os.path.join(CIF_DIR, f"icsd_{icsd_id}.cif")

        # --- Load the graphlet pickle ---
        with open(graph_path, "rb") as f:
            data = pickle.load(f)
        graphlet_inst = data.get("materials") or data.get("graphlet") or data
        if isinstance(graphlet_inst, list):
            graphlet_inst = graphlet_inst[0]

        # --- Process ---
        proc = ICSDGraphletProcessor(
            cif_file=cif_path,
            graphlet=graphlet_inst,
            bin_centers_clas_2d=bin_clas,
            bin_centers_reg_2d=bin_reg,
            feature_names=feat_names,
        )

        hist_path, gpath = proc.pickle_split_outputs(out_root=out_root)
        return {"graph_path": gpath, "hist_path": hist_path, "ok": True}

    except Exception as e:
        print(f"Failed for {graph_path}: {e}")
        return {"graph_path": graph_path, "error": str(e), "ok": False}


def BuildFromCIF(cif_path, bin_clas, bin_reg, feat_names, out_root=OUT_ROOT):
    """
    For a single CIF file:
      - Build graphlet from scratch
      - Compute histograms
      - Save split outputs
    """
    try:
        # --- Process ---
        # graphlet=None forces calculation from CIF
        proc = ICSDGraphletProcessor(
            cif_file=cif_path,
            graphlet=None, 
            bin_centers_clas_2d=bin_clas,
            bin_centers_reg_2d=bin_reg,
            feature_names=feat_names,
        )

        hist_path, gpath = proc.pickle_split_outputs(out_root=out_root)
        return {"cif_path": cif_path, "graph_path": gpath, "hist_path": hist_path, "ok": True}

    except Exception as e:
        print(f"Failed for {cif_path}: {e}")
        return {"cif_path": cif_path, "error": str(e), "ok": False}


def main():
    bin_clas, bin_reg, feat_names = load_bins_from_config()
    
    os.makedirs(os.path.join(OUT_ROOT, "Histogram_Features"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "Graphlet_Data"), exist_ok=True)

    items_to_process = []
    process_func = None
    
    if MODE == "PICKLE":
        print(f"Mode: PICKLE (Regenerating from existing graphlets in {GRAPHLET_DIR})")
        if not os.path.exists(GRAPHLET_DIR):
            print(f"Error: Graphlet directory not found: {GRAPHLET_DIR}")
            return
        items_to_process = [
            os.path.join(GRAPHLET_DIR, f)
            for f in os.listdir(GRAPHLET_DIR)
            if f.endswith("_histogram.pkl")
        ]
        process_func = BuildFromGraphlet

    elif MODE == "CIF":
        print(f"Mode: CIF (Generating from scratch from CIFs in {CIF_DIR})")
        if not os.path.exists(CIF_DIR):
            print(f"Error: CIF directory not found: {CIF_DIR}")
            return
        items_to_process = [
            os.path.join(CIF_DIR, f)
            for f in os.listdir(CIF_DIR)
            if f.endswith(".cif") and "icsd_" in f
        ]
        process_func = BuildFromCIF
        
    else:
        print(f"Error: Unknown MODE '{MODE}'. Use 'PICKLE' or 'CIF'.")
        return

    print(f"Found {len(items_to_process)} files to process.")
    max_w = min(24, len(items_to_process) or 1, os.cpu_count() or 24)
    runner = ParallelRunner(executor="auto", max_workers=max_w, per_item_timeout_secs=3600)

    results = runner.run(
        process_func,
        items_to_process,
        broadcast_kwargs={"bin_clas": bin_clas, "bin_reg": bin_reg, "feat_names": feat_names, "out_root": OUT_ROOT},
    )

    # --- Save manifest ---
    manifest_path = os.path.join(OUT_ROOT, f"manifest_{MODE.lower()}.pkl")
    with open(manifest_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nManifest written to: {manifest_path}\n")


if __name__ == "__main__":
    main()