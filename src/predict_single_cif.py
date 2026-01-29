#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single CIF GP Prediction Module.

Takes a single CIF file and returns 4 GP model predictions:
- Classification probability (is this a superconductor?)
- Classification uncertainty (std)
- Regression mean (predicted Tc)
- Regression uncertainty (std)

Usage:
    python predict_single_cif.py /path/to/structure.cif

Author: Aaditya Panigrahi
"""

from __future__ import annotations

import os
import sys
import json
import warnings
import argparse
import numpy as np

# Add current directory to path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "GP_Models"))

from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import spglib

# Local imports
from PYGraphlets import Create_Graphlets, Graphlet_AnalyzerFixedBins2D
from SplitGraphletSymmetryProcessor import (
    Config, _Util, SymmetryFeatureExtractor, load_bins_from_config
)
from GP_Models.GP_Classification_pred import (
    load_trained_class_gp_and_scaler,
    GP_prediction as GP_classification_prediction,
)
from GP_Models.GP_Regression_pred import (
    load_trained_reg_gp_and_scaler,
    GP_prediction as GP_regression_prediction,
    Hist_Keep_Idxs as REG_HIST_KEEP_IDXS,
)

warnings.filterwarnings("ignore")

# Classification uses histogram indices 10:31 (21 histograms)
CLAS_HIST_SLICE = slice(10, 31)


def extract_features_from_cif(cif_path: str) -> dict:
    """
    Extract all features needed for GP prediction from a single CIF file.
    
    Parameters
    ----------
    cif_path : str
        Path to the CIF file.
    
    Returns
    -------
    dict with keys:
        - 'clas_histograms': np.ndarray (1, 21, 20, 2) for classification
        - 'reg_histograms': np.ndarray (1, 4, 20, 2) for regression
        - 'symmetry_feature': np.ndarray (11,)
        - 'reduced_formula': str
    """
    cif_path = _Util.resolve_path(cif_path)
    config_dir = os.path.join(BASE_DIR, "../config")
    
    # Load atomic data
    with open(os.path.join(config_dir, "atomic_radii.json"), "r") as f:
        atomic_radii = json.load(f)
    with open(os.path.join(config_dir, "Filtered_atomic_features.json"), "r") as f:
        atomic_features_dict = json.load(f)
    
    # Load bin centers for both classification and regression
    bin_clas, bin_reg, feature_names = load_bins_from_config(
        bin_clas_path=os.path.join(config_dir, "bin_centers_classification.pkl"),
        bin_reg_path=os.path.join(config_dir, "bin_centers_regression.pkl"),
    )
    
    # Build graphlet from CIF
    structure = PMGStructure.from_file(cif_path).get_primitive_structure()
    graphlet = Create_Graphlets(structure, atomic_radii)
    graphlet.Get_1_site_graphlets()
    graphlet.Get_2_site_graphlets()
    graphlet.Get_3_site_graphlets()
    graphlet.get_features(atomic_features_dict)
    
    # Compute classification histograms
    analyser_clas = Graphlet_AnalyzerFixedBins2D(
        [graphlet],
        max_order=3,
        bin_centers_2d=bin_clas,
        feature_names=feature_names,
    )
    _, hist_array_clas, *_ = analyser_clas.get_histogram_features()
    
    # Compute regression histograms
    analyser_reg = Graphlet_AnalyzerFixedBins2D(
        [graphlet],
        max_order=3,
        bin_centers_2d=bin_reg,
        feature_names=feature_names,
    )
    _, hist_array_reg, *_ = analyser_reg.get_histogram_features()
    
    # Compute symmetry features
    sfe = SymmetryFeatureExtractor(
        excel_file=os.path.join(config_dir, "Space_group.xls"),
        sheet_name="Sheet3",
    )
    symm = sfe.from_cif(cif_path)
    
    # Get reduced formula
    reduced_formula = structure.composition.reduced_formula
    
    return {
        'clas_histograms': hist_array_clas[0, CLAS_HIST_SLICE, :, :],  # (21, 20, 2)
        'reg_histograms': hist_array_reg[0, REG_HIST_KEEP_IDXS, :, :],  # (4, 20, 2)
        'symmetry_feature': symm['feature_values'],  # (11,)
        'reduced_formula': reduced_formula,
    }


def predict_single_cif(cif_path: str, min_batch_size: int = 10) -> dict:
    """
    Predict superconductivity properties for a single CIF file.
    
    Parameters
    ----------
    cif_path : str
        Path to the CIF file.
    min_batch_size : int, default=10
        Minimum batch size for GP models. If less than this, duplicates
        the input to meet the requirement.
    
    Returns
    -------
    dict with keys:
        - 'classification_prob': float (0-1, probability of superconductor)
        - 'classification_std': float (uncertainty)
        - 'regression_mean': float (predicted Tc in K)
        - 'regression_std': float (uncertainty in Tc)
        - 'reduced_formula': str (chemical formula)
    """
    # Extract features from CIF
    features = extract_features_from_cif(cif_path)
    
    # Load models
    clas_scaler, clas_model, clas_likelihood, _, _ = load_trained_class_gp_and_scaler()
    reg_scaler, reg_model, reg_likelihood = load_trained_reg_gp_and_scaler()
    
    # Prepare input arrays - duplicate to meet min_batch_size
    clas_hist = np.tile(features['clas_histograms'][np.newaxis, :], (min_batch_size, 1, 1, 1))
    reg_hist = np.tile(features['reg_histograms'][np.newaxis, :], (min_batch_size, 1, 1, 1))
    symm_feat = np.tile(features['symmetry_feature'][np.newaxis, :], (min_batch_size, 1))
    
    # Run classification prediction
    clas_mean, clas_std = GP_classification_prediction(
        hist_feats=clas_hist,
        symm_feats=symm_feat,
        scaler=clas_scaler,
        model=clas_model,
        likelihood=clas_likelihood,
    )
    
    # Run regression prediction
    reg_mean, reg_std = GP_regression_prediction(
        hist_feats=reg_hist,
        symm_feats=symm_feat,
        scaler=reg_scaler,
        model=reg_model,
        likelihood=reg_likelihood,
    )
    
    # Return first result (all duplicates are identical)
    return {
        'classification_prob': float(clas_mean[0]),
        'classification_std': float(clas_std[0]),
        'regression_mean': float(reg_mean[0]),
        'regression_std': float(reg_std[0]),
        'reduced_formula': features['reduced_formula'],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict superconductivity properties from a CIF file."
    )
    parser.add_argument("cif_path", type=str, help="Path to the CIF file")
    parser.add_argument(
        "--min-batch-size", type=int, default=10,
        help="Minimum batch size for GP models (default: 10)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.cif_path):
        print(f"Error: CIF file not found: {args.cif_path}", file=sys.stderr)
        sys.exit(1)
    
    result = predict_single_cif(args.cif_path, min_batch_size=args.min_batch_size)
    
    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"GP-Tc Prediction Results")
        print(f"{'='*50}")
        print(f"Formula:               {result['reduced_formula']}")
        print(f"{'='*50}")
        print(f"Classification (SC?):")
        print(f"  Probability:         {result['classification_prob']:.4f}")
        print(f"  Uncertainty (std):   {result['classification_std']:.4f}")
        print(f"{'='*50}")
        print(f"Regression (Tc):")
        print(f"  Predicted Tc:        {result['regression_mean']:.2f} K")
        print(f"  Uncertainty (std):   {result['regression_std']:.2f} K")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
