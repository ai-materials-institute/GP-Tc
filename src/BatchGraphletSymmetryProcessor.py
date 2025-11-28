#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined module + batch runner
==============================

This file contains:

1) The combined crystallographic ML utilities module:
   - Config
   - SymmetryFeatureExtractor
   - ICSDGraphletProcessor  (graphlets + histograms + mandatory symmetry feature)

2) A batch runner class:
   - BatchGraphletRunner (process a list of CIFs and save per-file pickles)

Notes
-----
- Paths expand ~ and environment variables where defined in Config.
- PYGraphlets is required for the graphlet/histogram parts.

Author
------
Aaditya Panigrahi
"""

from __future__ import annotations

__all__ = ["Config", "SymmetryFeatureExtractor", "ICSDGraphletProcessor", "BatchGraphletRunner"]
__version__ = "0.11.7"

# -------------------- aggregated imports --------------------
import json
import os
import pickle
import warnings
import traceback
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from PYGraphlets import Create_Graphlets, Graphlet_AnalyzerFixedBins2D
# ------------------------------------------------------------


# ============================================================
# Config and Utility Classes
# ============================================================
class Config:
    """
    Centralized configuration defaults.

    Environment variable overrides
    ------------------------------
    EXCEL_MAPPING_XLS : str
        Path to Excel mapping (default: "../config/Space_group.xls").
    EXCEL_SHEET : str
        Sheet name or 0-based index (default: "Sheet3").
    ATOMIC_RADII_JSON : str
        Path to atomic radii JSON.
    ATOMIC_FEATURES_JSON : str
        Path to per-element feature JSON.
    """

    # --- Excel mapping ---
    EXCEL_MAPPING_XLS: str = os.path.expanduser(
        os.getenv("EXCEL_MAPPING_XLS", "../config/Space_group.xls")
    )
    EXCEL_SHEET: str = os.getenv("EXCEL_SHEET", "Sheet3")

    # --- Graphlet configs ---
    ATOMIC_RADII_JSON: str = os.path.expanduser(
        os.getenv("ATOMIC_RADII_JSON", "../config/atomic_radii.json")
    )
    ATOMIC_FEATURES_JSON: str = os.path.expanduser(
        os.getenv("ATOMIC_FEATURES_JSON", "../config/Filtered_atomic_features.json")
    )


class _Util:
    """Internal helper utilities."""

    @staticmethod
    def resolve_path(p: Optional[str]) -> Optional[str]:
        """Expand ~ and environment vars; return absolute path or None."""
        if p is None:
            return None
        return os.path.abspath(os.path.expanduser(os.path.expandvars(p)))

    @staticmethod
    def sheet_arg(sheet: Union[str, int]) -> Union[str, int]:
        """Normalize sheet spec for pandas (name or numeric index)."""
        if isinstance(sheet, int):
            return sheet
        s = str(sheet).strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                return sheet
        return sheet


# ============================================================
# SymmetryFeatureExtractor
# ============================================================
class SymmetryFeatureExtractor:
    """
    Compute averaged symmetry feature vectors from CIFs using a point-group map.

    Parameters
    ----------
    excel_file : str, default=Config.EXCEL_MAPPING_XLS
        Path to Excel workbook with the mapping.
    sheet_name : str or int, default=Config.EXCEL_SHEET
        Worksheet name or 0-based index.
    symbol_col_index : int, default=1
        Zero-based column index for point-group symbol.
    first_feat_col_index : int, default=2
        Zero-based column index for first feature column.

    Returns from `from_cif`
    -----------------------
    dict
        ``{"feature_names": list, "feature_values": np.ndarray}``
    """

    def __init__(
        self,
        excel_file: str = Config.EXCEL_MAPPING_XLS,
        sheet_name: Union[str, int] = Config.EXCEL_SHEET,
        symbol_col_index: int = 1,
        first_feat_col_index: int = 2,
    ):
        df = pd.read_excel(
            _Util.resolve_path(excel_file),
            sheet_name=_Util.sheet_arg(sheet_name),
            header=0,
        )

        self._feature_names: List[Any] = df.columns[first_feat_col_index:].tolist()
        symbols = [str(x) for x in df.iloc[:, symbol_col_index].tolist()]
        rows = df.iloc[:, first_feat_col_index:].values.tolist()
        self._pg_feature_map: Dict[str, List[float]] = {s: r for s, r in zip(symbols, rows)}

    def from_cif(
        self,
        cif_path: str,
        *,
        symprec: float = 1e-5,
        tol: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Compute averaged symmetry feature vector for a CIF.

        Parameters
        ----------
        cif_path : str
            Path to a CIF file.
        symprec : float, default=1e-5
            Symmetry precision for :class:`pymatgen.symmetry.analyzer.SpacegroupAnalyzer`.
        tol : float, default=1e-5
            Absolute tolerance for identifying site-fixing operations.

        Returns
        -------
        dict
            ``{"feature_names": list, "feature_values": np.ndarray}``
        """
        struct = PMGStructure.from_file(_Util.resolve_path(cif_path))
        sga = SpacegroupAnalyzer(struct, symprec=symprec)
        sym_ops = sga.get_symmetry_operations()

        n_feat = len(self._feature_names)
        site_feature_list: List[List[float]] = []

        for site in struct:
            fc = site.frac_coords % 1.0
            ops_fix = [op for op in sym_ops if np.allclose(op.operate(fc) % 1.0, fc, atol=tol)]

            if not ops_fix:
                ptg_symbol = "1"
            else:
                rot_mats = [np.rint(op.rotation_matrix).astype(int) for op in ops_fix]
                ptg_symbol, _, _ = spglib.get_pointgroup(rot_mats)

            site_feature_list.append(self._pg_feature_map.get(ptg_symbol, [0.0] * n_feat))

        avg = np.asarray(site_feature_list, dtype=float).mean(axis=0) if site_feature_list else np.zeros(n_feat)
        return {"feature_names": self._feature_names, "feature_values": avg}


# ============================================================
# ICSDGraphletProcessor
# ============================================================
class ICSDGraphletProcessor:
    """
    Build graphlets and dual fixed-bin 2D histograms, with **mandatory symmetry feature**.

    For each CIF, builds a graphlet, computes dual (classification & regression)
    histograms, and attaches the averaged symmetry vector from
    :class:`SymmetryFeatureExtractor` to the payload.

    Parameters
    ----------
    cif_file : str
        Path to the CIF file.
    atomic_radii_path : str, default=Config.ATOMIC_RADII_JSON
        Path to atomic radii JSON.
    atomic_features_path : str, default=Config.ATOMIC_FEATURES_JSON
        Path to per-element features JSON.
    bin_centers_clas_2d : Any
        Bin centers for classification histograms.
    bin_centers_reg_2d : Any
        Bin centers for regression histograms.
    feature_names : Any
        Feature names for histograms.
    excel_file : str, default=Config.EXCEL_MAPPING_XLS
        Path to the Excel mapping file.
    sheet_name : str or int, default=Config.EXCEL_SHEET
        Worksheet name or 0-based index.
    symbol_col_index : int, default=1
        Point-group symbol column index.
    first_feat_col_index : int, default=2
        First feature column index.
    """

    REQUIRED_ATTRS = ("one_site_features", "two_site_features", "three_site_features")

    def __init__(
        self,
        cif_file: str,
        *,
        graphlet: Optional[Any] = None,
        atomic_radii_path: str = Config.ATOMIC_RADII_JSON,
        atomic_features_path: str = Config.ATOMIC_FEATURES_JSON,
        bin_centers_clas_2d: Any = None,
        bin_centers_reg_2d: Any = None,
        feature_names: Any = None,
        validate_graphlet: bool = True,
        raise_on_error: bool = False,
        excel_file: str = Config.EXCEL_MAPPING_XLS,
        sheet_name: Union[str, int] = Config.EXCEL_SHEET,
        symbol_col_index: int = 1,
        first_feat_col_index: int = 2,
    ):
        self.cif_file = _Util.resolve_path(cif_file)
        self.atomic_radii_path = _Util.resolve_path(atomic_radii_path)
        self.atomic_features_path = _Util.resolve_path(atomic_features_path)
        self.bin_centers_clas_2d = bin_centers_clas_2d
        self.bin_centers_reg_2d = bin_centers_reg_2d
        self.feature_names = feature_names

        self.graphlet: Optional[Any] = graphlet
        self.histogram: Optional[Dict[str, Any]] = None
        self.ok: bool = False
        self.error: Optional[Exception] = None

        # Symmetry extractor config (mandatory)
        self._excel_file = _Util.resolve_path(excel_file)
        self._sheet_name = sheet_name
        self._symbol_col_index = symbol_col_index
        self._first_feat_col_index = first_feat_col_index

        try:
            # ---- Input validation
            if self.bin_centers_clas_2d is None:
                raise ValueError("bin_centers_clas_2d must be provided.")
            if self.bin_centers_reg_2d is None:
                raise ValueError("bin_centers_reg_2d must be provided.")
            if self.feature_names is None:
                raise ValueError("feature_names must be provided.")

            # ---- Load atomic data
            self.atomic_radii, self.atomic_features_dict = self._load_atomics(
                self.atomic_radii_path, self.atomic_features_path
            )

            # ---- Build graphlet
            if self.graphlet is None:
                self.graphlet = self._make_graphlet(self.cif_file)

            if validate_graphlet and not self._valid_graphlet(self.graphlet):
                raise RuntimeError("Graphlet object missing required attributes.")

            # ---- Compute histograms
            hist_names, hist_array_clas = self._compute_histogram(self.graphlet, self.bin_centers_clas_2d)
            _,          hist_array_reg  = self._compute_histogram(self.graphlet, self.bin_centers_reg_2d)

            # ---- Always compute symmetry features
            sfe = SymmetryFeatureExtractor(
                excel_file=self._excel_file,
                sheet_name=_Util.sheet_arg(self._sheet_name),
                symbol_col_index=self._symbol_col_index,
                first_feat_col_index=self._first_feat_col_index,
            )
            symm = sfe.from_cif(self.cif_file)

            # ---- Reduced chemical formula
            structure = PMGStructure.from_file(self.cif_file)
            chemical_formula = structure.composition.reduced_formula

            self.histogram = {
                "graphlet": self.graphlet,
                "cif_path": self.cif_file,
                "chemical_formula": chemical_formula,  # << added
                "clas_histograms": hist_array_clas,
                "reg_histograms": hist_array_reg,
                "hist_feat_name": hist_names,
                "symmetry_feature": symm["feature_values"],
                "symmetry_feature_names": symm["feature_names"],
            }
            self.ok = True

        except Exception as e:
            self.error = e
            self.ok = False
            if raise_on_error:
                raise
            warnings.warn(f"ICSDGraphletProcessor init failed: {e}", RuntimeWarning)

    # ---------------- Internal helpers ----------------

    def _load_atomics(
        self, atomic_radii_path: str, atomic_features_path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load atomic radii and per-element features from JSON files."""
        with open(_Util.resolve_path(atomic_features_path), "r") as f:
            atomic_features_dict = json.load(f)
        with open(_Util.resolve_path(atomic_radii_path), "r") as f:
            atomic_radii = json.load(f)
        return atomic_radii, atomic_features_dict

    def _valid_graphlet(self, g: Any) -> bool:
        """Return True if required graphlet attributes are present."""
        return (g is not None) and all(hasattr(g, a) for a in self.REQUIRED_ATTRS)

    def _make_graphlet(self, cif_file: str):
        """Construct and populate a PYGraphlets graphlet from a CIF path."""
        structure = PMGStructure.from_file(cif_file).get_primitive_structure()
        graphlets = Create_Graphlets(structure, self.atomic_radii)
        graphlets.Get_1_site_graphlets()
        graphlets.Get_2_site_graphlets()
        graphlets.Get_3_site_graphlets()
        graphlets.get_features(self.atomic_features_dict)
        return graphlets

    def _compute_histogram(self, graphlet_obj: Any, bin_centers_2d: Any):
        """Compute fixed-bin 2D histogram features for a graphlet."""
        analyser = Graphlet_AnalyzerFixedBins2D(
            [graphlet_obj],
            max_order=3,
            bin_centers_2d=bin_centers_2d,
            feature_names=self.feature_names,
        )
        hist_names, hist_array, *_ = analyser.get_histogram_features()
        return hist_names, hist_array

    # ---------------- Public API ----------------

    def pickle_histogram(self, out_path: str) -> str:
        """Serialize the histogram payload to a pickle file."""
        if not self.ok or self.histogram is None:
            raise RuntimeError("Nothing to pickle: initialization did not complete successfully.")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(self.histogram, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

    def __repr__(self) -> str:
        """Human-readable summary string with origin and status."""
        status = "ok" if self.ok else f"error={type(self.error).__name__}" if self.error else "not-ready"
        return f"<ICSDGraphletProcessor from=cif status={status}>"


# ============================================================
# Batch runner class
# ============================================================
class BatchGraphletRunner:
    """
    Batch runner for ICSDGraphletProcessor.

    Parameters
    ----------
    out_dir : str, optional
        Directory to save output pickles. Default: "../output".
    bin_clas_path : str, optional
        Pickle with classification 2D bin centers (expects key "bin_centers_list").
    bin_reg_path : str, optional
        Pickle with regression 2D bin centers (expects key "bin_centers_list").
    atomic_radii_json : str, optional
        Path to atomic radii JSON. Default: "../config/atomic_radii.json".
    atomic_features_json : str, optional
        Path to per-element features JSON. Default: "../config/Filtered_atomic_features.json".
    excel_mapping_xls : str, optional
        Path to Excel mapping for symmetry features. Default: "../config/Space_group.xls".
    excel_sheet : str or int, optional
        Sheet name/index for the Excel mapping. Default: "Sheet3".
    """

    def __init__(
        self,
        out_dir: str = "../output",
        bin_clas_path: str = "../config/bin_centers_classification.pkl",
        bin_reg_path: str = "../config/bin_centers_regression.pkl",
        atomic_radii_json: str = "../config/atomic_radii.json",
        atomic_features_json: str = "../config/Filtered_atomic_features.json",
        excel_mapping_xls: str = "../config/Space_group.xls",
        excel_sheet: Optional[str] = "Sheet3",
    ):
        self.out_dir = out_dir
        self.bin_clas_path = bin_clas_path
        self.bin_reg_path = bin_reg_path
        self.atomic_radii_json = atomic_radii_json
        self.atomic_features_json = atomic_features_json
        self.excel_mapping_xls = excel_mapping_xls
        self.excel_sheet = excel_sheet

        os.makedirs(self.out_dir, exist_ok=True)
        self._require_files(
            [
                self.bin_clas_path,
                self.bin_reg_path,
                self.atomic_radii_json,
                self.atomic_features_json,
                self.excel_mapping_xls,
            ]
        )
        self.bin_clas, self.bin_reg, self.feature_names = self._load_bins(self.bin_clas_path, self.bin_reg_path)
        assert len(self.feature_names) == len(self.bin_clas), \
            f"feature_names ({len(self.feature_names)}) must match classification bins ({len(self.bin_clas)})."

    # ---------- public API ----------

    def process(self, cif_list: List[str]) -> Dict[str, str]:
        """
        Process a list of CIF files and save <name>_hist.pkl for each.

        Parameters
        ----------
        cif_list : list of str
            Paths to CIF files.

        Returns
        -------
        dict
            Mapping cif_path -> "success" or error string.
        """
        results: Dict[str, str] = {}
        for cif_path in cif_list:
            name = os.path.splitext(os.path.basename(cif_path))[0]
            out_pkl = os.path.join(self.out_dir, f"{name}_hist.pkl")
            try:
                self._require_files([cif_path])
                proc = ICSDGraphletProcessor(
                    cif_file=cif_path,
                    bin_centers_clas_2d=self.bin_clas,
                    bin_centers_reg_2d=self.bin_reg,
                    feature_names=self.feature_names,
                    excel_file=self.excel_mapping_xls,
                    sheet_name=self.excel_sheet,
                    atomic_radii_path=self.atomic_radii_json,
                    atomic_features_path=self.atomic_features_json,
                )
                if not getattr(proc, "ok", False):
                    raise proc.error if getattr(proc, "error", None) else RuntimeError(
                        "ICSDGraphletProcessor initialization failed."
                    )
                proc.pickle_histogram(out_pkl)
                print(f"Saved: {out_pkl}")
                results[cif_path] = "success"
            except Exception as e:
                print(f"Failed for {cif_path}: {e}")
                traceback.print_exc(limit=1)
                results[cif_path] = repr(e)
        print("\nAll done.")
        return results

    # ---------- helpers ----------

    @staticmethod
    def _load_bins(bin_clas_path: str, bin_reg_path: str):
        """Load bins and feature names using exact keys."""
        with open(bin_clas_path, "rb") as f:
            clas_obj = pickle.load(f)
        with open(bin_reg_path, "rb") as f:
            reg_obj = pickle.load(f)
        bin_clas = clas_obj["bin_centers_list"]
        bin_reg = reg_obj["bin_centers_list"]
        feature_names = clas_obj["histogram_feature_names"]
        return bin_clas, bin_reg, feature_names

    @staticmethod
    def _require_files(paths: List[str]):
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Missing file(s):\n  " + "\n  ".join(missing))