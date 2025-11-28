#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined module
===============

Unified crystallographic ML pipeline utilities:

- ``SymmetryFeatureExtractor`` — computes averaged symmetry feature vectors
  from CIFs using a point-group → feature mapping in Excel.

- ``ICSDGraphletProcessor`` — builds graphlets and fixed-bin 2D histograms
  (classification & regression) and now **always** attaches the symmetry
  feature vector to the histogram payload.

Notes
-----
- All paths expand ``~`` and environment variables.
- You can override defaults in :class:`Config` via environment variables.
- ``ICSDGraphletProcessor`` depends on ``PYGraphlets``.

Author
------
Aaditya Panigrahi, Yanjun Liu
"""

from __future__ import annotations

__all__ = ["Config", "SymmetryFeatureExtractor", "ICSDGraphletProcessor"]
__version__ = "0.11.7"

# -------------------- standard deps --------------------
import json
import os
import pickle
import warnings
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Structure as PMGStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# -------------------- PYGraphlets deps --------------------
from PYGraphlets import Create_Graphlets, Graphlet_AnalyzerFixedBins2D


# ============================================================
# Config and Utility Classes
# ============================================================
class Config:
    """
    Centralized configuration defaults.

    Environment variable overrides
    ------------------------------
    EXCEL_MAPPING_XLS : str
        Path to Excel mapping (default: "Space_group.xls").
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

            self.histogram = {
                "graphlet": self.graphlet,
                "cif_path": self.cif_file,
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