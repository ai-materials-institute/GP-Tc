#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPTcPrediction Batch Inference App.

Web interface for predicting superconductivity properties from CIF files
using pre-trained Gaussian Process models, with batch inference support
for efficient GPU utilization when handling multiple concurrent users.

Usage:
    streamlit run GPTcPrediction_Batch.py --server.maxUploadSize=1

Author: Aaditya Panigrahi
"""

import os
import sys
import tempfile
import threading
import time
import uuid
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Dict, Any, Optional

import streamlit as st
import numpy as np
import pandas as pd

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if os.path.join(BASE_DIR, "GP_Models") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "GP_Models"))

# Import feature extraction logic
from predict_single_cif import extract_features_from_cif

# Import GP Model components
from GP_Models.GP_Classification_pred import (
    load_trained_class_gp_and_scaler,
    GP_prediction as GP_classification_prediction,
)
from GP_Models.GP_Regression_pred import (
    load_trained_reg_gp_and_scaler,
    GP_prediction as GP_regression_prediction,
)

# Configure Streamlit page
st.set_page_config(
    page_title="GP-Tc Prediction (Batch)",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        min-height: 160px;
    }
    
    .result-card h3 {
        margin: 0 0 1rem 0;
        color: #343a40;
        font-size: 1.2rem;
    }
    
    /* Metric styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #dee2e6;
    }
    
    .metric-container:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 0.95rem;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #343a40;
    }
    
    /* Formula badge */
    .formula-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.3);
    }
    
    /* Batch info badge */
    .batch-badge {
        background: linear-gradient(135deg, #17a2b8 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# BATCH INFERENCE MANAGER
# ============================================================================

@dataclass
class InferenceRequest:
    """A single inference request with features and result placeholder."""
    request_id: str
    clas_histograms: np.ndarray  # Shape: (H, 20, 2)
    reg_histograms: np.ndarray   # Shape: (H, 20, 2)
    symmetry_feature: np.ndarray # Shape: (11,)
    reduced_formula: str
    result: Optional[Dict[str, Any]] = None
    completed: threading.Event = None
    
    def __post_init__(self):
        if self.completed is None:
            self.completed = threading.Event()


class BatchInferenceManager:
    """
    Manages batch inference for multiple concurrent users.
    
    Collects inference requests and processes them in batches for
    efficient GPU utilization.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure one manager across all sessions."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.request_queue: Queue = Queue()
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.pending_lock = threading.Lock()
        
        # Batch configuration
        self.batch_size = 10        # Process when this many requests arrive
        self.timeout_ms = 500       # Or process after this timeout (ms)
        self.min_batch_size = 10    # Minimum batch size for GP models
        
        # Load models once
        self.models_loaded = False
        self.clas_scaler = None
        self.clas_model = None
        self.clas_likelihood = None
        self.reg_scaler = None
        self.reg_model = None
        self.reg_likelihood = None
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        
        print("[BatchManager] Initialized and worker started")
    
    def _load_models_if_needed(self):
        """Load GP models if not already loaded."""
        if not self.models_loaded:
            print("[BatchManager] Loading GP models...")
            self.clas_scaler, self.clas_model, self.clas_likelihood, _, _ = load_trained_class_gp_and_scaler()
            self.reg_scaler, self.reg_model, self.reg_likelihood = load_trained_reg_gp_and_scaler()
            self.models_loaded = True
            print("[BatchManager] Models loaded successfully")
    
    def submit_request(self, features: Dict[str, Any]) -> InferenceRequest:
        """
        Submit a new inference request.
        
        Args:
            features: Dictionary with 'clas_histograms', 'reg_histograms', 
                     'symmetry_feature', 'reduced_formula'
        
        Returns:
            InferenceRequest object with completed event to wait on
        """
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            clas_histograms=features['clas_histograms'],
            reg_histograms=features['reg_histograms'],
            symmetry_feature=features['symmetry_feature'],
            reduced_formula=features['reduced_formula'],
        )
        
        with self.pending_lock:
            self.pending_requests[request.request_id] = request
        
        self.request_queue.put(request.request_id)
        print(f"[BatchManager] Request {request.request_id[:8]} submitted, queue size: {self.request_queue.qsize()}")
        
        return request
    
    def _batch_worker(self):
        """Background thread that processes batched requests."""
        while True:
            batch_request_ids = []
            first_request_time = None
            
            # Collect requests until batch_size or timeout
            while len(batch_request_ids) < self.batch_size:
                try:
                    # Calculate remaining timeout
                    if first_request_time is None:
                        timeout = None  # Wait indefinitely for first request
                    else:
                        elapsed = (time.time() - first_request_time) * 1000
                        remaining = max(0.01, (self.timeout_ms - elapsed) / 1000)
                        timeout = remaining
                    
                    request_id = self.request_queue.get(timeout=timeout)
                    batch_request_ids.append(request_id)
                    
                    if first_request_time is None:
                        first_request_time = time.time()
                        
                except Empty:
                    # Timeout reached, process what we have
                    break
            
            if batch_request_ids:
                self._process_batch(batch_request_ids)
    
    def _process_batch(self, request_ids: list):
        """Process a batch of requests together."""
        print(f"[BatchManager] Processing batch of {len(request_ids)} requests")
        
        # Ensure models are loaded
        self._load_models_if_needed()
        
        # Gather requests
        with self.pending_lock:
            requests = [self.pending_requests.get(rid) for rid in request_ids]
            requests = [r for r in requests if r is not None]
        
        if not requests:
            return
        
        actual_batch_size = len(requests)
        
        # Pad to minimum batch size if needed
        effective_batch_size = max(actual_batch_size, self.min_batch_size)
        
        # Stack features into batches
        clas_hist_batch = np.stack([r.clas_histograms for r in requests], axis=0)
        reg_hist_batch = np.stack([r.reg_histograms for r in requests], axis=0)
        symm_batch = np.stack([r.symmetry_feature for r in requests], axis=0)
        
        # Pad if needed
        if actual_batch_size < effective_batch_size:
            pad_count = effective_batch_size - actual_batch_size
            clas_hist_batch = np.concatenate([
                clas_hist_batch,
                np.tile(clas_hist_batch[0:1], (pad_count, 1, 1, 1))
            ], axis=0)
            reg_hist_batch = np.concatenate([
                reg_hist_batch,
                np.tile(reg_hist_batch[0:1], (pad_count, 1, 1, 1))
            ], axis=0)
            symm_batch = np.concatenate([
                symm_batch,
                np.tile(symm_batch[0:1], (pad_count, 1))
            ], axis=0)
        
        try:
            # Run classification prediction
            clas_mean, clas_std = GP_classification_prediction(
                hist_feats=clas_hist_batch,
                symm_feats=symm_batch,
                scaler=self.clas_scaler,
                model=self.clas_model,
                likelihood=self.clas_likelihood,
            )
            
            # Run regression prediction
            reg_mean, reg_std = GP_regression_prediction(
                hist_feats=reg_hist_batch,
                symm_feats=symm_batch,
                scaler=self.reg_scaler,
                model=self.reg_model,
                likelihood=self.reg_likelihood,
            )
            
            # Distribute results back to requests
            for i, request in enumerate(requests):
                request.result = {
                    'classification_prob': float(clas_mean[i]),
                    'classification_std': float(clas_std[i]),
                    'regression_mean': float(reg_mean[i]),
                    'regression_std': float(reg_std[i]),
                    'reduced_formula': request.reduced_formula,
                    'batch_size': actual_batch_size,
                }
                request.completed.set()
                
            print(f"[BatchManager] Batch completed successfully")
            
        except Exception as e:
            print(f"[BatchManager] Batch processing error: {e}")
            # Signal error to all requests
            for request in requests:
                request.result = {'error': str(e)}
                request.completed.set()
        
        finally:
            # Clean up pending requests
            with self.pending_lock:
                for rid in request_ids:
                    self.pending_requests.pop(rid, None)


# ============================================================================
# STREAMLIT APP
# ============================================================================

@st.cache_resource
def get_batch_manager():
    """Get or create the singleton batch manager."""
    return BatchInferenceManager()


def format_formula_html(formula):
    """Convert numbers in chemical formula to HTML subscripts, hiding subscript 1."""
    import re
    # First remove standalone 1s (e.g., H1O -> HO)
    formula = re.sub(r'(?<=[A-Za-z])1(?=\D|$)', '', formula)
    # Then subscript remaining numbers > 1
    return re.sub(r'(\d+)', r'<sub>\1</sub>', formula)


def run_batch_prediction(cif_path: str, timeout_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Run prediction using the batch manager.
    
    Args:
        cif_path: Path to the CIF file
        timeout_seconds: Maximum time to wait for result
        
    Returns:
        Prediction results dictionary
    """
    # Extract features (CPU work, done per request)
    features = extract_features_from_cif(cif_path)
    
    # Submit to batch manager
    manager = get_batch_manager()
    request = manager.submit_request(features)
    
    # Wait for result
    if request.completed.wait(timeout=timeout_seconds):
        if 'error' in request.result:
            raise Exception(request.result['error'])
        return request.result
    else:
        raise TimeoutError("Prediction timed out waiting for batch processing")


def main():
    # Header with gradient styling
    st.markdown("""
    <div class="main-header">
        <h1>🔮 GP-Tc Prediction</h1>
        <p>AI-powered superconductivity prediction from crystal structures</p>
        <div class="batch-badge">⚡ Batch Processing Enabled</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize batch manager in background
    if 'batch_manager_initialized' not in st.session_state:
        with st.spinner("Initializing batch inference system..."):
            get_batch_manager()
        st.session_state['batch_manager_initialized'] = True
    
    # Sidebar config
    st.sidebar.title("Configuration")
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    
    # Classification model
    clas_models = {
        "2nd Order Graphlet + Symmetry": "Uses 2nd-order graphlet histograms and symmetry features to predict whether a material is a superconductor."
    }
    selected_clas = st.sidebar.selectbox(
        "Classification Model", 
        list(clas_models.keys()),
    )
    
    # Regression model
    reg_models = {
        "4 2nd Order Graphlet (EA, AWM, CM, BL) + Symmetry": "Uses 4 graphlet histograms (Electron Affinity, Atomic Weight Mean, Column Mean, Bond Length) with symmetry features to predict Tc."
    }
    selected_reg = st.sidebar.selectbox(
        "Regression Model", 
        list(reg_models.keys()),
    )
    
    # Dynamic model info expander
    with st.sidebar.expander("ℹ️ Model Info"):
        st.markdown(f"**Classification:** {clas_models[selected_clas]}")
        st.markdown(f"**Regression:** {reg_models[selected_reg]}")
    
    st.sidebar.markdown("---")
    
    # File uploader in sidebar
    st.sidebar.subheader("📁 Upload Structure")
    uploaded_file = st.sidebar.file_uploader("Choose a CIF file", type=['cif'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Check file size (1 MB limit)
        if uploaded_file.size > 1 * 1024 * 1024:
            st.error("File size exceeds 1 MB limit. Please upload a smaller file.")
        else:
            # Save uploaded file to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.cif') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                with st.spinner('Processing structure (batched inference)...'):
                    results = run_batch_prediction(tmp_path)
                    
                # Display Results
                st.markdown(f'<div style="text-align: center;"><div class="formula-badge">📊 {format_formula_html(results["reduced_formula"])}</div></div>', unsafe_allow_html=True)
                
                # Formatting results to 2 decimal places
                clas_prob = results['classification_prob']
                reg_mean = results['regression_mean']
                reg_std = results['regression_std']
                batch_size = results.get('batch_size', 1)
                
                # Create unified results card
                st.markdown("""
                <div class="result-card">
                    <h3>🔮 Prediction Results</h3>
                    <div class="metric-container">
                        <span class="metric-label">Superconductor Probability</span>
                        <span class="metric-value">{:.2f}</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Predicted Tc</span>
                        <span class="metric-value">{:.2f} K</span>
                    </div>
                    <div class="metric-container">
                        <span class="metric-label">Uncertainty (±)</span>
                        <span class="metric-value">{:.2f} K</span>
                    </div>
                </div>
                """.format(clas_prob, reg_mean, reg_std), unsafe_allow_html=True)
                
                # Show batch info
                if batch_size > 1:
                    st.info(f"⚡ Processed in batch of {batch_size} requests")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                import traceback
                st.text(traceback.format_exc())
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


if __name__ == "__main__":
    main()
