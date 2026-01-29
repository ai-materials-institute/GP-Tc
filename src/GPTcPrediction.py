#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPTcPrediction Streamlit App.

Web interface for predicting superconductivity properties from CIF files
using pre-trained Gaussian Process models.

Optimized with caching to prevent server overload.

Usage:
    streamlit run GPTcPrediction.py

Author: Aaditya Panigrahi
"""

import os
import sys
import tempfile
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
# Note: we import extract_features_from_cif instead of predict_single_cif
# to allow us to manage model loading separately with caching
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
    page_title="GP-Tc Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set max upload size to 1 MB (must be done via config file or CLI)
# Note: Run with: streamlit run GPTcPrediction.py --server.maxUploadSize=1

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

# --- Caching Mechanism to Prevent Server Overload ---
@st.cache_resource
def load_models():
    """
    Load GP models and scalers. Cached by Streamlit to run only once.
    """
    clas_scaler, clas_model, clas_likelihood, _, _ = load_trained_class_gp_and_scaler()
    reg_scaler, reg_model, reg_likelihood = load_trained_reg_gp_and_scaler()
    return (clas_scaler, clas_model, clas_likelihood), (reg_scaler, reg_model, reg_likelihood)

def run_prediction_with_cached_models(cif_path, min_batch_size=10):
    """
    Run prediction using cached models.
    """
    # Load cached models
    (clas_scaler, clas_model, clas_likelihood), (reg_scaler, reg_model, reg_likelihood) = load_models()
    
    # Extract features (CPU intensive but fast enough)
    features = extract_features_from_cif(cif_path)
    
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
    
    return {
        'classification_prob': float(clas_mean[0]),
        'classification_std': float(clas_std[0]),
        'regression_mean': float(reg_mean[0]),
        'regression_std': float(reg_std[0]),
        'reduced_formula': features['reduced_formula'],
    }

def format_formula_html(formula):
    """Convert numbers in chemical formula to HTML subscripts, hiding subscript 1."""
    import re
    # First remove standalone 1s (e.g., H1O -> HO)
    formula = re.sub(r'(?<=[A-Za-z])1(?=\D|$)', '', formula)
    # Then subscript remaining numbers > 1
    return re.sub(r'(\d+)', r'<sub>\1</sub>', formula)

def main():
    # Header with gradient styling
    st.markdown("""
    <div class="main-header">
        <h1>🔮 GP-Tc Prediction</h1>
        <p>AI-powered superconductivity prediction from crystal structures</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Trigger model loading in background if not loaded
    if 'models_loaded' not in st.session_state:
        with st.spinner("Initializing models..."):
            load_models()
        st.session_state['models_loaded'] = True
    
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
                with st.spinner('Processing structure and running GP models...'):
                    results = run_prediction_with_cached_models(tmp_path)
                # Display Results
                st.markdown(f'<div style="text-align: center;"><div class="formula-badge">📊 {format_formula_html(results["reduced_formula"])}</div></div>', unsafe_allow_html=True)
                
                # Formatting results to 2 decimal places
                clas_prob = results['classification_prob']
                reg_mean = results['regression_mean']
                reg_std = results['regression_std']
                
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
