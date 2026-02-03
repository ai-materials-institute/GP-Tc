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

# Custom CSS for professional styling with Yellow/Blue/Green scientific palette
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Color palette variables */
    :root {
        --primary-deep-teal: #0B4F6C;
        --primary-teal: #1A7A8C;
        --secondary-emerald: #01A887;
        --secondary-mint: #4ECDC4;
        --accent-gold: #F2B705;
        --accent-amber: #E5A700;
        --light-mint: #E8F4F2;
        --light-ice: #F4FAFA;
        --text-dark: #1A2E35;
        --text-muted: #5A7A82;
    }
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Stremlit background */
    .stApp {
        background: linear-gradient(180deg, var(--light-ice) 0%, var(--light-mint) 100%);
    }
    
    /* Header styling - Glassmorphism effect */
    .main-header {
        background: linear-gradient(135deg, var(--primary-deep-teal) 0%, var(--primary-teal) 50%, var(--secondary-emerald) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(11, 79, 108, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: shimmer 8s ease-in-out infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(25%, 25%); }
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        opacity: 0.95;
        font-size: 1.15rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Result cards - Premium glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.75rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(11, 79, 108, 0.1);
        border-left: 5px solid var(--secondary-emerald);
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 24px rgba(11, 79, 108, 0.08), 0 1px 3px rgba(0,0,0,0.04);
        min-height: 180px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(11, 79, 108, 0.12), 0 2px 6px rgba(0,0,0,0.06);
    }
    
    .result-card h3 {
        margin: 0 0 1.25rem 0;
        color: var(--primary-deep-teal);
        font-size: 1.3rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(11, 79, 108, 0.08);
        transition: background-color 0.2s ease;
    }
    
    .metric-container:hover {
        background-color: rgba(1, 168, 135, 0.04);
        margin: 0 -0.5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
        border-radius: 8px;
    }
    
    .metric-container:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.2px;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-dark);
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value.highlight {
        color: var(--secondary-emerald);
    }
    
    /* Formula badge - Premium with gold accent */
    .formula-badge {
        background: linear-gradient(135deg, var(--secondary-emerald) 0%, var(--secondary-mint) 100%);
        color: white;
        padding: 0.85rem 2rem;
        border-radius: 50px;
        font-size: 1.35rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        margin-bottom: 1.75rem;
        box-shadow: 0 4px 16px rgba(1, 168, 135, 0.35);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .formula-badge:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 24px rgba(1, 168, 135, 0.45);
    }
    
    /* Gold accent badge variant */
    .gold-accent {
        background: linear-gradient(135deg, var(--accent-gold) 0%, var(--accent-amber) 100%);
        box-shadow: 0 4px 16px rgba(242, 183, 5, 0.35);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--light-ice) 0%, #ffffff 100%);
        border-right: 1px solid rgba(11, 79, 108, 0.1);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: var(--primary-deep-teal);
        font-weight: 600;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed var(--secondary-mint);
        transition: border-color 0.3s ease, background 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--secondary-emerald);
        background: rgba(1, 168, 135, 0.05);
    }
    
    /* Select boxes */
    [data-testid="stSelectbox"] > div > div {
        border-color: var(--secondary-mint);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(11, 79, 108, 0.05);
        border-radius: 8px;
        color: var(--primary-deep-teal);
        font-weight: 500;
    }
    
    /* Spinner styling */
    .stSpinner > div > div {
        border-top-color: var(--secondary-emerald) !important;
    }
    
    /* Hide Streamlit footer and top toolbar, keep sidebar menu */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(1, 168, 135, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        border: 1px solid var(--secondary-emerald);
        border-radius: 12px;
        color: var(--primary-deep-teal);
    }
    
    /* Error message styling */
    .stError {
        border-radius: 12px;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, rgba(11, 79, 108, 0.08) 0%, rgba(26, 122, 140, 0.05) 100%);
        border: 1px solid rgba(11, 79, 108, 0.15);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: var(--text-dark);
    }
    
    /* Probability indicator */
    .prob-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .prob-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .prob-high { background: var(--secondary-emerald); }
    .prob-medium { background: var(--accent-gold); }
    .prob-low { background: var(--primary-teal); }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .result-card {
            padding: 1.25rem;
        }
        .metric-value {
            font-size: 1.25rem;
        }
    }
    
    /* Dark mode support - follows system preference */
    @media (prefers-color-scheme: dark) {
        :root {
            --light-mint: #1E2D2F;
            --light-ice: #162022;
            --text-dark: #E8F4F2;
            --text-muted: #9CB5BC;
        }
        
        .stApp {
            background: linear-gradient(180deg, #162022 0%, #1E2D2F 100%) !important;
        }
        
        .result-card {
            background: rgba(45, 65, 70, 0.95) !important;
            border: 2px solid var(--secondary-emerald) !important;
            box-shadow: 0 4px 24px rgba(1, 168, 135, 0.25), 0 0 0 1px rgba(78, 205, 196, 0.1) !important;
        }
        
        .result-card h3 {
            color: var(--secondary-mint) !important;
        }
        
        .metric-label {
            color: var(--text-muted) !important;
        }
        
        .metric-value {
            color: var(--text-dark) !important;
        }
        
        .metric-container {
            border-bottom-color: rgba(78, 205, 196, 0.15) !important;
        }
        
        .metric-container:hover {
            background-color: rgba(1, 168, 135, 0.1) !important;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1E2D2F 0%, #162022 100%) !important;
            border-right-color: rgba(78, 205, 196, 0.2) !important;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: var(--secondary-mint) !important;
        }
        
        [data-testid="stFileUploader"] {
            background: rgba(30, 45, 47, 0.7) !important;
            border-color: var(--primary-teal) !important;
        }
        
        .streamlit-expanderHeader {
            background: rgba(11, 79, 108, 0.2) !important;
            color: var(--secondary-mint) !important;
        }
        
        .info-box {
            background: linear-gradient(135deg, rgba(11, 79, 108, 0.2) 0%, rgba(26, 122, 140, 0.15) 100%) !important;
            border-color: rgba(78, 205, 196, 0.3) !important;
            color: var(--text-dark) !important;
        }
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
