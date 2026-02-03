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

# Custom CSS for Antigravity Design - Functional gradients from research schematic
st.markdown("""
<style>
    /* Import premium fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* GLOBAL: Antigravity Schematic Palette */
    :root {
        /* Panel A: Data Fusion / Curation - Lavender-Blue */
        --bg-curation: #E6EBFF;
        --bg-curation-end: #C8D1E6;
        /* Panel B: Featurization */
        --bg-featurization: #F5F5F5;
        --bg-featurization-end: #EBEBEB;
        /* Panel C: Insight */
        --bg-insight: #FFFFFF;
        /* Text: Cobalt Blue */
        --text-main: #284678;
        --text-muted: #64748B;
        /* Ground Truth Red gradient */
        --truth-red: #BE1E2D;
        --truth-red-end: #901622;
        /* Order-based gradients for graphlet complexity */
        --order1: #F5E18C;
        --order1-end: #E6C962;
        --order2: #96C3A0;
        --order2-end: #76A581;
        --order3: #64AFAF;
        --order3-end: #4D8E8E;
        /* Borders and shadows */
        --border-light: #D1D9E6;
        --shadow-soft: rgba(40, 70, 120, 0.08);
        --shadow-medium: rgba(40, 70, 120, 0.12);
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
    
    /* App background */
    .stApp {
        background: linear-gradient(180deg, #FAFBFF 0%, var(--bg-insight) 100%);
    }
    
    /* Header: Panel A - Teal Gradient (Translucent) */
    .main-header {
        background: linear-gradient(90deg, rgba(0, 167, 157, 0.65) 0%, rgba(20, 190, 180, 0.65) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #FFFFFF;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        border-bottom: 2px solid rgba(0, 95, 89, 0.5);
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 167, 157, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    
    .main-header p {
        margin: 0.75rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.9);
        opacity: 1;
    }
    
    /* Result cards: Top border with teal gradient */
    .result-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        border: 1px solid var(--border-light);
        border-top: 5px solid var(--order3);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        min-height: 160px;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .result-card h3 {
        margin: 0 0 1rem 0;
        color: var(--text-main);
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Streamlit native metrics: Top border styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 12px;
        padding: 20px;
        border-top: 5px solid var(--order3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 0;
        border-bottom: 1px solid var(--border-light);
    }
    
    .metric-container:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-main);
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Ground Truth / True Tc: Red gradient text */
    .truth-val {
        background: linear-gradient(90deg, var(--truth-red) 0%, var(--truth-red-end) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    /* Formula badge: 2nd to 3rd order gradient (Green to Teal) */
    .formula-badge {
        background: linear-gradient(135deg, var(--order2) 0%, var(--order3) 100%);
        color: var(--text-main);
        padding: 0.85rem 2rem;
        border-radius: 50px;
        font-size: 1.25rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(100, 175, 175, 0.3);
    }
    
    /* Order-based card gradient variants */
    .order1-card {
        border-image: linear-gradient(to bottom, var(--order1), var(--order1-end)) 1;
    }
    .order2-card {
        border-image: linear-gradient(to bottom, var(--order2), var(--order2-end)) 1;
    }
    .order3-card {
        border-image: linear-gradient(to bottom, var(--order3), var(--order3-end)) 1;
    }
    
    /* Sidebar: Featurization Grey */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-featurization) 0%, var(--bg-featurization-end) 100%);
        border-right: 1px solid var(--border-light);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: var(--text-main);
        font-weight: 600;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: var(--bg-insight);
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed var(--order3);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--text-main);
        background: rgba(230, 235, 255, 0.3);
    }
    
    /* Select boxes */
    [data-testid="stSelectbox"] > div > div {
        border-color: var(--border-light);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-featurization);
        border-radius: 8px;
        color: var(--text-main);
        font-weight: 500;
    }
    
    /* Spinner styling */
    .stSpinner > div > div {
        border-top-color: var(--order3) !important;
    }
    
    /* Hide Streamlit footer and make header transparent */
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, rgba(150, 195, 160, 0.2) 0%, rgba(150, 195, 160, 0.1) 100%);
        border: 1px solid var(--order2);
        border-radius: 12px;
        color: var(--text-main);
    }
    
    /* Error message styling */
    .stError {
        border-radius: 12px;
        border-left: 4px solid var(--truth-red);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header {
            padding: 1.5rem;
        }
        .result-card {
            padding: 1.25rem;
        }
        .metric-value {
            font-size: 1.2rem;
        }
    }
    
    /* DARK MODE: Antigravity Dark Gradients */
    @media (prefers-color-scheme: dark) {
        :root {
            /* Dark mode gradients per Gemini spec */
            --bg-curation: #1E2442;
            --bg-curation-end: #2D3664;
            --bg-featurization: #1E293B;
            --bg-featurization-end: #162032;
            --bg-insight: #0F172A;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --truth-red: #EF4444;
            --truth-red-end: #DC2626;
            /* Dark mode order gradients */
            --order1: #B4A048;
            --order1-end: #8E7D32;
            --order2: #3E6B48;
            --order2-end: #2D4F35;
            --order3: #246B6B;
            --order3-end: #1A4F4F;
            /* For bright accents in dark mode */
            --order3-bright: #2DD4BF;
            --order3-bright-end: #14B8A6;
            --border-light: #334155;
            --shadow-soft: rgba(0, 0, 0, 0.2);
            --shadow-medium: rgba(0, 0, 0, 0.3);
        }
        
        .stApp {
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        }
        
        .main-header {
            background: linear-gradient(90deg, var(--bg-curation) 0%, var(--bg-curation-end) 100%) !important;
            border-bottom-color: var(--order3-bright) !important;
        }
        
        .result-card {
            background: linear-gradient(145deg, #0F172A, #1E293B) !important;
            border: 1px solid var(--border-light) !important;
            border-top: 5px solid var(--order3-bright) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }
        
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #0F172A, #1E293B) !important;
            border-top: 5px solid var(--order3-bright) !important;
            border-radius: 12px !important;
        }
        
        .formula-badge {
            background: linear-gradient(135deg, #4ADE80 0%, #2DD4BF 100%) !important;
            color: #0F172A !important;
            box-shadow: 0 4px 16px rgba(45, 212, 191, 0.3) !important;
        }
        
        .truth-val {
            background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        
        /* Spinner visible in dark mode - plain and clean */
        .stSpinner > div > div,
        .stSpinner > div > div > div,
        [data-testid="stSpinner"] > div,
        [data-testid="stSpinner"] > div > div {
            border-top-color: #FFFFFF !important;
            border-right-color: rgba(255, 255, 255, 0.3) !important;
            border-bottom-color: rgba(255, 255, 255, 0.3) !important;
            border-left-color: rgba(255, 255, 255, 0.3) !important;
        }
        
        .stSpinner > div,
        .stSpinner,
        [data-testid="stSpinner"] {
            color: var(--text-main) !important;
        }
        
        .stSpinner p,
        [data-testid="stSpinner"] p {
            color: var(--text-main) !important;
        }
        
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-featurization) 0%, var(--bg-featurization-end) 100%) !important;
            border-right-color: var(--border-light) !important;
        }
        
        [data-testid="stFileUploader"] {
            background: var(--bg-featurization) !important;
            border-color: var(--order3-bright) !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            background: rgba(45, 212, 191, 0.1) !important;
        }
        
        /* File uploader inner dropzone */
        [data-testid="stFileUploader"] > div,
        [data-testid="stFileUploader"] > div > div,
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] section > div,
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploadDropzone"] > div {
            background: var(--bg-featurization) !important;
            background-color: var(--bg-featurization) !important;
        }
        
        /* File uploader text */
        [data-testid="stFileUploader"] *,
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] div {
            color: var(--text-main) !important;
        }
        
        [data-testid="stFileUploader"] button {
            background-color: var(--bg-curation) !important;
            color: var(--text-main) !important;
            border-color: var(--order3-bright) !important;
        }
        
        .streamlit-expanderHeader {
            background: linear-gradient(90deg, var(--bg-curation) 0%, var(--bg-featurization) 100%) !important;
            color: var(--text-main) !important;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.15) 0%, rgba(74, 222, 128, 0.05) 100%) !important;
            border-color: #4ADE80 !important;
        }
        
        /* Text color fixes */
        .main .block-container,
        .main .block-container p,
        .main .block-container span,
        .main .block-container label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stText"] {
            color: var(--text-main) !important;
        }
        
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: var(--text-main) !important;
        }
        
        .result-card h3,
        .result-card .metric-label {
            color: var(--text-muted) !important;
        }
        
        .result-card .metric-value {
            color: var(--text-main) !important;
        }
        
        [data-testid="stSelectbox"] label,
        [data-testid="stFileUploader"] label {
            color: var(--text-main) !important;
        }
        
        [data-testid="stSelectbox"] > div > div {
            background-color: var(--bg-featurization) !important;
            color: var(--text-main) !important;
        }
        
        hr {
            border-color: var(--border-light) !important;
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
