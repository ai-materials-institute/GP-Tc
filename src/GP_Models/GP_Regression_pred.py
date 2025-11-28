"""
GP Regression Prediction Module.

This module provides functionality for performing Gaussian Process (GP) regression
predictions using pre-trained models. It includes utilities for loading trained
models and scalers, processing input data (histogram and symmetry features),
and generating predictions in batches.

The module is designed to work with specific data structures involving
histogram features and symmetry features, typically used in materials science
applications (e.g., predicting critical temperatures).

Author: Aaditya Panigrahi, Natalie Maus, Yanjun Liu
"""

import sys, os, re, time, pickle, warnings
import numpy as np
import torch
import gpytorch
import gc
from sklearn.model_selection import train_test_split

# Silencing warnings / WANDB noise
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "True"

# Project imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "sc_train_gp-main"))
from utils.Histo_array_scaler import Histo_Array_Scaler
from models.new_sg_featurization_sg_emd_exact_gp import SgEmdExactGPModelV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- hyperparams actually used here
n_batches_emd_kernel = 10
Hist_Keep_Idxs = [13, 18, 26, 30]  # matches your original indices

def load_trained_reg_gp_and_scaler(Training_Data_Path = os.path.join(BASE_DIR, "sc_train_gp-main/data/regression_data_histogram&symmetry.pkl"),
                                   Histogram_Feature_Key = "histogram_features",
                                   Target_Key = "Tc",
                                   Symmetry_Key = "symmetry_features",
                                   Test_Size = 0.2,
                                   Random_State = 2,
                                   Model_Path = os.path.join(BASE_DIR, "Trained Models/Regressor_4-2odr_all-sym"),
                                   Hist_Feature_Keep_Idxs = Hist_Keep_Idxs,
                                   ):
    """
    Load a trained Gaussian Process regression model and its associated scaler.

    Parameters
    ----------
    Training_Data_Path : str, optional
        Path to the pickle file containing training data.
    Histogram_Feature_Key : str, optional
        Key for histogram features in the data dictionary.
    Target_Key : str, optional
        Key for the target variable in the data dictionary.
    Symmetry_Key : str, optional
        Key for symmetry features in the data dictionary.
    Test_Size : float, optional
        Proportion of the dataset to include in the test split.
    Random_State : int, optional
        Random seed for reproducibility.
    Model_Path : str, optional
        Directory containing the saved model state (`model_state.pt`) and
        likelihood state (`likelihood_state.pt`).
    Hist_Feature_Keep_Idxs : list of int, optional
        Indices of histogram features to keep.

    Returns
    -------
    scaler : Histo_Array_Scaler
        Fitted scaler for histogram features.
    model : SgEmdExactGPModelV2
        Loaded GP regression model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Loaded likelihood model.
    """
    # ---- load data
    with open(Training_Data_Path, "rb") as f:
        loaded = pickle.load(f)

    X_all  = np.array(loaded[Histogram_Feature_Key])  # (N, H, 20, 2)
    y_all  = np.array(loaded[Target_Key])             # (N,)
    S_all  = np.array(loaded[Symmetry_Key])           # (N, 11)

    # ---- split
    X_train_np, X_test_np, y_train_np, y_test_np, S_train_np, S_test_np = train_test_split(
        X_all, y_all, S_all, test_size=Test_Size, random_state=Random_State
    )

    # ---- fit scaler on TRAIN histograms and transform
    scaler = Histo_Array_Scaler()
    X_train_scaled = scaler.fit_transform(X_train_np[:,Hist_Feature_Keep_Idxs,:,:])
    X_test_scaled  = scaler.transform(X_test_np[:,Hist_Feature_Keep_Idxs,:,:])

    # ---- to torch
    # FIX: make arrays contiguous & writable before torch.from_numpy
    X_train = torch.from_numpy(np.ascontiguousarray(X_train_scaled).copy()).float().to(device)   # (n, H, 20, 2)
    y_train = torch.from_numpy(np.ascontiguousarray(y_train_np).copy()).float().to(device)       # (n,)
    S_train = torch.from_numpy(np.ascontiguousarray(S_train_np).copy()).float().to(device)       # (n, 11)

    # ---- shapes for model
    data_shape  = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])  # (h_keep, 20, 2)
    n_histogram = X_train.shape[-3]
    n_sg        = S_train.shape[-1]

    # ---- flatten histograms and prepend symmetry features
    X_train_flat = X_train.reshape(X_train.size(0), -1)
    X_train_cat  = torch.cat([S_train, X_train_flat], dim=-1)       # (n, n_sg + h_keep*20*2)

    # ---- init model/likelihood and load states
    model = SgEmdExactGPModelV2(
        train_x=X_train_cat,
        train_y=y_train,
        likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device),
        n_sg=n_sg,
        map_saas_tau=None,
        n_histogram=n_histogram,
        data_shape=data_shape,
        n_batches_emd_kernel=n_batches_emd_kernel,
        add_saas_ls_prior=False,
    ).to(device)

    likelihood = model.likelihood  # already created above

    PATH_TO_MODEL = os.path.join(Model_Path, "model_state.pt")
    PATH_TO_LIK   = os.path.join(Model_Path, "likelihood_state.pt")
    model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
    likelihood.load_state_dict(torch.load(PATH_TO_LIK, map_location=device))

    model.eval()
    likelihood.eval()

    # return what GP_prediction needs
    return scaler, model, likelihood


# ---- standardized GP_prediction signature (like your classification script) ----
def GP_prediction(hist_feats, symm_feats, scaler, model, likelihood, keep_idx=None):
    """
    Perform GP regression prediction on new data.

    Parameters
    ----------
    hist_feats : np.ndarray
        Histogram features of shape (B, H, 20, 2).
    symm_feats : np.ndarray
        Symmetry features of shape (B, 11).
    scaler : Histo_Array_Scaler
        Fitted scaler used to transform histogram features.
    model : SgEmdExactGPModelV2
        Trained GP model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Trained likelihood.
    keep_idx : torch.Tensor or list, optional
        Indices to keep along the H dimension.

    Returns
    -------
    GP_mean : np.ndarray
        Mean predictions from the GP posterior.
    GP_std : np.ndarray
        Standard deviation of predictions from the GP posterior.
    """
    model.eval()
    likelihood.eval()

    # scale histograms
    # FIX: ensure contiguous & writable before torch.from_numpy
    X_rescaled = scaler.transform(hist_feats)                  # (B, h_keep, 20, 2)
    X_rescaled = torch.from_numpy(np.ascontiguousarray(X_rescaled).copy()).float().to(device)
    S = torch.from_numpy(np.ascontiguousarray(symm_feats).copy()).float().to(device)        # (B, 11)

    # flatten + concat [sg | flat_hist]
    X_flat = X_rescaled.reshape(X_rescaled.size(0), -1)
    X_cat  = torch.cat([S, X_flat], dim=-1)

    with torch.no_grad():
        GP_out = likelihood(model(X_cat))
        GP_mean = GP_out.mean.detach().cpu().numpy()
        GP_std  = GP_out.stddev.detach().cpu().numpy()

    return GP_mean, GP_std


# ---------- optional utilities (kept for parity with your classification script) ----------
def extract_icsd_id(filename: str) -> str:
    match = re.search(r"icsd_(\d+)_histogram\.pkl", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract ICSD ID from '{filename}'")

def merge_dicts(dict1, dict2):
    """Merge two dictionaries whose values are lists."""
    return {k: dict1.get(k, []) + dict2.get(k, []) for k in set(dict1) | set(dict2)}

def chunk_list(lst, chunk_size=2000):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
def GP_Batch_Predictor(files_chunk, folder_path, scaler, model, likelihood, keep_idx):
    """
    Predict outcomes for a batch of files using the GP model.

    Parameters
    ----------
    files_chunk : list of str
        List of filenames to process in this batch.
    folder_path : str
        Directory containing the files.
    scaler : Histo_Array_Scaler
        Fitted scaler.
    model : SgEmdExactGPModelV2
        Trained GP model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Trained likelihood.
    keep_idx : list of int
        Indices of histogram features to use.

    Returns
    -------
    gp_batch_dict : dict
        Dictionary containing:
        - 'icsd_id': List of ICSD IDs.
        - 'reduced_formula': List of chemical formulas.
        - 'reg_pred': List of predicted means.
        - 'reg_std': List of predicted standard deviations.
    """
    gp_input_dict = {
        'icsd_id' : [],
        'reduced_formula' : [],
        'reg_histograms' : [],
        'symmetry_feature' : [],
    }
    for file in files_chunk:
        with open(os.path.join(folder_path, file), 'rb') as f:
            data = pickle.load(f)
        gp_input_dict['reg_histograms'].append(data['reg_histograms'][0,keep_idx,:,:].tolist())
        gp_input_dict['symmetry_feature'].append(data['symmetry_feature'].tolist())
        gp_input_dict['icsd_id'].append(extract_icsd_id(file))
        gp_input_dict['reduced_formula'].append(data['reduced_formula'])
    preds_mean, preds_std = GP_prediction(np.array(gp_input_dict['reg_histograms']), 
                                          np.array(gp_input_dict['symmetry_feature']),
                                          scaler,
                                          model,
                                          likelihood,
                                          keep_idx = keep_idx
                                        )

    gp_batch_dict = {}
    gp_batch_dict['icsd_id'] = gp_input_dict['icsd_id']
    gp_batch_dict['reduced_formula'] = gp_input_dict['reduced_formula']
    gp_batch_dict['reg_pred'] = preds_mean.tolist()
    gp_batch_dict['reg_std'] = preds_std.tolist()
    return gp_batch_dict


# ---- minimal example usage
if __name__ == "__main__":
    scaler, model, likelihood = load_trained_reg_gp_and_scaler()
    folder_path = os.path.join(BASE_DIR, "../ICSD_Features/Histogram_Features")
    if not os.path.exists(folder_path):
        # Fallback or warning if the relative path doesn't exist (e.g. if running in a different env)
        print(f"Warning: Data folder not found at {folder_path}")
    
    # Check if folder exists before listing
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
    else:
        files = []
        
    files_chunks = chunk_list(files)
    num_chunk = len(files_chunks)
    gp_pred_dict = {}

    for i in range(num_chunk): 
        print(f"\n Batch: {i}/{num_chunk}")
        gp_batch_dict = GP_Batch_Predictor(
            files_chunk = files_chunks[i],
            folder_path = folder_path,
            scaler = scaler, 
            model = model, 
            likelihood = likelihood,
            keep_idx = Hist_Keep_Idxs
        )
        gp_pred_dict = merge_dicts(gp_pred_dict, gp_batch_dict)

        # ---- free up memory between batches ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # wait for all kernels to finish
            torch.cuda.empty_cache()   # release unused GPU memory
        gc.collect()                   # force Python garbage collection

    with open('../gp_regression_preds.pkl','wb') as f:
        pickle.dump(gp_pred_dict,f)