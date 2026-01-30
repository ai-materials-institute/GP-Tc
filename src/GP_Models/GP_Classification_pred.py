"""
GP Classification Prediction Module.

This module provides functionality for performing Gaussian Process (GP) classification
predictions using pre-trained models. It includes utilities for loading trained
models and scalers, processing input data (histogram and symmetry features),
and generating predictions (probabilities and uncertainties) in batches.

The module is designed to work with specific data structures involving
histogram features and symmetry features, typically used in materials science
applications (e.g., classifying superconductivity).

Author: Aaditya Panigrahi, Natalie Maus, Yanjun Liu
"""

import sys, pickle
import numpy as np
import os
import re
import torch
import gpytorch
import warnings
import time
from sklearn.model_selection import train_test_split

# Silencing the warning 
warnings.filterwarnings("ignore")
gpytorch.settings.fast_pred_var.off()  # avoid GPyTorch noisy "fast_pred_var" warnings if any

# project imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "sc_train_gp-main"))
from models.class_emd_gp_w_sg import EmdSgGpClassificationModel
from utils.Histo_array_scaler import Histo_Array_Scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- hyperparams actually used here
n_batches_emd_kernel = 10
n_ind_pts = 1024

def load_trained_class_gp_and_scaler(Training_Data_Path = os.path.join(BASE_DIR, "sc_train_gp-main/data/classification_data_3DSCnonsc_labeled.pkl"),
                                     Histogram_Feature_Key = "X_all",
                                     Label_Key = "label",
                                     Symmetry_Key = "symm_features",
                                     Test_Size = 0.2,
                                     Random_State = 2,
                                     Model_Path = os.path.join(BASE_DIR, "Trained Models/Classifier_2odr_all-sym"),
                                    ):
    """
    Load a trained Gaussian Process classification model and its associated scaler.

    Parameters
    ----------
    Training_Data_Path : str, optional
        Path to the pickle file containing training data.
    Histogram_Feature_Key : str, optional
        Key for histogram features in the data dictionary.
    Label_Key : str, optional
        Key for the label variable in the data dictionary.
    Symmetry_Key : str, optional
        Key for symmetry features in the data dictionary.
    Test_Size : float, optional
        Proportion of the dataset to include in the test split.
    Random_State : int, optional
        Random seed for reproducibility.
    Model_Path : str, optional
        Directory containing the saved model state (`model_state.pt`) and
        likelihood state (`likelihood_state.pt`).

    Returns
    -------
    scaler : Histo_Array_Scaler
        Fitted scaler for histogram features.
    model : EmdSgGpClassificationModel
        Loaded GP classification model.
    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Loaded likelihood model.
    X_train_np : np.ndarray
        Training data (histogram features) as numpy array.
    S_train_np : np.ndarray
        Training data (symmetry features) as numpy array.
    """
    # ---- load data
    with open(Training_Data_Path, "rb") as f:
        loaded = pickle.load(f)

    X_all  = np.array(loaded[Histogram_Feature_Key])         # (N, 21, 20, 2)
    y_all  = np.array(loaded[Label_Key])         # (N,)
    S_all  = np.array(loaded[Symmetry_Key]) # (N, 11)

    # keep last 2 as "special" and split on the rest
    X_special, y_special, S_special = X_all[-2:], y_all[-2:], S_all[-2:]
    X_base, y_base, S_base = X_all[:-2], y_all[:-2], S_all[:-2]

    X_train_np, X_test_np, y_train_np, y_test_np, S_train_np, S_test_np = train_test_split(
        X_base, y_base, S_base, test_size=0.2, random_state=2
    )

    # ---- fit scaler on TRAIN histograms and transform both
    scaler = Histo_Array_Scaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_test_scaled  = scaler.transform(X_test_np)

    # ---- to torch tensors
    X_train = torch.from_numpy(X_train_scaled).float().to(device)    # (n, 21, 20, 2)
    X_test  = torch.from_numpy(X_test_scaled).float().to(device)     # (m, 21, 20, 2)
    y_train = torch.from_numpy(y_train_np).float().to(device)        # (n,)
    y_test  = torch.from_numpy(y_test_np).float().to(device)         # (m,)
    S_train = torch.from_numpy(S_train_np).float().to(device)        # (n, 11)
    S_test  = torch.from_numpy(S_test_np).float().to(device)         # (m, 11)

    # ---- shapes for model
    n_sg = S_train.shape[-1]                   # 11
    n_histogram = X_train.shape[-3]            # 21
    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])  # (21,20,2)

    # flatten histograms and prepend symmetry features
    def _concat_symm_hist(X_hist: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return torch.cat([S, X_hist.reshape(X_hist.size(0), -1)], dim=-1)

    X_train_cat = _concat_symm_hist(X_train, S_train)  # (n, 11 + 21*20*2)
    X_test_cat  = _concat_symm_hist(X_test,  S_test)

    # ---- init model/likelihood and load states
    model = EmdSgGpClassificationModel(
        train_x=X_train_cat[:n_ind_pts],
        n_histogram=n_histogram,
        n_sg=n_sg,
        data_shape=data_shape,
        n_batches_emd_kernel=n_batches_emd_kernel,
    ).to(device)

    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)

    PATH_TO_MODEL = f"{Model_Path}/model_state.pt"
    PATH_TO_LIK   = f"{Model_Path}/likelihood_state.pt"
    model.load_state_dict(torch.load(PATH_TO_MODEL, map_location=device))
    likelihood.load_state_dict(torch.load(PATH_TO_LIK, map_location=device))

    # return what GP_prediction needs, plus train arrays for the example call
    return scaler, model, likelihood, X_train_np, S_train_np


# ---- your GP_prediction
def GP_prediction(hist_feats, symm_feats, scaler, model, likelihood):
    """
    Perform GP classification prediction on new data.

    Parameters
    ----------
    hist_feats : np.ndarray
        Histogram features of shape (B, H, 20, 2).
    symm_feats : np.ndarray
        Symmetry features of shape (B, 11).
    scaler : Histo_Array_Scaler
        Fitted scaler used to transform histogram features.
    model : EmdSgGpClassificationModel
        Trained GP model.
    likelihood : gpytorch.likelihoods.BernoulliLikelihood
        Trained likelihood.

    Returns
    -------
    GP_mean : np.ndarray
        Mean predictions (probabilities) from the GP posterior.
    GP_stddev : np.ndarray
        Standard deviation of predictions from the GP posterior.
    """
    model.eval()
    likelihood.eval()

    X_rescaled = scaler.transform(hist_feats)
    X_rescaled = torch.from_numpy(X_rescaled).float().to(device=device)
    symm_feats = torch.from_numpy(symm_feats).float().to(device=device)

    n_sg = symm_feats.shape[-1]                      # 11
    n_histogram = X_rescaled.shape[-3]               # 21
    data_shape = (X_rescaled.shape[-3], X_rescaled.shape[-2], X_rescaled.shape[-1])  # (21,20,2)

    X_rescaled = X_rescaled.reshape(X_rescaled.size(0), -1)
    X_rescaled = torch.cat((symm_feats, X_rescaled), -1)  # [sg | flat_hist]

    print(X_rescaled.shape)
    with torch.no_grad():
        GP_out = likelihood(model(X_rescaled))
        GP_stddev = GP_out.stddev.cpu()
        GP_mean = GP_out.mean.cpu()

    return GP_mean.numpy(), GP_stddev.numpy()


def extract_icsd_id(filename: str) -> str:
    match = re.search(r"icsd_(\d+)_histogram\.pkl", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract ICSD ID from '{filename}'")

def merge_dicts(dict1, dict2):
    """
    Merge two dictionaries whose values are lists.
    If a key exists in both, their lists are concatenated.
    """
    return {k: dict1.get(k, []) + dict2.get(k, []) for k in set(dict1) | set(dict2)}

def chunk_list(lst, chunk_size=5000):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def GP_Batch_Predictor(files_chunk):
    """
    Predict outcomes for a batch of files using the GP classification model.

    Parameters
    ----------
    files_chunk : list of str
        List of filenames to process in this batch.

    Returns
    -------
    gp_batch_dict : dict
        Dictionary containing:
        - 'icsd_id': List of ICSD IDs.
        - 'reduced_formula': List of chemical formulas.
        - 'clas_pred': List of predicted probabilities.
        - 'clas_std': List of predicted standard deviations.
    """
    gp_input_dict={
            'icsd_id' : [],
            'reduced_formula' : [],
            'clas_histograms' : [],
            'symmetry_feature' : [],     
            }
    for file in files_chunk:
        with open(f"{folder_path}/{file}",'rb') as f:
            data = pickle.load(f)
        gp_input_dict['icsd_id'].append(extract_icsd_id(file))
        gp_input_dict['reduced_formula'].append(data['reduced_formula'])
        gp_input_dict['clas_histograms'].append(data['clas_histograms'][0,10:31,:,:].tolist())
        gp_input_dict['symmetry_feature'].append(data['symmetry_feature'].tolist())
    GP_preds, GP_stddev = GP_prediction(np.array(gp_input_dict['clas_histograms']), np.array(gp_input_dict['symmetry_feature']), scaler, model, likelihood)
    gp_batch_dict = {}
    gp_batch_dict['icsd_id'] = gp_input_dict['icsd_id']
    gp_batch_dict['reduced_formula'] = gp_input_dict['reduced_formula']
    gp_batch_dict['clas_pred'] = GP_preds.tolist()
    gp_batch_dict['clas_std'] = GP_stddev.tolist()
    return gp_batch_dict

    
# ---- minimal example usage
if __name__ == "__main__":
    scaler, model, likelihood, X_train_np, S_train_np = load_trained_class_gp_and_scaler()
    folder_path = os.path.join(BASE_DIR, "../ICSD_Features/Histogram_Features")
    
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
    else:
        print(f"Warning: Data folder not found at {folder_path}")
        files = []
        
    files_chunks = chunk_list(files)
    num_chunk = len(files_chunks)
    gp_pred_dict = {}
    start_time = time.time()
    for i in range(num_chunk): 
        print(f"\n Batch: {i}/{num_chunk} \n")
        gp_batch_dict = GP_Batch_Predictor(files_chunks[i])
        gp_pred_dict = merge_dicts(gp_pred_dict, gp_batch_dict)
    print(time.time()-start_time)
    with open('../gp_classification_preds.pkl','wb') as f:
        pickle.dump(gp_pred_dict,f)