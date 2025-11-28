import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_class_data_07_23_2025(
    dataset_name, 
    device, 
    only_test_on_special=False,
    list_of_features_to_remove=[], # single list of feature idxs to remove, if idx > n_hist assume they are sg features!! 
):
    # only_test_on_special: only test on 'special' final two inputs: MgB2 and graphite
    #   otherwise test on special final two inputs plus significant additional test set 
    assert dataset_name == "classification_data_metal_3DSCnonsc_symm_20250723"
    with open(f"../data_class_fixed_v2_07_23_2025/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    # dict_keys(['X_all', 'label', 'cif', 'formula', 'class', 'source', 'feat_name', 'symm_features', 'symm_feature_name'])
    #   'cif': paths to cif files, not needed 
    #   'formula': string formulas, not needed 
    #   'class': string class names, not needed 
    #   'feat_name': string names of features, not needed 
    #   'symm_feature_name': string names of symm features, not needed 
    X_all = loaded_data["X_all"] # np array (8411, 21, 20, 2) --> 21 hist features 
    labels = np.array(loaded_data["label"])  # np array (8411,)  dtype('int64') (1's and 0's)
    symm_features_11d = np.array(loaded_data['symm_features']) # ((8411, 11), dtype('float64')) symmetry features 

    # num_non_sc = (labels == 0).sum() # 4204
    # num_sc = (labels == 1).sum() # 4207

    # XXX 
    special_test_x = X_all[-2:] # (2, 21, 20, 2)
    special_test_y = labels[-2:] # (2,)  # array([1, 0])
    special_test_symm = symm_features_11d[-2:] # (2, 11)
    
    x_without_special = X_all[0:-2] # (8409, 21, 20, 2)
    y_without_special = labels[0:-2] # (8409,)
    symm_without_special = symm_features_11d[0:-2] # (8409, 11)

    if only_test_on_special:
        X_train = x_without_special
        y_train = y_without_special
        symm_train = symm_without_special
        X_test = special_test_x
        y_test = special_test_y
        symm_test = special_test_symm
    else:
        X_train, X_test, y_train, y_test, symm_train, symm_test = train_test_split(x_without_special, y_without_special, symm_without_special, test_size=0.2, random_state=2)
        # X_train.shape, X_test.shape, y_train.shape, y_test.shape, symm_train.shape, symm_test.shape 
        # ((6727, 21, 20, 2), (1682, 21, 20, 2), (6727,), (1682,), (6727, 11), (1682, 11))
        X_test = np.concatenate((X_test, special_test_x)) # (1684, 21, 20, 2)
        y_test = np.concatenate((y_test, special_test_y)) # (1684,)
        symm_test = np.concatenate((symm_test, special_test_symm)) # (1684, 11)

    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device) # torch.Size([6727, 21, 20, 2])
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device) # torch.Size([1684, 21, 20, 2])
    y_train = torch.from_numpy(y_train).float().to(device=device) # torch.Size([6727])
    y_test = torch.from_numpy(y_test).float().to(device=device) # torch.Size([1684])
    symm_train = torch.from_numpy(symm_train).float().to(device=device) # torch.Size([6727, 11])
    symm_test = torch.from_numpy(symm_test).float().to(device=device) # torch.Size([1684, 11])

    # In list_of_features_to_remove: 
    # indexes 0-20 = 21 hist features 
    # indexes 21-32 = 11 sg features 
    n_hist_full = X_train.shape[-3]
    hist_features_to_remove = []
    symm_features_to_remove = []
    for feature_idx in list_of_features_to_remove:
        if feature_idx < n_hist_full: 
            hist_features_to_remove.append(feature_idx)
        else:
            symm_features_to_remove.append(feature_idx - n_hist_full)
    
    if len(hist_features_to_remove) > 0:
        # Create the indices of features to keep
        all_indices_hist = torch.arange(n_hist_full)
        keep_indices_hist = all_indices_hist[~torch.isin(all_indices_hist, torch.tensor(hist_features_to_remove))]
        # Select the required features
        X_train = X_train[:, keep_indices_hist, :, :] # (N, n_hist - len(hist_features_to_remove), 20, 2) i.e. torch.Size([3460, 16, 20, 2])
        X_test = X_test[:, keep_indices_hist, :, :]

    # remove symm feautres in removal list features 
    if len(symm_features_to_remove) > 0:
        # Create the indices of features to keep
        all_indices_symm = torch.arange(symm_train.shape[-1]) # n sg full 
        keep_indices = all_indices_symm [~torch.isin(all_indices_symm, torch.tensor(symm_features_to_remove))]
        # Select the required features
        symm_train = symm_train[:, keep_indices] 
        symm_test = symm_test[:, keep_indices]

    n_sg = symm_train.shape[-1] # 11 
    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (21, 20, 2)
    n_histogram = X_train.shape[-3] # 21
    X_train = X_train.reshape(X_train.size(0), -1) # torch.Size([6727, 840])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([1684, 840])
    
    if n_histogram > 0:
        if n_sg > 0:
            # if we have some of both features, concat 
            # add sg feature as the first feature(s), rest are histogram features 
            X_train = torch.cat((symm_train, X_train), -1) # (torch.Size([6727, 851])
            X_test = torch.cat((symm_test, X_test), -1) # torch.Size([1684, 851])) 
        # else: just hist features, no concat needed 
    else:
        # if no hist features left, just using symm features 
        X_train = symm_train
        X_test = symm_test

    return X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "classification_data_metal_3DSCnonsc_symm_20250723"
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_class_data_07_23_2025(
        dataset_name=dataset_name,
        device=device,
        only_test_on_special=False,
    )

