import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

# 1. train model on all ord2 histograms + symmetry features (ord2 hist = 10:31)
#       --> list_of_hist_features_to_keep = np.arange(10, 31).tolist()
# 2. train model on ([13, 18, 26, 30]) + symmetry features
#       --> list_of_hist_features_to_keep = [13, 18, 26, 30]

def load_data_specified_hist_features_and_all_symm_10_21_25(
    dataset_name, 
    device, 
    list_of_hist_features_to_keep=[13, 18, 26, 30],
):
    assert dataset_name == "regression_data_histogram&symmetry"
    with open(f"../data_10_6_25/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    # dict_keys(['histogram_features', 'Tc', 'formula', 'class', 'histogram_feat_names', 'symmetry_features', 'symmetry_feature_names'])
    X_good = loaded_data['histogram_features']
    y_good = loaded_data['Tc']
    symm_good = loaded_data['symmetry_features']
    
    X_good = np.array(X_good) # (4325, 67, 20, 2)
    y_good = np.array(y_good) # (4325,)
    symm_good = np.array(symm_good) # (4325, 11)

    X_train, X_test, y_train, y_test, symm_train, symm_test = train_test_split(X_good, y_good, symm_good, test_size=0.2, random_state=2)
    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device) # torch.Size([3460, 67, 20, 2])
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device) # torch.Size([3460])
    symm_train = torch.from_numpy(symm_train).float().to(device=device) 
    symm_test = torch.from_numpy(symm_test).float().to(device=device) 

    # keep only listed histogram indicies 
    keep_indices = torch.tensor(list_of_hist_features_to_keep) 
    X_train = X_train[:, keep_indices, :, :] # torch.Size([3460, 4, 20, 2])
    X_test = X_test[:, keep_indices, :, :] # torch.Size([865, 4, 20, 2])

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (4, 20, 2)
    n_histogram = X_train.shape[-3]
    n_sg = symm_train.shape[-1] # 11 

    X_train = X_train.reshape(X_train.size(0), -1) # (torch.Size([3460, 160])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([865, 160])

    # add sg feature as the first feature(s), rest are histogram features 
    X_train = torch.cat((symm_train, X_train), -1) 
    X_test = torch.cat((symm_test, X_test), -1) 

    return X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg
