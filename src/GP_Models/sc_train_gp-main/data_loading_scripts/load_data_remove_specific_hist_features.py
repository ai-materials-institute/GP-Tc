import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_data_without_specified_features(
    dataset_name, 
    device, 
    list_of_features_to_remove=[],
    also_remove_ord1_features=False,
    symm_features_only=False,
):
    with open(f"../data/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    X_good = loaded_data["X_good"]
    y_good = loaded_data["y_good"]
    X_good=np.array(X_good)
    y_good=np.array(y_good)
    X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=0.2, random_state=2)
    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)

    if also_remove_ord1_features:
        assert not symm_features_only
        X_train = X_train[:,10:,:,:] # torch.Size([3460, 21, 20, 2])
        X_test= X_test[:,10:,:,:] # torch.Size([865, 21, 20, 2]) --> n_histogram 21 
    if symm_features_only:
        assert not also_remove_ord1_features
        assert dataset_name == "good_data_ord2_thre1%_with_symm" # last 11 are symm for this dataset 
        X_train = X_train[:,21:,:,:] # last 11 hist features are symm features 
        X_test= X_test[:,21:,:,:] # last 11 hist features are symm features 


    # remove undesirable features 
    if len(list_of_features_to_remove) > 0:
        # Create the indices of features to keep
        all_indices = torch.arange(X_train.shape[-3])
        keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(list_of_features_to_remove))]
        # Select the required features
        X_train = X_train[:, keep_indices, :, :] # (N, n_hist - len(list_of_features_to_remove), 20, 2) i.e. torch.Size([3460, 16, 20, 2])
        X_test = X_test[:, keep_indices, :, :]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 

