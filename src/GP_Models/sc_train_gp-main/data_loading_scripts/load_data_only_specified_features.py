import sys 
sys.path.append("../")
from path_utils import resolve_replication_data_file
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_data_with_only_specified_features(
    dataset_name, 
    device, 
    list_of_features_to_keep="all",
):
    path_to_data = resolve_replication_data_file("data", f"{dataset_name}.pkl")
    with open(path_to_data, "rb") as f:
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

    # keep only listed histogram indicies 
    if list_of_features_to_keep != "all":
        keep_indices = torch.tensor(list_of_features_to_keep) 
        X_train = X_train[:, keep_indices, :, :] 
        X_test = X_test[:, keep_indices, :, :]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 
