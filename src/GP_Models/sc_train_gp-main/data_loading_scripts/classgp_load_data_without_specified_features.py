import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *


def load_class_data_remove_specified_features(
    dataset_name, 
    device, 
    list_of_features_to_remove=[],
    order2_features_only=True,
):
    with open(f"../data/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    X_all = loaded_data["X_all"] # (7627, 56, 20, 2)
    labels = np.array(loaded_data["label"]) # (7627,)

    X_train, X_test, y_train, y_test = train_test_split(X_all, labels, test_size=0.2, random_state=2)
    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)

    # X_train.shape torch.Size([6101, 56, 20, 2])
    #  X_train[:,0:21,:,:] .shape # torch.Size([6101, 21, 20, 2])
    if order2_features_only:
        X_train = X_train[:,0:21,:,:] 
        X_test= X_test[:,0:21,:,:] 

    # remove feautres in removal list features 
    if len(list_of_features_to_remove) > 0:
        # Create the indices of features to keep
        all_indices = torch.arange(X_train.shape[-3])
        keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(list_of_features_to_remove))]
        # Select the required features
        X_train = X_train[:, keep_indices, :, :] # (N, n_hist - len(list_of_features_to_remove), 20, 2) i.e. torch.Size([3460, 16, 20, 2])
        X_test = X_test[:, keep_indices, :, :]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (56, 20, 2)
    n_histogram = X_train.shape[-3] # 56
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    from constants import DATA_ID_TO_NAME
    dataset_id = 13
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = DATA_ID_TO_NAME[dataset_id]
    X_train, X_test, y_train, y_test, data_shape, n_histogram  = load_class_data_remove_specified_features(
        dataset_name, 
        device, 
        list_of_features_to_remove=[],
        order2_features_only=True,
    )
    print(n_histogram, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train, X_test, y_train, y_test, data_shape, n_histogram  = load_class_data_remove_specified_features(
        dataset_name, 
        device, 
        list_of_features_to_remove=[0, 5, 20],
        order2_features_only=True,
    )
    print(n_histogram, X_train.shape, X_test.shape, y_train.shape, y_test.shape)