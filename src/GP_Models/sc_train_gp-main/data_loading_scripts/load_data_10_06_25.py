import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *


desired_features = ["EA_abs_2_ord", "AtomicWeight_mean_2_ord", "Column_mean_2_ord", "bond_len_2_ord"]

def load_data_with_only_four_specific_features(
    dataset_name, 
    device, 
):
    list_of_features_to_keep = [13, 18, 26, 30]
    assert dataset_name == "regression_data_histogram&symmetry"
    with open(f"../data_10_6_25/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    # dict_keys(['histogram_features', 'Tc', 'formula', 'class', 'histogram_feat_names', 'symmetry_features', 'symmetry_feature_names'])
    X_good = loaded_data['histogram_features']
    y_good = loaded_data['Tc']
    X_good=np.array(X_good) # (4325, 67, 20, 2)
    y_good=np.array(y_good) # (4325,)
    
    # Sanity check that we're keeping the features we want
    i_ = 0
    for idx in list_of_features_to_keep: 
        keep_feature_name = loaded_data['histogram_feat_names'][idx]
        assert keep_feature_name == desired_features[i_]
        i_ += 1 

    X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=0.2, random_state=2)
    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device) # torch.Size([3460, 67, 20, 2])
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device) # torch.Size([3460])

    # keep only listed histogram indicies 
    keep_indices = torch.tensor(list_of_features_to_keep) 
    X_train = X_train[:, keep_indices, :, :] # torch.Size([3460, 4, 20, 2])
    X_test = X_test[:, keep_indices, :, :] # torch.Size([865, 4, 20, 2])

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (4, 20, 2)
    n_histogram = X_train.shape[-3]
    assert n_histogram == 4 

    X_train = X_train.reshape(X_train.size(0), -1) # (torch.Size([3460, 160])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([865, 160])

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test, data_shape, n_histogram  = load_data_with_only_four_specific_features(
        dataset_name="regression_data_histogram&symmetry", 
        device=device, 
    )