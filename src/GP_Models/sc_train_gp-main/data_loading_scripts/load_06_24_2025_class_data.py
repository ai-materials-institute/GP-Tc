import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_class_data_06_24_2025(dataset_name, device, only_test_on_special=False):
    # only_test_on_special: only test on 'special' final two inputs: MgB2 and graphite
    #   otherwise test on special final two inputs plus significant additional test set  
    assert dataset_name == "classification_data_whole_20250528"
    with open(f"../data_class_fixed_06_24_2025/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    # dict_keys(['X_all', 'label', 'cif', 'formula', 'class', 'feat_name'])
    #   'cif': paths to cif files, not needed 
    #   'formula': string formulas, not needed 
    #   'class': string class names, not needed 
    #   'feat_name': string names of features, not needed 
    X_all = loaded_data["X_all"] # np array (5737, 21, 20, 2) --> 21 hist features 
    labels = np.array(loaded_data["label"])  # np array (5737,)  dtype('int64') (1's and 0's)

    # num_non_sc = (labels == 0).sum() # 1530
    # num_sc = (labels == 1).sum() # 4207

    special_test_x = X_all[-2:] # (2, 21, 20, 2)
    special_test_y = labels[-2:] # (2,)  # array([1, 0])
    x_without_special = X_all[0:-2] # (5735, 21, 20, 2)
    y_without_special = labels[0:-2] # (5735,)

    if only_test_on_special:
        X_train = x_without_special
        X_test = special_test_x
        y_train = y_without_special
        y_test = special_test_y
    else:
        X_train, X_test, y_train, y_test = train_test_split(x_without_special, y_without_special, test_size=0.2, random_state=2)
        X_test = np.concatenate((X_test, special_test_x))
        y_test = np.concatenate((y_test, special_test_y))

    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (21, 20, 2)
    n_histogram = X_train.shape[-3] # 21
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = "classification_data_whole_20250528"
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_class_data_06_24_2025(
        dataset_name=dataset_name,
        device=device,
    )

    # 13: classification_data_2025_3_22 (56, 20, 2) 56 torch.Size([6101, 2240]) torch.Size([6101])
    # 16: classification_data_2025_4_12 (56, 20, 2) 56 torch.Size([6101, 2240]) torch.Size([6101])

