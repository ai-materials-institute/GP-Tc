import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from utils.Histo_array_scaler import *

def load_special_example_class_data(
    device,
    order2_features_only=True,
    list_of_features_to_keep="all",
):
    # load train data
    with open(f"../data_for_special_class_expt_4_16_25/classification_data_no_MgB2&C_train.pkl", "rb") as f1:
        loaded_train_data = pickle.load(f1) # dict_keys(['X_all', 'label', 'cif', 'feat_name'])
    # load test data 
    with open(f"../data_for_special_class_expt_4_16_25/classification_data_MgB2&C_test.pkl", "rb") as f2:
        loaded_test_data = pickle.load(f2) # dict_keys(['X_MgB2_C', 'label', 'feat_name'])
    
    y_test_labels = loaded_test_data['label'] # list ['MgB2', 'C']
    y_test = []
    for lab in y_test_labels:
        if lab == 'MgB2':
            y_test.append(1) # SC 
        elif lab == 'C':
            y_test.append(0) # NON-SC
        else:
            assert 0, f"invalid test label: {lab}"
    # Np arrays: 
    X_train = loaded_train_data['X_all'] # (8412, 56, 20, 2) np array 
    y_train = np.array(loaded_train_data['label']) # (8412,) # array([1, 1, 1, ..., 0, 0, 0])
    X_test = loaded_test_data['X_MgB2_C'] # (2, 56, 20, 2)
    y_test = np.array(y_test) #  (2,) array([1, 0])

    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device) 

    if order2_features_only:
        X_train = X_train[:,0:21,:,:] 
        X_test= X_test[:,0:21,:,:] 
    
    # keep only listed histogram indicies 
    if list_of_features_to_keep != "all":
        keep_indices = torch.tensor(list_of_features_to_keep) 
        X_train = X_train[:, keep_indices, :, :] 
        X_test = X_test[:, keep_indices, :, :]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (56, 20, 2)
    n_histogram = X_train.shape[-3] # 56
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_special_example_class_data(
        device=device,
        order2_features_only=True,
        list_of_features_to_keep="all",
    )
    print(data_shape, n_histogram, X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_special_example_class_data(
        device=device,
        order2_features_only=False,
        list_of_features_to_keep="all",
    )
    print(data_shape, n_histogram, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_special_example_class_data(
        device=device,
        order2_features_only=True,
        list_of_features_to_keep=[0,4,7,12],
    )
    print(data_shape, n_histogram, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

