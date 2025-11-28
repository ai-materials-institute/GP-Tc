import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_class_data(dataset_name, device,):
    with open(f"../data/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    # dict_keys(['X_all', 'label', 'cif', 'class', 'feat_name'])
    # 'class' = string labels for classes, not needed 
    # 'cif' = path to cif file, not needed 
    # feature_names_list = loaded_data['feat_name'] # strings w/ features names, but there are 66 not 56 idk why
    X_all = loaded_data["X_all"] # np array (7627, 56, 20, 2) --> 56 hist features 
    labels = np.array(loaded_data["label"]) # list->np array (7627,) 0/1
    # num_non_sc = (labels == 0).sum() # 3302
    # num_sc = (labels == 1).sum() # 4325

    X_train, X_test, y_train, y_test = train_test_split(X_all, labels, test_size=0.2, random_state=2)
    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)
    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1]) # (56, 20, 2)
    n_histogram = X_train.shape[-3] # 56
    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data_id in [13,16]:
        name1 = DATA_ID_TO_NAME[data_id]
        X_train, X_test, y_train, y_test, data_shape, n_histogram = load_class_data(
            dataset_name=name1,
            device=device,
        )
        print(name1, data_shape, n_histogram, X_train.shape, y_train.shape)
    # 13: classification_data_2025_3_22 (56, 20, 2) 56 torch.Size([6101, 2240]) torch.Size([6101])
    # 16: classification_data_2025_4_12 (56, 20, 2) 56 torch.Size([6101, 2240]) torch.Size([6101])

