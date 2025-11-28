import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *
from constants import DATA_ID_TO_NAME


def load_class_data_only_specified_features(
    dataset_name, 
    device, 
    list_of_features_to_keep="all",
    order2_features_only=True,
):
    with open(f"../data/{dataset_name}.pkl", "rb") as f:
        loaded_data = pickle.load(f)
    X_all = loaded_data["X_all"] # (7627, 56, 20, 2)
    labels = np.array(loaded_data["label"]) # (7627,)

    # Looking into feature names provided: 
    # ft_names = loaded_data["feat_name"]
    # ord_dict = {
    #     1:[],
    #     2:[],
    #     3:[],
    #     4:[],
    # }
    # for ft_name in ft_names:
    #     if '1_ord' in ft_name:
    #         ord_dict[1].append(ft_name)
    #     elif '2_ord' in ft_name:
    #         ord_dict[2].append(ft_name)
    #     elif '3_ord' in ft_name:
    #         ord_dict[3].append(ft_name)
    #     else:
    #         ord_dict[4].append(ft_name)
    # len(ord_dict[1]) 10,
    # len(ord_dict[2]) 21,
    # len(ord_dict[3]) 35,
    # len(ft_names) 66, 
    # ft_names[10:10+21] # the order 2 features 
    # Note: order 1 names included even though they are not in this dataset 

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
    for dataset_id in [13,16]:
        dataset_name = DATA_ID_TO_NAME[dataset_id]
        print(f"\nDataset {dataset_id}, {dataset_name}:")
        X_train, X_test, y_train, y_test, data_shape, n_histogram  = load_class_data_only_specified_features(
            dataset_name, 
            device, 
            list_of_features_to_keep="all",
            order2_features_only=True,
        )
        print("Keeping all:", n_histogram, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        list_of_features_to_keep = [0,3,14,5]
        X_train, X_test, y_train, y_test, data_shape, n_histogram  = load_class_data_only_specified_features(
            dataset_name, 
            device, 
            list_of_features_to_keep=list_of_features_to_keep,
            order2_features_only=True,
        )
        print(f"Keeping {list_of_features_to_keep}:", n_histogram, X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Dataset 13, classification_data_2025_3_22:
# Keeping all: 21 torch.Size([6101, 840]) torch.Size([1526, 840]) torch.Size([6101]) torch.Size([1526])
# Keeping [0, 3, 14, 5]: 4 torch.Size([6101, 160]) torch.Size([1526, 160]) torch.Size([6101]) torch.Size([1526])

# Dataset 16, classification_data_2025_4_12:
# Keeping all: 21 torch.Size([6101, 840]) torch.Size([1526, 840]) torch.Size([6101]) torch.Size([1526])
# Keeping [0, 3, 14, 5]: 4 torch.Size([6101, 160]) torch.Size([1526, 160]) torch.Size([6101]) torch.Size([1526])