import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME, FOUR_BEST_ORD2_HIST_FEATURES_V2
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *


# 08/29/2025:
# train model w/ the 4, 9, 17, 21 hist features and the 11d symmetry features 
#   4, 9, 17, 21 ranked 2nd among all combinations in prev all-combo experiment (FOUR_BEST_ORD2_HIST_FEATURES_V2)
#   NOTE: we think 9 might be interesting. 

def load_data_08_29_25(
    dataset_name, 
    device, 
):
    assert dataset_name == "good_data_symm_thre1%_no_MgB2"
    path_to_data_sg = f"../data_08_29_2025/{dataset_name}.pkl"
    with open(path_to_data_sg, "rb") as f:
        loaded_data_sg = pickle.load(f)

    sg_good = np.array(loaded_data_sg["symm_feature_good"]) # (4325, 33) for 14 / (4325, 11) for 15  (sg features)

    path_to_data = f"../data_08_29_2025/good_data_ord2_thre1%_with_symm_no_MgB2.pkl" # ord2
    with open(path_to_data, "rb") as f:
        loaded_data = pickle.load(f)
    X_good = loaded_data["X_good"][:,0:21,:,:] # (4325, 32, 20, 2) --> (4325, 21, 20, 2) (order 2 hist featurs)
    y_good = np.array(loaded_data["y_good"]) # (4325,)
    
    X_train, X_test, y_train, y_test, sg_train, sg_test = train_test_split(X_good, y_good, sg_good, test_size=0.2, random_state=2)

    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)

    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)
    sg_train = torch.from_numpy(sg_train).float().to(device=device) # torch.Size([3460])
    sg_test = torch.from_numpy(sg_test).float().to(device=device)
    n_sg = sg_train.shape[-1]

    # Use only the 4 hist features in FOUR_BEST_ORD2_HIST_FEATURES_V2
    keep_indices = torch.tensor(FOUR_BEST_ORD2_HIST_FEATURES_V2).to(device=device)
    X_train = X_train[:, keep_indices, :, :] 
    X_test = X_test[:, keep_indices, :, :]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1) # torch.Size([3460, 1240])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([865, 1240])

    # add sg feature as the first feature(s), rest are histogram features 
    X_train = torch.cat((sg_train, X_train), -1) # torch.Size([3460, 1240 + 33])
    X_test = torch.cat((sg_test, X_test), -1) # torch.Size([865, 1240 + 33]) 

    return X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_id = 19 
    dataset_name = DATA_ID_TO_NAME[data_id]
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_08_29_25(
        dataset_name=dataset_name, 
        device=device, 
    )
    print(f"\nDataset {data_id }: {dataset_name}")
    # Dataset 15: good_data_symm_thre1%
    print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)
    # X_train: torch.Size([3460, 171]) X_test: torch.Size([865, 171]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: (4, 20, 2) n_histogram: 4 n_sg: 11

