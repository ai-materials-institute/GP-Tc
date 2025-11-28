import sys 
sys.path.append("../")
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *


# 09/26/25: 
#   Same 11d sg features as 08/01/2025
#   train model with and without on 2 datasets 
# 1. good_data_ord1_20250926.pkl 
# 2. good_data_ordall_20250926.pkl

def load_data_09_26_25(
    hist_features_ord_id, 
    include_sg_features,
    device, 
):
    sg_dataset_name = "good_data_symm_thre1%"
    path_to_data_sg = f"../data_new_sg_featurization/{sg_dataset_name}.pkl"
    with open(path_to_data_sg, "rb") as f:
        loaded_data_sg = pickle.load(f)
    sg_good = np.array(loaded_data_sg["symm_feature_good"]) # (4325, 33) for 14 / (4325, 11) for 15  (sg features)

    assert hist_features_ord_id in ["ord1", "ordall"] # two options we want to test (w/ and w/ out sg)
    path_to_data = f"../data_09_26_2025/good_data_{hist_features_ord_id}_20250926.pkl"
    with open(path_to_data, "rb") as f:
        loaded_data = pickle.load(f)
    X_good = loaded_data["X_good"] # (4325, 67, 20, 2)
    if hist_features_ord_id == "ordall":
        # Removed dim [:,66,:,:] (fit_transform converts this whole dim to NaNs!)
        X_good_1 = X_good[:,0:66,:,:]
        X_good_2 = X_good[:,67:,:,:]
        X_good = np.concatenate((X_good_1, X_good_2), axis=1) # (4325, 66, 20, 2)
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

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1) # torch.Size([3460, 1240])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([865, 1240])

    if include_sg_features:
        # add sg feature as the first feature(s), rest are histogram features 
        X_train = torch.cat((sg_train, X_train), -1) # torch.Size([3460, 1240 + 33])
        X_test = torch.cat((sg_test, X_test), -1) # torch.Size([865, 1240 + 33]) 
    else:
        n_sg = 0 

    return X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # V1
    print("\nord1 No Sg Featurs:")
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_09_26_25(
        hist_features_ord_id="ord1", 
        device=device, 
        include_sg_features=False,
    )
    print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)
    
    # V2
    print("\nord1 w/ Sg Featurs:")
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_09_26_25(
        hist_features_ord_id="ord1", 
        device=device, 
        include_sg_features=True,
    )
    print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)
    
    # V3
    print("\nordall No Sg Featurs:")
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_09_26_25(
        hist_features_ord_id="ordall", 
        device=device, 
        include_sg_features=False,
    )
    print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)
    
    # V4
    print("\nordall w/ Sg Featurs:")
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_09_26_25(
        hist_features_ord_id="ordall", 
        device=device, 
        include_sg_features=True,
    )
    print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)
    

