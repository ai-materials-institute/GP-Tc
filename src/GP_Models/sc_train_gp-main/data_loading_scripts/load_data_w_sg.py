import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
import os 
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_data_with_sgs(
    dataset_name, 
    device, 
    remove_ord1_features=False,
    use_11d_symm_sg_features=False,
):
    path_to_data = f"../data_w_sg/{dataset_name}.pkl"
    if not os.path.exists(path_to_data):
        # okay to use other data without sg_good i.e. when adding 11d_symm_sg_features instead
        path_to_data = f"../data/{dataset_name}.pkl"
    with open(path_to_data, "rb") as f:
        loaded_data = pickle.load(f)

    X_good = loaded_data["X_good"]
    # X_rescaled_good = loaded_data['X_rescaled_good']
    y_good = loaded_data["y_good"]
    # cif_good = loaded_data['cif_good']
    # mat_good = loaded_data['mat_good']

    # X_rescaled_good has bin centers rescaled, but it's fit_transfromed on the whole icsd data set
    # if you want to strictly fit_transform the training set and transform the test set
    X_good=np.array(X_good)
    y_good=np.array(y_good)

    if use_11d_symm_sg_features:
        symm_sg_features_name = "good_data_symm_thre1%"
        with open(f"../data_w_sg/{symm_sg_features_name}.pkl", "rb") as f_sg:
            loaded_data_sg = pickle.load(f_sg)
            sg_good = loaded_data_sg["symm_feature_good"]
            sg_good = np.array(sg_good) # N,11
    else:
        sg_good = loaded_data["sg_good"] # list of integers, len(sgs) == X_good.shape[0]
        sg_good = np.array(sg_good) # each value is an int giving the sg id / graph node id for material 

    X_train, X_test, y_train, y_test, sg_train, sg_test = train_test_split(X_good, y_good, sg_good, test_size=0.2, random_state=2)

    scaler=Histo_Array_Scaler()
    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)

    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)
    sg_train = torch.from_numpy(sg_train ).float().to(device=device) # torch.Size([3460])
    sg_test = torch.from_numpy(sg_test).float().to(device=device)

    if remove_ord1_features:
        X_train = X_train[:,10:,:,:] 
        X_test= X_test[:,10:,:,:] 

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1) # torch.Size([3460, 1240])
    X_test = X_test.reshape(X_test.size(0), -1) # torch.Size([865, 1240])
    if len(sg_train.shape) == 1:
        sg_train = sg_train.unsqueeze(-1) # (N,1)
        sg_test = sg_test.unsqueeze(-1) # (N,1)
    # add sg feature as the first feature(s), rest are histogram features 
    X_train = torch.cat((sg_train, X_train), -1) # torch.Size([3460, 1241])
    X_test = torch.cat((sg_test, X_test), -1) # torch.Size([865, 1241]) * OR (N,1251) if use_11d_symm_sg_features=True

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    remove_ord1_features = False 
    use_11d_symm_sg_features = True 
    for id in [5,9]: # [3,5]: # datasets w/ sgs 
        name1 = DATA_ID_TO_NAME[id]
        X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_with_sgs(
            dataset_name=name1,
            device=device,
            remove_ord1_features=remove_ord1_features,
            use_11d_symm_sg_features=use_11d_symm_sg_features,
        )
        print(name1, data_shape, n_histogram, X_train.shape, y_train.shape)
