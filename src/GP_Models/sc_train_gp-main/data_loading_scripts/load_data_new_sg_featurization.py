import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME, FOUR_BEST_ORD2_HIST_FEATURES 
from path_utils import resolve_replication_data_file
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *


# Load new version of sg featurization data, allow options for 
# 1. only sg, 
# 2. sg + only 4 most importat ord2 feaures 
# 3. sg + all ord2 features 
# For each of these three options, train model for both dataset_name=14,15 (latest sg datasets form Yanjun)
def load_data_new_sg_featurization(
    dataset_name, 
    device, 
    use_hist_features_too=True,
    only_4_best_hist_features=True,
):
    assert dataset_name in ["good_data_symm_33d_thre1%", "good_data_symm_thre1%"]

    path_to_data_sg = resolve_replication_data_file("data_new_sg_featurization", f"{dataset_name}.pkl")
    with open(path_to_data_sg, "rb") as f:
        loaded_data_sg = pickle.load(f)

    sg_good = np.array(loaded_data_sg["symm_feature_good"]) # (4325, 33) for 14 / (4325, 11) for 15  (sg features)

    path_to_data = resolve_replication_data_file("data", "good_data_ord2_thre1%_with_symm.pkl")
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

    if not use_hist_features_too:
        X_train = sg_train 
        X_test = sg_test 
        n_histogram = 0 
        data_shape = None 
    else:
        if only_4_best_hist_features:
            keep_indices = torch.tensor(FOUR_BEST_ORD2_HIST_FEATURES).to(device=device)
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
    for data_id in [14,15]:
        dataset_name = DATA_ID_TO_NAME[data_id]
        # X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_new_sg_featurization(
        for combo in [[True, True], [True, False], [False, False]]:
            X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_new_sg_featurization(
                dataset_name, 
                device, 
                use_hist_features_too=combo[0],
                only_4_best_hist_features=combo[1],
            )
            print(f"\nDataset {data_id }: {dataset_name}, use_hist_features_too:{combo[0]}, only_4_best_hist_features:{combo[1]}")
            print("X_train:",X_train.shape,"X_test:",X_test.shape,"y_train:",y_train.shape,"y_test:",y_test.shape,"data_shape:",data_shape,"n_histogram:",n_histogram,"n_sg:",n_sg)

    # Dataset 14: good_data_symm_33d_thre1%, use_hist_features_too:True, only_4_best_hist_features:True
    # X_train: torch.Size([3460, 193]) X_test: torch.Size([865, 193]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: (4, 20, 2) n_histogram: 4 n_sg: 33

    # Dataset 14: good_data_symm_33d_thre1%, use_hist_features_too:True, only_4_best_hist_features:False
    # X_train: torch.Size([3460, 873]) X_test: torch.Size([865, 873]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: (21, 20, 2) n_histogram: 21 n_sg: 33

    # Dataset 14: good_data_symm_33d_thre1%, use_hist_features_too:False, only_4_best_hist_features:False
    # X_train: torch.Size([3460, 33]) X_test: torch.Size([865, 33]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: None n_histogram: 0 n_sg: 33

    # Dataset 15: good_data_symm_thre1%, use_hist_features_too:True, only_4_best_hist_features:True
    # X_train: torch.Size([3460, 171]) X_test: torch.Size([865, 171]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: (4, 20, 2) n_histogram: 4 n_sg: 11

    # Dataset 15: good_data_symm_thre1%, use_hist_features_too:True, only_4_best_hist_features:False
    # X_train: torch.Size([3460, 851]) X_test: torch.Size([865, 851]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: (21, 20, 2) n_histogram: 21 n_sg: 11

    # Dataset 15: good_data_symm_thre1%, use_hist_features_too:False, only_4_best_hist_features:False
    # X_train: torch.Size([3460, 11]) X_test: torch.Size([865, 11]) y_train: torch.Size([3460]) y_test: torch.Size([865]) data_shape: None n_histogram: 0 n_sg: 11

# What data do we have? 
    # for key in loaded_data.keys():
    #     data1 = loaded_data[key]
    #     type1 = type(data1)
    #     if type1 == list:
    #         data1 = np.array(data1)
    #     if len(data1.shape) == 4:
    #         first_item = data1[0][0][0][0].item() 
    #     elif len(data1.shape) == 3:
    #         first_item = data1[0][0][0].item() 
    #     elif len(data1.shape) == 2:
    #         first_item = data1[0][0].item() 
    #     else:
    #         first_item = data1[0].item() 
    #     print(f"key:{key}, datatype:{type1}, To np array data shape:{data1.shape}, first item: {type(first_item)}, {first_item}")

# Dataset name: good_data_symm_33d_thre1%  (14)
# key:cif_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, /data/3DSC_ICSD/3DSC/superconductors_3D/data/final/ICSD/cifs/Ag0.002Al0.998-ICSD-604645-synth_doped.cif
# key:mat_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, Ag0.002Al0.998
# key:symm_feature_good, datatype:<class 'list'>, To np array data shape:(4325, 33), first item: <class 'float'>, 1.0
# key:symm_feature_name, datatype:<class 'list'>, To np array data shape:(33,), first item: <class 'str'>, i_min


# Dataset name: good_data_symm_thre1%    (15)
# key:cif_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, /data/3DSC_ICSD/3DSC/superconductors_3D/data/final/ICSD/cifs/Ag0.002Al0.998-ICSD-604645-synth_doped.cif
# key:mat_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, Ag0.002Al0.998
# key:symm_feature_good, datatype:<class 'list'>, To np array data shape:(4325, 11), first item: <class 'float'>, 1.0
# key:symm_feature_name, datatype:<class 'list'>, To np array data shape:(11,), first item: <class 'str'>, i


# Data used for labels and hist features: good_data_ord2_thre1%_with_symm
# key:X_good, datatype:<class 'numpy.ndarray'>, To np array data shape:(4325, 32, 20, 2), first item: <class 'float'>, 0.28315
# key:y_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'float'>, 1.128
# key:cif_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, /data/3DSC_ICSD/3DSC/superconductors_3D/data/final/ICSD/cifs/Ag0.002Al0.998-ICSD-604645-synth_doped.cif
# key:mat_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'str'>, Ag0.002Al0.998
# key:sg_good, datatype:<class 'list'>, To np array data shape:(4325,), first item: <class 'int'>, 225
# key:feat_name, datatype:<class 'list'>, To np array data shape:(32,), first item: <class 'str'>, Pauling_EN_mean_2_ord
