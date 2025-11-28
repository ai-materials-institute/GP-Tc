import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME
import numpy as np
import torch 
from utils.PYGraphlets import *
import pickle
from sklearn.model_selection import train_test_split
from utils.Histo_array_scaler import *

def load_data(dataset_name, device, remove_ord1_features=False):
    with open(f"../data/{dataset_name}.pkl", "rb") as f:
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

    X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=0.2, random_state=2)

    scaler=Histo_Array_Scaler()

    X_train_rescaled=scaler.fit_transform(X_train)
    X_test_rescaled=scaler.transform(X_test)

    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)

    if remove_ord1_features:
        X_train = X_train[:,10:,:,:] # torch.Size([3460, 21, 20, 2])
        X_test= X_test[:,10:,:,:] # torch.Size([865, 21, 20, 2])
        # data_shape: (21, 20, 2)
        # n_histogram 21 

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    return X_train, X_test, y_train, y_test, data_shape, n_histogram 


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    remove_ord1_features = True 
    for id in [1,2,3,4,6,7,8,9]:
        name1 = DATA_ID_TO_NAME[id]
        X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data(
            dataset_name=name1,
            device=device,
            remove_ord1_features=remove_ord1_features,
        )
        print(name1, data_shape, n_histogram, X_train.shape, y_train.shape)
    
    # Data shapes per dataset: 
    # 1,2,5: n_histogram=31, data_shape=(31, 20, 2)
    # 3,4: n_histogram=64, data_shape=(64, 20, 2)

# good_data_thre1% (31, 20, 2) 31 torch.Size([3460, 1240]) torch.Size([3460])
# good_data_ord2_thre1% (31, 20, 2) 31 torch.Size([3460, 1240]) torch.Size([3460])
# good_data_ord3_thre0.001% (64, 20, 2) 64 torch.Size([2770, 2560]) torch.Size([2770])
# good_data_ord3_thre1% (64, 20, 2) 64 torch.Size([3460, 2560]) torch.Size([3460])
# good_data_thre0.001% (31, 20, 2) 31 torch.Size([2770, 1240]) torch.Size([2770])

    