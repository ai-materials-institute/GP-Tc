import sys
from pathlib import Path

import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split

SC_TRAIN_ROOT = Path(__file__).resolve().parents[2]
if str(SC_TRAIN_ROOT) not in sys.path:
    sys.path.append(str(SC_TRAIN_ROOT))

from path_utils import resolve_replication_data_file
from utils.Histo_array_scaler import Histo_Array_Scaler


def load_data_09_26_25(hist_features_ord_id, include_sg_features, device):
    sg_dataset_name = 'good_data_symm_thre1%'
    path_to_data_sg = resolve_replication_data_file('data_new_sg_featurization', f'{sg_dataset_name}.pkl')
    with open(path_to_data_sg, 'rb') as file:
        loaded_data_sg = pickle.load(file)
    sg_good = np.array(loaded_data_sg['symm_feature_good'])

    assert hist_features_ord_id in ['ord1', 'ordall']
    path_to_data = resolve_replication_data_file('data_09_26_2025', f'good_data_{hist_features_ord_id}_20250926.pkl')
    with open(path_to_data, 'rb') as file:
        loaded_data = pickle.load(file)
    X_good = loaded_data['X_good']
    if hist_features_ord_id == 'ordall':
        X_good_1 = X_good[:, 0:66, :, :]
        X_good_2 = X_good[:, 67:, :, :]
        X_good = np.concatenate((X_good_1, X_good_2), axis=1)
    y_good = np.array(loaded_data['y_good'])

    X_train, X_test, y_train, y_test, sg_train, sg_test = train_test_split(
        X_good,
        y_good,
        sg_good,
        test_size=0.2,
        random_state=2,
    )
    scaler = Histo_Array_Scaler()
    X_train_rescaled = scaler.fit_transform(X_train)
    X_test_rescaled = scaler.transform(X_test)

    X_train = torch.from_numpy(X_train_rescaled).float().to(device=device)
    X_test = torch.from_numpy(X_test_rescaled).float().to(device=device)
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device)
    sg_train = torch.from_numpy(sg_train).float().to(device=device)
    sg_test = torch.from_numpy(sg_test).float().to(device=device)
    n_sg = sg_train.shape[-1]

    data_shape = (X_train.shape[-3], X_train.shape[-2], X_train.shape[-1])
    n_histogram = X_train.shape[-3]

    X_train = X_train.reshape(X_train.size(0), -1)
    X_test = X_test.reshape(X_test.size(0), -1)

    if include_sg_features:
        X_train = torch.cat((sg_train, X_train), -1)
        X_test = torch.cat((sg_test, X_test), -1)
    else:
        n_sg = 0

    return X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg
