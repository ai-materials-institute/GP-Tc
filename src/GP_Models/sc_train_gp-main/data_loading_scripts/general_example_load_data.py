import sys 
sys.path.append("../")
import numpy as np
import torch 
import pickle
from sklearn.model_selection import train_test_split
from utils.PYGraphlets import * 
from utils.Histo_array_scaler import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_and_test_data(
    path_to_data_file,
    hist_features_key='histogram_features',
    labels_key='Tc',
    symm_features_key='symmetry_features',
    list_of_symm_features_to_use=[],
    list_of_hist_features_to_use=[],
    random_split_seed=2,
    test_size=0.2,
    use_oversampling_for_class_data=False,
):
    """
    Load and preprocess data from a .pkl file containing histogram and/or symmetry features,
    split into training and test sets, and return PyTorch tensors ready for model input.

    This function supports flexible selection of histogram and/or symmetry features, handles
    scaling for histogram data, performs random train/test splits, and automatically moves
    all returned tensors to GPU if available.

    Parameters
    ----------
    path_to_data_file : str
        Path to the `.pkl` file containing the dataset. Example: "regression_data_histogram&symmetry.pkl".
    hist_features_key : str, optional
        Key used to access histogram features in the `.pkl` file. Default is 'histogram_features'.
    labels_key : str, optional
        Key used to access target labels in the `.pkl` file. Default is 'Tc'.
    symm_features_key : str, optional
        Key used to access symmetry features in the `.pkl` file. Default is 'symmetry_features'.
    list_of_symm_features_to_use : list of int, optional
        List of indices specifying which symmetry features to keep.
        If empty, only histogram features are used. Default is [].
        Indicies assume zero indexing (so possible symm feature IDs are 0,1,2,...)
        Empty list (default) --> no symm features used (only specified hist features will be used)
    list_of_hist_features_to_use : list of int, optional
        List of indices specifying which histogram features to keep.
        If empty, only symmetry features are used. Default is [].
        Indicies assume zero indexing (so possible hist feature IDs are 0,1,2,...)
        Empty list (default) --> no hist features used (only specified symm features will be used)
    random_split_seed : int, optional
        Random seed for reproducible train/test splits. Default is 2.
    test_size : float, optional
        Fraction of the dataset reserved for the test set. Must be in (0, 1). Default is 0.2.
    use_oversampling_for_class_data : bool, optional
        Whether to apply random oversampling to balance class distributions
        in the training set (for binary classification tasks). If True,
        minority class samples are duplicated until both classes have equal
        representation. Default is False. 
        NOTE: only set to True if loading in a binary classification dataset 

    Returns
    -------
    X_train : torch.FloatTensor
        Training feature tensor of shape (n_train_samples, total_feature_dim).
    X_test : torch.FloatTensor
        Test feature tensor of shape (n_test_samples, total_feature_dim).
    y_train : torch.FloatTensor
        Training labels tensor of shape (n_train_samples,).
    y_test : torch.FloatTensor
        Test labels tensor of shape (n_test_samples,).
    hist_data_shape : tuple of int or None
        Original shape of histogram data before flattening, as (C, H, W),
        or None if histogram features are not used.
    n_hist : int
        Number of histogram features used (after filtering).
    n_symm : int
        Number of symmetry features used (after filtering).

    Notes
    -----
    - At least one of `list_of_hist_features_to_use` or `list_of_symm_features_to_use`
      must be non-empty.
    - Histogram features are scaled using `Histo_Array_Scaler`.
    - Histogram data are reshaped from (C, H, W) to flat vectors per sample.
    - If both feature types are used, symmetry features are concatenated first
      in the final feature tensor.
    - All outputs are moved to the GPU ('cuda') if available.
    """
    n_hist = len(list_of_hist_features_to_use)
    n_symm = len(list_of_symm_features_to_use)
    assert (n_hist + n_symm) > 0, "Need a non-zero number of features to train on" 
    # load data 
    with open(path_to_data_file, "rb") as f:
        loaded_data = pickle.load(f)
    labels = np.array(loaded_data[labels_key])
    if n_hist > 0:
        hist_features = np.array(loaded_data[hist_features_key])
    if n_symm > 0:
        symm_features = np.array(loaded_data[symm_features_key])
    
    # Create random train/test split: 
    if n_hist == 0:
        # only symmetry features 
        symm_train, symm_test, y_train, y_test = train_test_split(symm_features, labels, test_size=test_size, random_state=random_split_seed)
    elif n_symm == 0:
        # only histogram features 
        hist_train, hist_test, y_train, y_test = train_test_split(hist_features, labels, test_size=test_size, random_state=random_split_seed)
    else:
        # both feature types 
        hist_train, hist_test, y_train, y_test, symm_train, symm_test = train_test_split(hist_features, labels, symm_features, test_size=test_size, random_state=random_split_seed)


    # In that case that we have classification data, use oversampling to get 
    #   an equal number of each class in the TRAINING set 
    #   (only relevant for classification w/ binary labels)
    if use_oversampling_for_class_data:
        # Count classes in training set
        classes, counts = np.unique(y_train, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        n_majority = counts.max()
        n_minority = counts.min()
        # Get indices for each class
        minority_idx = np.where(y_train == minority_class)[0]
        # Randomly sample (with replacement) from the minority to match majority count
        oversampled_idx = np.random.choice(minority_idx, size=n_majority - n_minority, replace=True)
        # Concatenate oversampled with original training data
        hist_train_bal = np.concatenate([hist_train, hist_train[oversampled_idx]], axis=0)
        y_train_bal = np.concatenate([y_train, y_train[oversampled_idx]], axis=0)
        symm_train_bal = np.concatenate([symm_train, symm_train[oversampled_idx]], axis=0)
        # Shuffle the balanced training set
        perm = np.random.permutation(len(y_train_bal))
        hist_train = hist_train_bal[perm]
        y_train = y_train_bal[perm]
        symm_train = symm_train_bal[perm]
        print("Training class distribution after balancing:", np.bincount(y_train_bal))
        print("Test class distribution:", np.bincount(y_test))
    
    # convert from numpy to torch 
    y_train = torch.from_numpy(y_train).float().to(device=device)
    y_test = torch.from_numpy(y_test).float().to(device=device) 

    hist_data_shape = None 
    if n_hist > 0:
        # scale histogram features 
        scaler=Histo_Array_Scaler()
        hist_train_rescaled=scaler.fit_transform(hist_train)
        hist_test_rescaled=scaler.transform(hist_test)
        # convert from numpy to torch 
        hist_train = torch.from_numpy(hist_train_rescaled).float().to(device=device) 
        hist_test = torch.from_numpy(hist_test_rescaled).float().to(device=device)
        # remove all but the specified indicies
        keep_indices_hist = torch.tensor(list_of_hist_features_to_use) 
        hist_train = hist_train[:, keep_indices_hist, :, :] 
        hist_test = hist_test[:, keep_indices_hist, :, :] 
        # record original shape of hist data 
        hist_data_shape = (hist_train.shape[-3], hist_train.shape[-2], hist_train.shape[-1]) 
        assert n_hist == hist_train.shape[-3] 
        # reshape histogram feature data for training 
        hist_train = hist_train.reshape(hist_train.size(0), -1) 
        hist_test = hist_test.reshape(hist_test.size(0), -1) 

    
    if n_symm > 0:
        # convert from numpy to torch 
        symm_train = torch.from_numpy(symm_train).float().to(device=device) 
        symm_test = torch.from_numpy(symm_test).float().to(device=device) 
        # remove all but the specified indicies
        keep_indices_symm = torch.tensor(list_of_symm_features_to_use) 
        symm_train = symm_train[:, keep_indices_symm] 
        symm_test = symm_test[:, keep_indices_symm] 
        assert n_symm == symm_train.shape[-1] # e.g., 11 


    if n_hist == 0:
        # only symmetry features 
        X_train = symm_train
        X_test = symm_test 
    elif n_symm == 0:
        # only histogram features 
        X_train = hist_train
        X_test = hist_test
    else:
        # both feature types 
        #   (symm features should always come first
        #   model assumes that symm features are the first n_symm 
        #   and that remaining are hist)
        X_train = torch.cat((symm_train, hist_train), -1) 
        X_test = torch.cat((symm_test, hist_test), -1) 


    # return train and test data 
    return X_train, X_test, y_train, y_test, hist_data_shape, n_hist, n_symm 

