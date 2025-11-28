"""
General Example GP Classification Script.

This script demonstrates how to train and evaluate a Variational Gaussian Process (GP)
classification model using histogram and/or symmetry features. It handles data loading,
model initialization, training (using Variational ELBO), evaluation, and saving of results.

The script is designed to be run from the command line with various arguments
to control the data source, features used, and training hyperparameters.

Author: Natalie Maus
"""

import sys 
sys.path.append("../")
from data_loading_scripts.general_example_load_data import load_train_and_test_data
from utils.str2bool_for_argparse import str2bool
import torch 
import os 
import json 
import gpytorch
import argparse 
from torch.utils.data import TensorDataset, DataLoader
from models.class_emd_gp_w_sg import EmdSgGpClassificationModel
import numpy as np 
from utils.classification_pred_utils import get_class_model_preds
from utils.get_acc_stats_classification import get_performance_stats_classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(
    save_data_dir, 
    path_to_data_file,
    hist_features_key='histogram_features',
    labels_key='label',
    symm_features_key='symmetry_features',
    list_of_symm_features_to_use=[],
    list_of_hist_features_to_use=[],
    random_split_seed=2,
    test_set_size=0.2,
    n_epochs=32,
    n_batches_emd_kernel=10,
    lr=0.05, 
    n_inducing_pts=1024,
    use_oversampling=True,
    mini_batch_size=1024, # NOTE: use smaller bsz if you run into GPU OOM issues 
):
    """
    Main function to run the GP classification workflow.

    Parameters
    ----------
    save_data_dir : str
        Directory where results, model weights, predictions, and kernel parameters will be saved.
    path_to_data_file : str
        Path to the .pkl file containing the dataset.
    hist_features_key : str, optional
        Key for accessing histogram features in the .pkl file.
    labels_key : str, optional
        Key for accessing binary class labels in the .pkl file.
    symm_features_key : str, optional
        Key for accessing symmetry features in the .pkl file.
    list_of_symm_features_to_use : list of int, optional
        Indices of symmetry features to use.
    list_of_hist_features_to_use : list of int, optional
        Indices of histogram features to use.
    random_split_seed : int, optional
        Random seed for reproducible train/test split.
    test_set_size : float, optional
        Fraction of data to use for the test set.
    n_epochs : int, optional
        Number of training epochs.
    n_batches_emd_kernel : int, optional
        Number of batches used to compute the EMD kernel.
    lr : float, optional
        Learning rate for the Adam optimizer.
    n_inducing_pts : int, optional
        Number of inducing points for variational GP classification.
    use_oversampling : bool, optional
        Whether to apply oversampling to balance class labels in the training set.
    mini_batch_size : int, optional
        Mini-batch size for training.
    """
    # create save_data_dir to save outputs 
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    # load data 
    X_train, X_test, y_train, y_test, hist_data_shape, n_hist, n_symm = load_train_and_test_data(
        path_to_data_file=path_to_data_file, 
        hist_features_key=hist_features_key, 
        labels_key=labels_key, 
        symm_features_key=symm_features_key, 
        list_of_symm_features_to_use=list_of_symm_features_to_use,
        list_of_hist_features_to_use=list_of_hist_features_to_use,
        random_split_seed=random_split_seed, 
        test_size=test_set_size, 
        use_oversampling_for_class_data=use_oversampling, # only relevant for classification (not regression)
    )
    # initialize likelihood and model
    model = EmdSgGpClassificationModel(
        train_x=X_train[0:n_inducing_pts], 
        n_histogram=n_hist,
        n_sg=n_symm,
        data_shape=hist_data_shape, 
        n_batches_emd_kernel=n_batches_emd_kernel,
    ).to(device=device)
    # binary classification likelihood: BernoulliLikelihood
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device=device)
    # model in train model 
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    # "Loss" for binary classification GP (the ELBO)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
    # set up training dataset 
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)
    # train model 
    for _ in range(n_epochs):
        for (x_batch, y_batch) in train_loader:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_batch) 
            # Calc loss and backprop gradients
            loss = -mll(output, y_batch) 
            loss.backward()
            optimizer.step()

    # get test and train preds 
    train_preds, test_preds = get_class_model_preds(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
        bsz=mini_batch_size,
    ) # outputs numpy arrays of classification preds 
    y_test = y_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    statistics_dict = get_performance_stats_classification(
        y_test=y_test,
        y_train=y_train,
        test_preds=test_preds,
        train_preds=train_preds,
    )
    np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    if n_hist > 0:
        # save hist feature kernel lengthscales, weights 
        np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
        learned_ls = model.covar_module.lengthscales.detach().cpu().numpy()
        np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", learned_ls)
    if n_symm > 0:
        # save symm feature kernel lengthscales 
        sg_kernel_lengthscale = model.covar_module.sg_kernel.base_kernel.lengthscale.squeeze().detach().cpu().numpy() 
        np.save(f"{save_data_dir}/sg_kernel_lengthscale.npy", np.array(sg_kernel_lengthscale))
    # save peroformance stats 
    with open(f"{save_data_dir}/performance.json", 'w') as file:
        json.dump(statistics_dict, file, indent=4) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run variational GP classification using symmetry and/or histogram features."
    )

    parser.add_argument('--save_data_dir', type=str, required=True,
                        help='Directory where results, model weights, predictions, and kernel parameters will be saved.')

    parser.add_argument('--path_to_data_file', type=str, required=True,
                        help='Path to the .pkl file containing the dataset.')

    parser.add_argument('--hist_features_key', type=str, default='histogram_features',
                        help="Key for accessing histogram features in the .pkl file. Default: 'histogram_features'.")

    parser.add_argument('--labels_key', type=str, default='label',
                        help="Key for accessing binary class labels in the .pkl file. Default: 'label'.")

    parser.add_argument('--symm_features_key', type=str, default='symmetry_features',
                        help="Key for accessing symmetry features in the .pkl file. Default: 'symmetry_features'.")

    parser.add_argument('--list_of_symm_features_to_use', type=int, nargs='*', default=[],
                        help="Indices of symmetry features to use (e.g., 0 1 2). Default: use none.")

    parser.add_argument('--list_of_hist_features_to_use', type=int, nargs='*', default=[],
                        help="Indices of histogram features to use (e.g., 0 1 2). Default: use none.")

    parser.add_argument('--random_split_seed', type=int, default=2,
                        help="Random seed used for reproducible train/test split. Default: 2.")

    parser.add_argument('--test_set_size', type=float, default=0.2,
                        help="Fraction of data to use for the test set. Must be between 0 and 1. Default: 0.2.")

    parser.add_argument('--n_epochs', type=int, default=32,
                        help="Number of training epochs. Default: 32.")

    parser.add_argument('--n_batches_emd_kernel', type=int, default=10,
                        help="Number of batches used to compute the EMD kernel. Default: 10.")

    parser.add_argument('--lr', type=float, default=0.05,
                        help="Learning rate for the Adam optimizer. Default: 0.05.")

    parser.add_argument('--n_inducing_pts', type=int, default=1024,
                        help="Number of inducing points for variational GP classification. Default: 1024.")

    parser.add_argument('--use_oversampling', type=str2bool, default=True,
                        help="Whether to apply oversampling to balance class labels in the training set. Default: True.")

    parser.add_argument('--mini_batch_size', type=int, default=1024,
                        help="Mini-batch size for training. Use smaller values if running into GPU memory issues. Default: 1024.")

    args = parser.parse_args()

    main(
        save_data_dir=args.save_data_dir,
        path_to_data_file=args.path_to_data_file,
        hist_features_key=args.hist_features_key,
        labels_key=args.labels_key,
        symm_features_key=args.symm_features_key,
        list_of_symm_features_to_use=args.list_of_symm_features_to_use,
        list_of_hist_features_to_use=args.list_of_hist_features_to_use,
        random_split_seed=args.random_split_seed,
        test_set_size=args.test_set_size,
        n_epochs=args.n_epochs,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        lr=args.lr,
        n_inducing_pts=args.n_inducing_pts,
        use_oversampling=args.use_oversampling,
        mini_batch_size=args.mini_batch_size,
    )
