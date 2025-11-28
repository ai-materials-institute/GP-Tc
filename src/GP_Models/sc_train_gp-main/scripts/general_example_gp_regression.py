"""
General Example GP Regression Script.

This script demonstrates how to train and evaluate an Exact Gaussian Process (GP)
regression model using histogram and/or symmetry features. It handles data loading,
model initialization, training, evaluation, and saving of results.

The script is designed to be run from the command line with various arguments
to control the data source, features used, and training hyperparameters.

Author: Natalie Maus
"""

import torch 
import numpy as np 
import gpytorch
import os 
import json
import matplotlib.pyplot as plt 
import argparse 
import sys 
sys.path.append("../")
from data_loading_scripts.general_example_load_data import load_train_and_test_data
from utils.get_performance_stats import get_performance_stats
from utils.get_model_preds import get_model_predictions 
from models.new_sg_featurization_sg_emd_exact_gp import SgEmdExactGPModelV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    save_data_dir, 
    path_to_data_file,
    hist_features_key='histogram_features',
    labels_key='Tc',
    symm_features_key='symmetry_features',
    list_of_symm_features_to_use=[],
    list_of_hist_features_to_use=[],
    random_split_seed=2,
    test_set_size=0.2,
    n_epochs=32,
    n_batches_emd_kernel=10,
    lr=0.1, 
):
    """
    Main function to run the GP regression workflow.

    Parameters
    ----------
    save_data_dir : str
        Directory where results, predictions, model weights, and plots will be saved.
    path_to_data_file : str
        Path to the .pkl data file containing features and targets.
    hist_features_key : str, optional
        Key to access histogram features in the .pkl file.
    labels_key : str, optional
        Key to access labels (targets) in the .pkl file.
    symm_features_key : str, optional
        Key to access symmetry features in the .pkl file.
    list_of_symm_features_to_use : list of int, optional
        List of indices of symmetry features to use.
    list_of_hist_features_to_use : list of int, optional
        List of indices of histogram features to use.
    random_split_seed : int, optional
        Random seed for train/test split.
    test_set_size : float, optional
        Fraction of data used for the test set.
    n_epochs : int, optional
        Number of training epochs.
    n_batches_emd_kernel : int, optional
        Number of batches used to compute the EMD kernel.
    lr : float, optional
        Learning rate for the Adam optimizer.
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
        use_oversampling_for_class_data=False, # only relevant for classification (not regression)
    )
    # initialize likelihood 
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    # initialize model 
    model = SgEmdExactGPModelV2(
        train_x=X_train, 
        train_y=y_train, 
        likelihood=likelihood, 
        n_sg=n_symm,
        map_saas_tau=None,
        n_histogram=n_hist, 
        data_shape=hist_data_shape, 
        n_batches_emd_kernel=n_batches_emd_kernel,
        add_saas_ls_prior=False,
    ).to(device=device)

    # model in train model 
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    # "Loss" for GP regression the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # training loop 
    for _ in range(n_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train) 
        # Calc loss and backprop gradients
        loss = -mll(output, y_train) 
        loss.backward()
        optimizer.step()

    # get trained model predictions on train and test set
    train_preds, test_preds = get_model_predictions(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
    ) # outputs numpy arrays 
    y_train = y_train.cpu().numpy()
    y_test = y_test.cpu().numpy()
    # get performance statistics for regression (on train and test set)
    statistics_dict = get_performance_stats(
        train_preds=train_preds,
        y_train=y_train,
        test_preds=test_preds,
        y_test=y_test,
    )
    statistics_dict = {k: float(v) for k, v in statistics_dict.items()} 
    # save everything in specified save_data_dir
    with open(f"{save_data_dir}/performance_stats.json", 'w') as file: json.dump(statistics_dict, file, indent=4) 
    np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    if n_hist > 0:
        # save emd kernel weights and length scales (kernel for hist features)
        np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
        np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", model.covar_module.lengthscales.detach().cpu().numpy())
    if n_symm > 0:
        # save symm feature kernel lengthscales 
        sg_kernel_lengthscale = model.covar_module.sg_kernel.base_kernel.lengthscale.squeeze().detach().cpu().numpy() 
        np.save(f"{save_data_dir}/sg_kernel_lengthscale.npy", np.array(sg_kernel_lengthscale))

    # plot test set performance 
    test_r2_ = round(statistics_dict["test_r2"], 3)
    test_mae_ = round(statistics_dict["test_mae"], 3)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, test_preds)
    plt.xlabel("True Test Y Value")
    plt.ylabel("Exact GP Model's Predicted Value")
    plt.title(f"Test R2:{test_r2_}, Test MAE:{test_mae_}")
    plt.savefig(f"{save_data_dir}/test_result.png")
    plt.clf()

    # plot train set performance 
    train_r2_ = round(statistics_dict["train_r2"], 3)
    train_mae_ = round(statistics_dict["train_mae"], 3)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, train_preds)
    plt.xlabel("True Train Y Value")
    plt.ylabel("Exact GP Model's Predicted Value")
    plt.title(f"Train R2:{train_r2_}, Train MAE:{train_mae_}")
    plt.savefig(f"{save_data_dir}/train_result.png")
    plt.clf()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run regression using symmetry and/or histogram features with an Exact GP model."
    )

    parser.add_argument('--save_data_dir', type=str, required=True,
                        help='Directory where results, predictions, model weights, and plots will be saved.')

    parser.add_argument('--path_to_data_file', type=str, required=True,
                        help='Path to the .pkl data file (e.g., regression_data_histogram&symmetry.pkl).')

    parser.add_argument('--hist_features_key', type=str, default='histogram_features',
                        help="Key to access histogram features in the .pkl file. Default: 'histogram_features'.")

    parser.add_argument('--labels_key', type=str, default='Tc',
                        help="Key to access labels in the .pkl file. Default: 'Tc'.")

    parser.add_argument('--symm_features_key', type=str, default='symmetry_features',
                        help="Key to access symmetry features in the .pkl file. Default: 'symmetry_features'.")

    parser.add_argument('--list_of_symm_features_to_use', type=int, nargs='*', default=[],
                        help="List of symmetry feature indices to use (e.g., 0 1 2). Default: use none.")

    parser.add_argument('--list_of_hist_features_to_use', type=int, nargs='*', default=[],
                        help="List of histogram feature indices to use (e.g., 0 1 2). Default: use none.")

    parser.add_argument('--random_split_seed', type=int, default=2,
                        help="Random seed used for train/test split. Default: 2.")

    parser.add_argument('--test_set_size', type=float, default=0.2,
                        help="Fraction of data used for test set. Must be between 0 and 1. Default: 0.2.")

    parser.add_argument('--n_epochs', type=int, default=32,
                        help="Number of training epochs for the GP model. Default: 32.")

    parser.add_argument('--n_batches_emd_kernel', type=int, default=10,
                        help="Number of batches used to compute the EMD kernel. Default: 10.")

    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate for the Adam optimizer. Default: 0.1.")

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
    )
