import sys 
sys.path.append("../")
from data_loading_scripts.load_data_10_06_25 import load_data_with_only_four_specific_features
from constants import DATA_ID_TO_NAME
import matplotlib.pyplot as plt 
import json 
import torch 
import gc 
import copy 
import pandas as pd 
import gpytorch
import argparse 
from models.emd_exact_gp import EMDExactGPModel
from utils.get_model_preds import get_model_predictions
from utils.get_performance_stats import get_performance_stats
from utils.create_wandb_tracker import create_tracker
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DATA_DIRECTORY = "../save_model_data"

# 10/06/25 
# Train model with only the features: 
# # ["EA_abs_2_ord", "AtomicWeight_mean_2_ord", "Column_mean_2_ord", "bond_len_2_ord"]

def main(
    save_data_dir,
    dataset_name, 
    tracker,
    n_epochs=32, 
):
    lr=0.1 
    n_batches_emd_kernel=10 # shouldn't have to worry about OOM w/ only i.e. 4 hist features 
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_with_only_four_specific_features(
        dataset_name=dataset_name,
        device=device, 
    )
    plot_titles_string = "Performance w Hist. Features: 13, 18, 26, 30"
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model = EMDExactGPModel(
        train_x=X_train, 
        train_y=y_train, 
        likelihood=likelihood, 
        data_shape=data_shape, 
        n_histogram=n_histogram,
        map_saas_tau=None,
        add_saas_ls_prior=False,
        n_batches_emd_kernel=n_batches_emd_kernel,
    ).to(device=device)
    # train model 
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(n_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train) 
        # Calc loss and backprop gradients
        loss = -mll(output, y_train) 
        loss.backward()
        optimizer.step()

    train_preds, test_preds = get_model_predictions(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
    ) # outputs numpy arrays 
    y_test = y_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    statistics_dict = get_performance_stats(
        train_preds=train_preds,
        y_train=y_train,
        test_preds=test_preds,
        y_test=y_test,
    )
    statistics_dict = {k: float(v) for k, v in statistics_dict.items()}
    test_r2_ = statistics_dict["test_r2"]
    test_mae_ = statistics_dict["test_mae"]
    train_r2_ = statistics_dict["train_r2"]
    train_mae_ = statistics_dict["train_mae"]

    with open(f"{save_data_dir}/performance_stats.json", 'w') as file: json.dump(statistics_dict, file, indent=4) 
    tracker.log(statistics_dict)


    np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
    learned_ls = model.covar_module.lengthscales.detach().cpu().numpy()
    np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", learned_ls)

    # test
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, test_preds)
    plt.xlabel("True Test Y Value")
    plt.ylabel("Exact GP Model's Predicted Value")
    title_string = "Test " + plot_titles_string
    title_string = title_string + f"\n R2:{test_r2_:.3f}, MAE:{test_mae_:.3f}"
    plt.title(title_string)
    plt.savefig(f"{save_data_dir}/test_result.png")
    plt.clf()

    # train 
    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, train_preds)
    plt.xlabel("True Train Y Value")
    plt.ylabel("Exact GP Model's Predicted Value")
    title_string = "Train " + plot_titles_string
    title_string = title_string + f"\nR2:{train_r2_:.3f}, MAE:{train_mae_:.3f}"
    plt.title(title_string)
    plt.savefig(f"{save_data_dir}/train_result.png")
    plt.clf()

    # empty garbage, clear cache
    gc.collect()
    torch.cuda.empty_cache()



def add_to_dir(dir_path, add_this):
    dir_path = dir_path + f"/{add_this}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path 
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        help="int for dataset to train on",
        type=int,
        default=21,
        required=False,
    )
    parser.add_argument(
        "--n_epochs",
        help="int, num training epochs",
        type=int,
        default=32, 
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    wandb_config_dict = {
        "dataset_id":args.dataset_id,
        "dataset_name":dataset_name,
        "n_epochs":args.n_epochs,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-10-06-25",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"{wandb_run_name}_4fts_epochs{args.n_epochs}_10_06_25")
    main(
        save_data_dir=save_data_dir_toptop,
        dataset_name=dataset_name, 
        tracker=tracker,
        n_epochs=args.n_epochs,
    )
