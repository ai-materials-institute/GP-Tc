# TO-DO: brute force over all combinations of four features (both datasets)
# Which four get best test r2? 

import sys 
sys.path.append("../")
from data_loading_scripts.load_data_only_specified_features import load_data_with_only_specified_features
from constants import DATA_ID_TO_NAME
import matplotlib.pyplot as plt 
import torch 
import gc 
import copy 
from itertools import combinations
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

# path = ../data/datasetset_name.pdkl

def main(
    save_data_dir_top,
    dataset_name, 
    tracker,
    wandb_run_name,
    n_features_keep=4,
    n_epochs=32, 
    missing_set=None,
):
    lr=0.1 
    n_batches_emd_kernel=10 # shouldn't have to worry about OOM w/ only i.e. 4 hist features 
    X_train, X_test, y_train, y_test, data_shape, n_histogram_full = load_data_with_only_specified_features(
        dataset_name=dataset_name,
        device=device,
        list_of_features_to_keep="all",
    )
    if missing_set is None:
        # all combination of n_features_keep that we could select 
        all_combos = torch.tensor(list(combinations(range(n_histogram_full), n_features_keep))) # torch.Size([35960, 4]) for 32, 4 
        # randomly shuffle combos to prevent runs from overlapping (write error)
        num_rows = all_combos.size(0)
        random_indices = torch.randperm(num_rows)
        all_combos = all_combos[random_indices]
    else:
        all_combos = [missing_set]
        print("Running missing set: ", missing_set)

    # record data 
    test_r2s = []
    test_maes = []
    train_r2s = []
    train_maes = []
    features_kept_strings = []
    for features_to_keep in all_combos: 
        if missing_set is None:
            features_to_keep = features_to_keep.tolist()
        plot_titles_string = "Performance w Hist. Features:"
        features_kept_string = "features"
        for ft_kept in features_to_keep:
            features_kept_string = features_kept_string + f"-{ft_kept + 1}"
            plot_titles_string = plot_titles_string + f" {ft_kept + 1}"
        # if another run has already started on this feature combo, continue to new combo 
        if missing_set is None:
            if os.path.exists(f"{save_data_dir_top}/{features_kept_string}"):
                continue
        save_data_dir = add_to_dir(save_data_dir_top, features_kept_string)
        features_kept_strings.append(features_kept_string)
        X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_with_only_specified_features(
            dataset_name=dataset_name,
            device=device,
            list_of_features_to_keep=features_to_keep,
        )
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
        test_r2_ = statistics_dict["test_r2"]
        test_mae_ = statistics_dict["test_mae"]
        train_r2_ = statistics_dict["train_r2"]
        train_mae_ = statistics_dict["train_mae"]
        test_r2s.append(test_r2_)
        test_maes.append(test_mae_)
        train_r2s.append(train_r2_)
        train_maes.append(train_mae_)

        dict_log = {
            "test-r2":test_r2_,
        }
        for i in range(n_features_keep):
            dict_log[f"feature{i+1}"] = features_to_keep[i] + 1 
        tracker.log(dict_log)


        if train_preds is not None:
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
    
    # save dt of all data, get best ft to remove next 
    save_df = {
        "features_kept":features_kept_strings,
        "test_r2":test_r2s,
        "test_mae":test_maes,
        "train_r2":train_r2s,
        "train_mae":train_maes,
    }
    save_df = pd.DataFrame.from_dict(save_df)
    save_df = save_df.sort_values(by='test_r2', ascending=False) # sort by top test r2 
    save_df.to_csv(f"{save_data_dir_top}/all_results_{wandb_run_name}.csv", index=False)



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
        default=8,
        required=False,
    )
    parser.add_argument(
        "--n_epochs",
        help="int, num training epochs",
        type=int,
        default=32, 
        required=False,
    )
    parser.add_argument(
        "--n_features_keep",
        help="int, num features to keep at a time",
        type=int,
        default=4, 
        required=False,
    )
    parser.add_argument(
        "--missing_set_id",
        help="int, missing set id",
        type=int,
        default=None, 
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    # 
    wandb_config_dict = {
        "dataset_id":args.dataset_id,
        "dataset_name":dataset_name,
        "n_epochs":args.n_epochs,
        "n_features_keep":args.n_features_keep,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-all-combos",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"all-combos-{args.n_features_keep}-features-exact-gp")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"epochs{args.n_epochs}")
    
    # NOTE: must be feature id (feature num = ft id + 1! )
    missing_sets = [
        [0, 4, 26, 27], 
        [0, 4, 26, 28], 
        [0, 10, 14, 17], 
        [3, 8, 10, 29], 
        [3, 21, 26, 28], 
        [8, 9, 13, 22], 
        [9, 13, 29, 30], 
        [14, 15, 22, 29],
    ]
                                                                              
    if args.missing_set_id is None:
        missing_set = None
    else:
        missing_set = missing_sets[args.missing_set_id]                                                                                                                                              
    main(
        save_data_dir_top=save_data_dir_toptop,
        dataset_name=dataset_name, 
        tracker=tracker,
        n_features_keep=args.n_features_keep,
        n_epochs=args.n_epochs, 
        wandb_run_name=wandb_run_name,
        missing_set=missing_set,
    )
