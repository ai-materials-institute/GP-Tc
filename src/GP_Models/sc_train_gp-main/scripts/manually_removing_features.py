import sys 
sys.path.append("../")
from data_loading_scripts.load_data_remove_specific_hist_features import load_data_without_specified_features
from constants import DATA_ID_TO_NAME
import matplotlib.pyplot as plt 
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

# path = ../data/datasetset_name.pdkl

def main(
    save_data_dir_top,
    dataset_name, 
    n_epochs=50, 
    n_batches_emd_kernel=None,
    always_remove_features_list=[],
):
    lr=0.1 
    also_remove_ord1_features=False
    symm_features_only=True # last 11 features 
    X_train, X_test, y_train, y_test, data_shape, n_histogram_full = load_data_without_specified_features(
        dataset_name=dataset_name,
        device=device,
        list_of_features_to_remove=[],
        also_remove_ord1_features=also_remove_ord1_features,
        symm_features_only=symm_features_only,
    )

    # count num not in always remove list
    n_features_remaining = 0
    for hist_id in range(n_histogram_full):
        if not (hist_id in always_remove_features_list):
            n_features_remaining += 1
    if n_features_remaining == 0:
        assert 0, "always_remove_features_list is now all features, DONE"
    
    # Or maybe would be easier with new data loading script that loads without specific feature ids 
    #   takes list of hist feautre ids to remove! Perfect yes 
    test_r2s = []
    test_maes = []
    feature_removed = []
    train_r2s = []
    train_maes = []
    for hist_id in range(-1, n_histogram_full):
        if hist_id in always_remove_features_list:
            continue 
        if hist_id == -1:
            n_histogram = n_histogram_full
            feature_removed.append(-1)
        else:
            X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_without_specified_features(
                dataset_name, 
                device, 
                list_of_features_to_remove=[hist_id] + always_remove_features_list,
                also_remove_ord1_features=also_remove_ord1_features,
                symm_features_only=symm_features_only,
            )
            feature_removed.append(hist_id +1)
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
        
        if hist_id == -1:
            all_features_test_r2 = test_r2_
            save_data_dir = add_to_dir(save_data_dir_top, f"all_features")
        else:
            save_data_dir = add_to_dir(save_data_dir_top, f"removed_feature_{hist_id + 1}")

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
        if hist_id == -1:
            title_string = f"Test performance with All Features"
        else:
            title_string = f"Test performance with Histogram Features Removed: {hist_id + 1}"
            for ft_i in always_remove_features_list:
                title_string = title_string + f" ,{ft_i + 1}"
        title_string = title_string + f"\n R2:{test_r2_:.3f}, MAE:{test_mae_:.3f}"
        plt.title(title_string)
        plt.savefig(f"{save_data_dir}/test_result.png")
        plt.clf()

        # train 
        plt.figure(figsize=(10, 8))
        plt.scatter(y_train, train_preds)
        plt.xlabel("True Train Y Value")
        plt.ylabel("Exact GP Model's Predicted Value")
        if hist_id == -1:
            title_string = f"Train performance with All Features"
        else:
            title_string = f"Train performance with Histogram Features Removed: {hist_id + 1}"
            for ft_i in always_remove_features_list:
                title_string = title_string + f" ,{ft_i + 1}"
        title_string = title_string + f"\nR2:{train_r2_:.3f}, MAE:{train_mae_:.3f}"
        plt.title(title_string)
        plt.savefig(f"{save_data_dir}/train_result.png")
        plt.clf()

        # ls 
        learned_ls = learned_ls.tolist()
        if hist_id != -1:
            temp = []
            counter = 0
            # fill in missing hist_id with 0 
            for feature_id in range(n_histogram_full):
                if feature_id == hist_id:
                    temp.append(0.0)
                elif feature_id in always_remove_features_list:
                    temp.append(0.0)
                else:
                    ls_val = learned_ls[counter]
                    temp.append(ls_val)
                    counter += 1
            learned_ls = temp 
        categories = [f"{i+1}" for i in range(len(learned_ls))]
        fig_width = len(learned_ls)//3 
        plt.figure(figsize=(fig_width, 4))
        plt.bar(categories, learned_ls, color='skyblue')
        if hist_id == -1:
            title_string = f"Histogram Learned Lengthscales with All Features"
        else:
            title_string = f"Histogram Learned Lengthscales with Histogram Features Removed: {hist_id + 1}"
            for ft_i in always_remove_features_list:
                title_string = title_string + f" ,{ft_i + 1}"
        plt.xlabel('Histogram Feature')
        plt.ylabel('Lengthscale Value')
        plt.title(title_string)
        plt.savefig(f"{save_data_dir}/emd_learned_lengthscales.png")
        plt.clf()
        # empty garbage, clear cache
        gc.collect()
        torch.cuda.empty_cache()
    
    # save dt of all data, get best ft to remove next 
    save_df = {
        "latest_feature_removed":feature_removed,
        "test_r2":test_r2s,
        "test_mae":test_maes,
        "train_r2":train_r2s,
        "train_mae":train_maes,
    }
    save_df = pd.DataFrame.from_dict(save_df)
    save_df = save_df.sort_values(by='test_r2', ascending=False) # sort by top test r2 
    # next we should remove the feature where we got highest test r2 without it: 
    best_feature_to_remove_next = save_df["latest_feature_removed"].values[0].item()
    best_test_r2_ = save_df["test_r2"].values[0].item()
    # but make sure this isn't -1 (indicating all features dept)
    if best_feature_to_remove_next == -1:
        best_feature_to_remove_next = save_df["latest_feature_removed"].values[1].item()
        best_test_r2_ = save_df["test_r2"].values[1].item()
    print(f"\nBest Feature to Remove Next: {best_feature_to_remove_next}, Acheives Best Test R2: {best_test_r2_}")
    assert type(best_feature_to_remove_next) == int
    save_df.to_csv(f"{save_data_dir_top}/all_results.csv", index=False)
    
    # test r2 by feature removed 
    categories = [f"{ft_removed}" for ft_removed in feature_removed]
    categories[0] = "None"
    fig_width = len(feature_removed)//2 
    plt.figure(figsize=(fig_width, 4))
    plt.bar(categories, test_r2s, color='skyblue')
    title_string = f"Test R2 Acheived with Each Histogram Feature Removed"
    title_string = title_string + "\n In Addition to Already Removed Features:"
    for ft_i in always_remove_features_list:
        title_string = title_string + f"{ft_i + 1},"
    plt.xlabel('Additional Histogram Feature Removed')
    plt.ylabel('Test R2')
    plt.title(title_string)
    plt.savefig(f"{save_data_dir_top}/test_r2_w_each_ft_removed.png")
    plt.clf()

    return best_feature_to_remove_next, best_test_r2_, n_histogram_full, all_features_test_r2


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
        "--n_batches_emd_kernel",
        help="int, n_batches_emd_kernel",
        type=int,
        default=10, 
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    # 
    wandb_config_dict = {
        "dataset_id":args.dataset_id,
        "dataset_name":dataset_name,
        "n_batches_emd_kernel":args.n_batches_emd_kernel,
        "n_epochs":args.n_epochs,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-ft-removal",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "removing-features-exact-gp-symm-only")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"epochs{args.n_epochs}_{wandb_run_name}")
    always_remove_features_list = []
    step = 0 
    while True: 
        # create sub dir indiciating features always removed: 
        always_removing_string = "fts-awys-rmvd"
        if len(always_remove_features_list) == 0:
            always_removing_string = always_removing_string + f"-none"
        for ft_id in always_remove_features_list:
            always_removing_string = always_removing_string + f"-{ft_id + 1}"
        save_data_dir_loop = add_to_dir(save_data_dir_toptop, always_removing_string)
        # run to find best feature to remove next 
        best_feature_to_remove_next, best_test_r2_, full_n_hist, all_features_test_r2 = main(
            dataset_name=dataset_name,
            save_data_dir_top=save_data_dir_loop,
            n_epochs=args.n_epochs,
            n_batches_emd_kernel=args.n_batches_emd_kernel,
            always_remove_features_list=always_remove_features_list,
        )
        # best_feature_to_remove_next is hist_id + 1, subtract to get actual index
        ft_index_to_remove_next = best_feature_to_remove_next - 1
        # add feature w/ biggest test r2 when removed to always remove list 
        always_remove_features_list.append(ft_index_to_remove_next)
        # empty garbage, clear cache
        gc.collect()
        torch.cuda.empty_cache()
        if step == 0:
            # log remove no features 
            dict_log = {
                "latest-ftid-removed":-1,
                "best-test-r2":all_features_test_r2,
                "n-fts-removed":0,
                "n-fts-kept":full_n_hist,
            }
            tracker.log(dict_log)
        # wandb tracking 
        total_num_removed = len(always_remove_features_list)
        dict_log = {
            "latest-ftid-removed":best_feature_to_remove_next,
            "best-test-r2":best_test_r2_,
            "n-fts-removed":total_num_removed,
            "n-fts-kept":full_n_hist - total_num_removed,
        }
        tracker.log(dict_log)
        step += 1 


# Now do full scale for a single set num epochs, i.e. 
# python3 manually_removing_features.py --dataset_id 1 --n_epochs 2 



# Prev one at a time: 
    # Update here once deicions made about removing features 
    # Removing FT 10 consistently best for initial one ft removal --> ft id 9: 
    # always_remove_features_list = [9]
    # Removing FT 20 consistently best w/ 10 also gone: 
    # always_remove_features_list = [9, 19]
    # Removing FT 1 consistently best w/ 10,20 also gone: 
    # always_remove_features_list = [0, 9, 19] 
    # Removing FT 13 consistently best w/ 1,10,20 also gone: 
    # always_remove_features_list = [0, 9, 12, 19] 
