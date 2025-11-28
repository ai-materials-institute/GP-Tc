import sys 
sys.path.append("../")
from data_loading_scripts.classgp_load_data_without_specified_features import load_class_data_remove_specified_features
from constants import DATA_ID_TO_NAME
import matplotlib.pyplot as plt 
from utils.str2bool_for_argparse import str2bool
import torch 
import gc 
import copy 
import pandas as pd 
import json 
import gpytorch
import argparse 
from torch.utils.data import TensorDataset, DataLoader
from models.classification_emd_gp import EmdGpClassificationModel 
from utils.create_wandb_tracker import create_tracker
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 
from utils.classification_pred_utils import get_class_model_preds
from utils.get_acc_stats_classification import get_performance_stats_classification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DATA_DIRECTORY = "../save_model_data"

# path = ../data/datasetset_name.pdkl

def main(
    save_data_dir_top,
    dataset_name, 
    n_epochs=50, 
    n_batches_emd_kernel=None,
    always_remove_features_list=[],
    order2_features_only=True,
    n_ind_pts=1024,
    add_saas_ls_prior=False,
    map_saas_tau=None,
    bsz=32,
    lr=0.1, 
):
    X_train, X_test, y_train, y_test, data_shape, n_histogram_full = load_class_data_remove_specified_features(
        dataset_name=dataset_name,
        device=device,
        list_of_features_to_remove=[],
        order2_features_only=order2_features_only,
    ) 
    # count num not in always remove list 
    n_features_remaining = 0
    for hist_id in range(n_histogram_full):
        if not (hist_id in always_remove_features_list):
            n_features_remaining += 1
    if n_features_remaining == 0:
        assert 0, "always_remove_features_list is now all features, DONE"
    

    test_accs = [] 
    feature_removed = []
    train_accs = []
    for hist_id in range(-1, n_histogram_full):
        if hist_id in always_remove_features_list:
            continue 
        if hist_id == -1:
            n_histogram = n_histogram_full
            feature_removed.append(-1)
        else:
            X_train, X_test, y_train, y_test, data_shape, n_histogram = load_class_data_remove_specified_features(
                dataset_name, 
                device, 
                list_of_features_to_remove=[hist_id] + always_remove_features_list,
                order2_features_only=order2_features_only,
            )
            feature_removed.append(hist_id +1)
        # initialize likelihood and model
        model = EmdGpClassificationModel(
            train_x=X_train[0:n_ind_pts], 
            n_histogram=n_histogram,
            data_shape=data_shape, 
            n_batches_emd_kernel=n_batches_emd_kernel,
            map_saas_tau=map_saas_tau,
            add_saas_ls_prior=add_saas_ls_prior,
        ).to(device=device)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device=device)
        # train model 
        model.train()
        likelihood.train()
        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)

        for e in range(n_epochs):
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
            bsz=bsz,
        ) # outputs numpy arrays of classification preds 
        y_test = y_test.cpu().numpy()
        y_train = y_train.cpu().numpy()
        statistics_dict = get_performance_stats_classification(
            y_test=y_test,
            y_train=y_train,
            test_preds=test_preds,
            train_preds=train_preds,
        )
        test_acc_ = statistics_dict["test_acc"]
        train_acc_ = statistics_dict["train_acc"]
        test_accs.append(test_acc_)
        train_accs.append(train_acc_)
        
        if hist_id == -1:
            all_features_test_acc = test_acc_
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
        with open(f"{save_data_dir}/performance.json", 'w') as file:
            json.dump(statistics_dict, file, indent=4) 

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
    
    # save df of all data, get best ft to remove next 
    save_df = {
        "latest_feature_removed":feature_removed,
        "test_acc":test_accs,
        "train_acc":train_accs,
    }
    save_df = pd.DataFrame.from_dict(save_df)
    save_df = save_df.sort_values(by='test_acc', ascending=False) # sort by top test acc 
    # 
    # next we should remove the feature where we got highest test acc without it: 
    best_feature_to_remove_next = save_df["latest_feature_removed"].values[0].item()
    best_test_acc_ = save_df["test_acc"].values[0].item()
    # but make sure this isn't -1 (indicating all features dept)
    if best_feature_to_remove_next == -1:
        best_feature_to_remove_next = save_df["latest_feature_removed"].values[1].item()
        best_test_acc_ = save_df["test_acc"].values[1].item()
    print(f"\nBest Feature to Remove Next: {best_feature_to_remove_next}, Acheives Best Test ACC: {best_test_acc_}")
    assert type(best_feature_to_remove_next) == int
    save_df.to_csv(f"{save_data_dir_top}/all_results.csv", index=False)
    
    # test r2 by feature removed 
    categories = [f"{ft_removed}" for ft_removed in feature_removed]
    categories[0] = "None"
    fig_width = len(feature_removed)//2 
    plt.figure(figsize=(fig_width, 4))
    plt.bar(categories, test_accs, color='skyblue')
    title_string = f"Test ACC Acheived with Each Histogram Feature Removed"
    title_string = title_string + "\n In Addition to Already Removed Features:"
    for ft_i in always_remove_features_list:
        title_string = title_string + f"{ft_i + 1},"
    plt.xlabel('Additional Histogram Feature Removed')
    plt.ylabel('Test R2')
    plt.title(title_string)
    plt.savefig(f"{save_data_dir_top}/test_acc_w_each_ft_removed.png")
    plt.clf()

    return best_feature_to_remove_next, best_test_acc_, n_histogram_full, all_features_test_acc


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
        default=13,
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
    parser.add_argument(
        "--order2_features_only",
        help="bool, order2_features_only",
        type=str2bool,
        default=True, 
        required=False,
    )
    parser.add_argument(
        "--n_ind_pts",
        help="int, n_ind_pts",
        type=int,
        default=1024, 
        required=False,
    )
    parser.add_argument(
        "--add_saas_ls_prior",
        help="bool, add_saas_ls_prior",
        type=str2bool,
        default=False, 
        required=False,
    )
    parser.add_argument(
        "--bsz",
        help="int, bsz",
        type=int,
        default=1024, 
        required=False,
    )
    parser.add_argument(
        "--lr",
        help="int, lr",
        type=float,
        default=0.05, 
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
        "order2_features_only":args.order2_features_only,
        "n_ind_pts":args.n_ind_pts,
        "add_saas_ls_prior":args.add_saas_ls_prior,
        "bsz":args.bsz,
        "lr":args.lr,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-ft-removal",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    # order2_features_only
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "removing-features-class-gp")
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
        best_feature_to_remove_next, best_test_acc_, full_n_hist, all_features_test_acc = main(
            save_data_dir_top=save_data_dir_loop,
            dataset_name=dataset_name,
            n_epochs=args.n_epochs,
            n_batches_emd_kernel=args.n_batches_emd_kernel,
            always_remove_features_list=always_remove_features_list,
            order2_features_only=args.order2_features_only,
            n_ind_pts=args.n_ind_pts,
            add_saas_ls_prior=args.add_saas_ls_prior,
            map_saas_tau=None,
            bsz=args.bsz,
            lr=args.lr, 
        )
        # best_feature_to_remove_next is hist_id + 1, subtract to get actual index
        ft_index_to_remove_next = best_feature_to_remove_next - 1
        # add feature w/ biggest test acc when removed to always remove list 
        always_remove_features_list.append(ft_index_to_remove_next)
        # empty garbage, clear cache
        gc.collect()
        torch.cuda.empty_cache()
        if step == 0:
            # log remove no features 
            dict_log = {
                "latest-ftid-removed":-1,
                "best-test-r2":all_features_test_acc,
                "n-fts-removed":0,
                "n-fts-kept":full_n_hist,
            }
            tracker.log(dict_log)
        # wandb tracking 
        total_num_removed = len(always_remove_features_list)
        dict_log = {
            "latest-ftid-removed":best_feature_to_remove_next,
            "best-test-r2":best_test_acc_,
            "n-fts-removed":total_num_removed,
            "n-fts-kept":full_n_hist - total_num_removed,
        }
        tracker.log(dict_log)
        step += 1 

# python3 class_gp_manually_removing_features.py --n_epochs 32 

