import sys 
sys.path.append("../")
from data_loading_scripts.load_10_02_2025_class_data import load_class_data_10_02_2025
from constants import DATA_ID_TO_NAME
from utils.str2bool_for_argparse import str2bool
import torch 
import gc 
import copy 
import pandas as pd 
import json 
import gpytorch
import argparse 
from torch.utils.data import TensorDataset, DataLoader
from models.class_emd_gp_w_sg import EmdSgGpClassificationModel
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
    n_ind_pts=1024,
    bsz=32,
    lr=0.1, 
    balanced_train_set=False,
):
    X_train, X_test, y_train, y_test, data_shape, n_histogram_full, n_sg_full = load_class_data_10_02_2025(
        dataset_name=dataset_name,
        device=device,
        list_of_features_to_remove=[],
        only_test_on_special=False,
        balanced_train_set=balanced_train_set,
    ) 

    # count num not in always remove list 
    
    n_hist_features_remaining = 0
    n_sg_features_remaining = 0 
    for hist_idx in range(n_histogram_full):
        if not (hist_idx in always_remove_features_list):
            n_hist_features_remaining += 1
    for symm_id_zeroed in range(n_sg_full):
        symm_id = symm_id_zeroed + n_histogram_full
        if not (symm_id in always_remove_features_list):
            n_sg_features_remaining += 1
    n_features_remaining = n_hist_features_remaining + n_sg_features_remaining
    if n_features_remaining == 0:
        assert 0, "always_remove_features_list is now all features, DONE"

    test_accs = [] 
    feature_removed = []
    train_accs = []
    total_num_features = n_histogram_full + n_sg_full
    for rm_new_ft_id in range(-1, total_num_features):
        if rm_new_ft_id in always_remove_features_list:
            continue 
        if rm_new_ft_id == -1:
            n_histogram = n_histogram_full
            n_sg = n_sg_full
            feature_removed.append(-1) # None 
        else:       
            X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_class_data_10_02_2025(
                dataset_name=dataset_name,
                device=device,
                list_of_features_to_remove=[rm_new_ft_id] + always_remove_features_list,
                only_test_on_special=False,
                balanced_train_set=balanced_train_set,
            )
            feature_removed.append(rm_new_ft_id +1)
        # initialize likelihood and model
        model = EmdSgGpClassificationModel(
            train_x=X_train[0:n_ind_pts], 
            n_histogram=n_histogram,
            n_sg=n_sg,
            data_shape=data_shape, 
            n_batches_emd_kernel=n_batches_emd_kernel,
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
        
        if rm_new_ft_id == -1:
            all_features_test_acc = test_acc_
            save_data_dir = add_to_dir(save_data_dir_top, f"all_features")
        else:
            save_data_dir = add_to_dir(save_data_dir_top, f"removed_feature_{rm_new_ft_id + 1}")

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

    return best_feature_to_remove_next, best_test_acc_, n_histogram_full, n_sg_full, all_features_test_acc, n_hist_features_remaining, n_sg_features_remaining


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
        default=20,
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
        "--n_ind_pts",
        help="int, n_ind_pts",
        type=int,
        default=1024, 
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
    parser.add_argument(
        "--balanced_train_set",
        help="bool, 50/50 sc vs non-sc in train set",
        type=str2bool,
        default=False, 
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
        "n_ind_pts":args.n_ind_pts,
        "bsz":args.bsz,
        "lr":args.lr,
        "balanced_train_set":args.balanced_train_set,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-ft-removal",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "removing-features-class-gp-10-02-25")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    if args.balanced_train_set:
        save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "balanced_train_set")
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
        best_feature_to_remove_next, best_test_acc_, full_n_hist, full_n_sg, all_features_test_acc, nh_left, nsg_left = main(
            save_data_dir_top=save_data_dir_loop,
            dataset_name=dataset_name,
            n_epochs=args.n_epochs,
            n_batches_emd_kernel=args.n_batches_emd_kernel,
            always_remove_features_list=always_remove_features_list,
            n_ind_pts=args.n_ind_pts,
            bsz=args.bsz,
            lr=args.lr, 
            balanced_train_set=args.balanced_train_set,
        )
        full_n_features = full_n_hist + full_n_sg
        # best_feature_to_remove_next is hist_id + 1, subtract to get actual index
        ft_index_to_remove_next = best_feature_to_remove_next - 1
        # add feature w/ biggest test acc when removed to always remove list 
        always_remove_features_list.append(ft_index_to_remove_next)
        if ft_index_to_remove_next < full_n_hist: # if feature to remove next is hist feature 
            nh_left -= 1
        else:
            nsg_left -= 1
        # empty garbage, clear cache
        gc.collect()
        torch.cuda.empty_cache()
        if step == 0:
            # log remove no features 
            dict_log = {
                "latest-ftid-removed":-1,
                "best-test-r2":all_features_test_acc,
                "n-fts-removed":0,
                "n-fts-kept":full_n_features, # full_n_hist,
                "n-hist-kept":full_n_hist,
                "n-sg-kept":full_n_sg,
            }
            tracker.log(dict_log)
        # wandb tracking 
        total_num_removed = len(always_remove_features_list)
        dict_log = {
            "latest-ftid-removed":best_feature_to_remove_next,
            "best-test-r2":best_test_acc_,
            "n-fts-removed":total_num_removed,
            "n-fts-kept":full_n_features - total_num_removed,
            "n-hist-kept":nh_left,
            "n-sg-kept":nsg_left,
        }
        tracker.log(dict_log)
        step += 1 

# python3 manually_rm_ftrs_classgp_10_02_2025.py ----balanced_train_set False --n_epochs 2

