import sys 
sys.path.append("../")
from data_loading_scripts.classgp_load_data_only_specified_features import load_class_data_only_specified_features
from constants import DATA_ID_TO_NAME
import matplotlib.pyplot as plt 
from utils.str2bool_for_argparse import str2bool
import torch 
from itertools import combinations
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
    n_epochs=32, 
    n_batches_emd_kernel=None,
    order2_features_only=True,
    n_ind_pts=1024,
    add_saas_ls_prior=False,
    map_saas_tau=None,
    bsz=32,
    lr=0.1, 
    n_features_keep=4,
):
    X_train, X_test, y_train, y_test, data_shape, n_histogram_full = load_class_data_only_specified_features(
        dataset_name=dataset_name,
        device=device, 
        list_of_features_to_keep="all",
        order2_features_only=order2_features_only,
    )
    
    # all combination of n_features_keep that we could select 
    all_combos = torch.tensor(list(combinations(range(n_histogram_full), n_features_keep))) # torch.Size([35960, 4]) for 32, 4 
    # randomly shuffle combos to prevent runs from overlapping (write error)
    num_rows = all_combos.size(0)
    random_indices = torch.randperm(num_rows)
    all_combos = all_combos[random_indices]


    test_accs = [] 
    train_accs = []
    features_kept_strings = []
    for features_to_keep in all_combos: 
        features_kept_string = "features"
        for ft_kept in features_to_keep:
            features_kept_string = features_kept_string + f"-{ft_kept + 1}"
        # don't repeat computation already done by another run 
        if os.path.exists(f"{save_data_dir_top}/{features_kept_string}"):
            continue
        save_data_dir = add_to_dir(save_data_dir_top, features_kept_string)
        features_kept_strings.append(features_kept_string)

        X_train, X_test, y_train, y_test, data_shape, n_histogram = load_class_data_only_specified_features(
            dataset_name=dataset_name,
            device=device, 
            list_of_features_to_keep=features_to_keep,
            order2_features_only=order2_features_only,
        )
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

        dict_log = {
            "test-acc":test_acc_,
        }
        for i in range(n_features_keep):
            dict_log[f"feature{i+1}"] = features_to_keep[i] + 1 
        tracker.log(dict_log)

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
    
        # save dt of all data, get best ft to remove next 
    save_df = {
        "features_kept":features_kept_strings,
        "test_acc":test_accs,
        "train_acc":train_accs,
    }
    save_df = pd.DataFrame.from_dict(save_df)
    save_df = save_df.sort_values(by='test_acc', ascending=False) # sort by top test acc
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
        default=16,
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
    parser.add_argument(
        "--n_features_keep",
        help=" num features to keep",
        type=int,
        default=4, 
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
        "n_features_keep":args.n_features_keep,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-class-gp-allcombos",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"all-combos-{args.n_features_keep}-features-class-gp")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"epochs{args.n_epochs}")

    main(
        save_data_dir_top=save_data_dir_toptop,
        dataset_name=dataset_name,
        n_epochs=args.n_epochs, 
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        order2_features_only=args.order2_features_only,
        n_ind_pts=args.n_ind_pts,
        add_saas_ls_prior=args.add_saas_ls_prior,
        map_saas_tau=None,
        bsz=args.bsz,
        lr=args.lr, 
        n_features_keep=args.n_features_keep,
    )
