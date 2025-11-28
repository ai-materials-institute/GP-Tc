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
    tracker,
    save_data_dir,
    dataset_name, 
    n_epochs=32, 
    n_batches_emd_kernel=10,
    only_test_on_special=False,
    balanced_train_set=False,
    n_ind_pts=1024,
    bsz=32,
    lr=0.1, 
):
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_class_data_10_02_2025(
        dataset_name=dataset_name,
        device=device,
        only_test_on_special=only_test_on_special,
        balanced_train_set=balanced_train_set,
        list_of_features_to_remove=[],
    ) 
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
    tracker.log(statistics_dict)
    tracker.finish()



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
        "--only_test_on_special",
        help="bool, only test on MgB2 and graphite",
        type=str2bool,
        default=False, 
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
        "only_test_on_special":args.only_test_on_special,
        "balanced_train_set":args.balanced_train_set,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-class-gp-10-02-25",
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "class-gp-10-02-25")

    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    if args.balanced_train_set:
        save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "balanced_train_set")
    if args.only_test_on_special:
        save_data_dir_toptop = add_to_dir(save_data_dir_toptop, "test_set_is_MgB2_and_graphite_only")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"epochs{args.n_epochs}_{wandb_run_name}")
    

    # run to find best feature to remove next 
    main(
        save_data_dir=save_data_dir_toptop,
        dataset_name=dataset_name,
        n_epochs=args.n_epochs,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        n_ind_pts=args.n_ind_pts,
        bsz=args.bsz,
        lr=args.lr, 
        tracker=tracker,
        only_test_on_special=args.only_test_on_special,
        balanced_train_set=args.balanced_train_set,
    )

# python3 classgp_10_02_2025.py --only_test_on_special False --balanced_train_set False 

