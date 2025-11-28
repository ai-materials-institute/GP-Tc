import sys 
sys.path.append("../")
from data_loading_scripts.load_data import load_data
from constants import DATA_ID_TO_NAME
import torch 
import copy 
import gpytorch
import argparse 
from models.emd_exact_gp import EMDExactGPModel
from utils.get_model_preds import get_model_predictions
from utils.get_performance_stats import get_performance_stats
from utils.create_wandb_tracker import create_tracker
from utils.str2bool_for_argparse import str2bool
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DATA_DIRECTORY = "../save_model_data"

# path = ../data/datasetset_name.pdkl

def main(
    dataset_id, # for wandb tracking 
    dataset_name, 
    n_epochs=50, 
    lr=0.01, 
    map_saas_tau=None,
    add_saas_ls_prior=False,
    remove_ord1_features=False,
    n_batches_emd_kernel=None,
):
    model = "exact-gp" # for wandb config dict 
    wandb_config_dict = {k: v for k, v in locals().items()}
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
    )
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data(
        dataset_name=dataset_name,
        device=device,
        remove_ord1_features=remove_ord1_features,
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lr_str = str(lr).replace(".", "p") 
    model_id_string = copy.deepcopy(model)
    if add_saas_ls_prior:
        model_id_string = model_id_string + "-map-saas"
        if map_saas_tau is not None:
            model_id_string = model_id_string + f"-tau{map_saas_tau}"
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model = EMDExactGPModel(
        train_x=X_train, 
        train_y=y_train, 
        likelihood=likelihood, 
        data_shape=data_shape, 
        n_histogram=n_histogram,
        map_saas_tau=map_saas_tau,
        add_saas_ls_prior=add_saas_ls_prior,
        n_batches_emd_kernel=n_batches_emd_kernel,
    ).to(device=device)

    # train model 
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for e in range(n_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train) 
        # Calc loss and backprop gradients
        loss = -mll(output, y_train) 
        loss.backward()
        
        dict_log = {
            "epoch":e + 1,
            "loss":loss.item(),
        }
        tracker.log(dict_log)
        optimizer.step()

    train_preds, test_preds = get_model_predictions(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
    ) # outputs numpy arrays 
    statistics_dict = get_performance_stats(
        train_preds=train_preds,
        y_train=y_train.cpu().numpy(),
        test_preds=test_preds,
        y_test=y_test.cpu().numpy(),
    )
    tracker.log(statistics_dict)

    save_data_dir = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir = add_to_dir(save_data_dir, model_id_string)
    save_data_dir = add_to_dir(save_data_dir, dataset_name)
    save_data_dir = add_to_dir(save_data_dir, f"epochs{n_epochs}_lr{lr_str}_{wandb_run_name}")

    if train_preds is not None:
        np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
    learned_ls = model.covar_module.lengthscales.detach().cpu().numpy()
    np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", learned_ls)
    rounded_ls = [round(value, 2) for value in learned_ls]
    unique_ls = np.unique(np.array( rounded_ls))
    n_unique_ls = unique_ls.shape[0]
    # model.covar_module.weights 
    print("learned_ls", learned_ls, "N unique:", n_unique_ls)
    
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
        default=1,
        required=False,
    )
    parser.add_argument(
        "--n_epochs",
        help="int, num training epochs",
        type=int,
        default=4, 
        required=False,
    )
    parser.add_argument(
        "--lr",
        help="float, learning rate",
        type=float,
        default=0.1, 
        required=False,
    )
    parser.add_argument(
        "--add_saas_ls_prior",
        help="bool, if true add saasbo stype HalfCatchy LS prior",
        type=str2bool,
        default=True, 
        required=False,
    ) 
    parser.add_argument(
        "--remove_ord1_features",
        help="bool, if remove first 10 histogram features (order 1 features)",
        type=str2bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--n_batches_emd_kernel",
        help="int, n_batches_emd_kernel",
        type=int,
        default=None, 
        required=False,
    )
    parser.add_argument(
        "--map_saas_tau",
        help="float, value for tau param for map saas, if None, learned from data",
        type=float,
        default=None,
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    main(
        dataset_id=args.dataset_id,
        dataset_name=dataset_name,
        n_epochs=args.n_epochs,
        lr=args.lr,
        add_saas_ls_prior=args.add_saas_ls_prior,
        map_saas_tau=args.map_saas_tau,
        remove_ord1_features=args.remove_ord1_features,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
    )


# python3 train_exact_gp.py --dataset_id 1 --n_epochs 4
# python3 train_exact_gp.py --dataset_id 2 --n_epochs 4
# python3 train_exact_gp.py --dataset_id 3 --n_epochs 4
# python3 train_exact_gp.py --dataset_id 4 --n_epochs 4
# python3 train_exact_gp.py --dataset_id 6 --n_epochs 4
# python3 train_exact_gp.py --dataset_id 7 --n_epochs 4
