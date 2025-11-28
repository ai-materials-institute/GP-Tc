import sys 
sys.path.append("../")
from torch.utils.data import TensorDataset, DataLoader
from data_loading_scripts.load_06_24_2025_class_data import load_class_data_06_24_2025
from constants import DATA_ID_TO_NAME
import torch 
import copy 
import gpytorch
import argparse 
from models.classification_emd_gp import EmdGpClassificationModel
from utils.create_wandb_tracker import create_tracker
from utils.str2bool_for_argparse import str2bool
from utils.classification_pred_utils import get_class_model_preds
from utils.get_acc_stats_classification import get_performance_stats_classification
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DATA_DIRECTORY = "../save_model_data"

# path = ../data/datasetset_name.pdkl

# https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html
def main(
    dataset_id, # for wandb log 
    dataset_name, 
    n_epochs=50, 
    lr=0.01, 
    n_ind_pts=1024,
    bsz=1024,
    add_saas_ls_prior=False,
    map_saas_tau=None,
    n_batches_emd_kernel=None,
    clear_cache=False,
    only_test_on_special=False,
):
    model = "class-gp" # for wandb config dict 
    wandb_config_dict = {k: v for k, v in locals().items()}
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
    )
    assert dataset_id == 17 
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_class_data_06_24_2025(
        dataset_name=dataset_name,
        device=device,
        only_test_on_special=only_test_on_special,
    )

    # log stats on num sc vs non-sc  
    num_train_nonsc = (y_train == 0.0).sum().item()
    num_train_sc = (y_train == 1.0).sum().item()
    assert y_train.shape[0] == num_train_nonsc + num_train_sc
    num_test_nonsc = (y_test == 0.0).sum().item()
    num_test_sc = (y_test == 1.0).sum().item()
    assert y_test.shape[0] == num_test_nonsc + num_test_sc
    perc_train_sc = num_train_sc/(num_train_sc + num_train_nonsc)
    perc_test_sc = num_test_sc/(num_test_sc + num_test_nonsc)
    dict_log = {
        "perc_train_sc":perc_train_sc,
        "num_train_nonsc":num_train_nonsc,
        "num_train_sc":num_train_sc,
        "num_train":num_train_nonsc + num_train_sc,
        "perc_test_sc":perc_test_sc,
        "num_test_nonsc":num_test_nonsc,
        "num_test_sc":num_test_sc,
        "num_test":num_test_sc + num_test_nonsc,
    }
    tracker.log(dict_log)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lr_str = str(lr).replace(".", "p") 
    model_id_string = copy.deepcopy(model)
    if add_saas_ls_prior:
        model_id_string = model_id_string + "-map-saas"

    # Initialize model and likelihood
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y_train.numel())
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    for e in range(n_epochs):
        total_loss = 0
        for (x_batch, y_batch) in train_loader:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_batch) 
            # Calc loss and backprop gradients
            loss = -mll(output, y_batch) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if clear_cache:
                model._clear_cache()
                model.covar_module._clear_cache()
                torch.cuda.empty_cache()
        dict_log = {
            "epoch":e + 1,
            "loss":total_loss,
        }
        tracker.log(dict_log)

    if clear_cache:
        del train_dataset
        del train_loader
        del x_batch
        del y_batch 
        del optimizer 
        del total_loss
        del mll
        model._clear_cache()
        model.covar_module._clear_cache()
        torch.cuda.empty_cache()
    # get test and train preds 
    train_preds, test_preds = get_class_model_preds(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
        bsz=bsz,
    ) # outputs numpy arrays of classification preds 
    statistics_dict = get_performance_stats_classification(
        y_test=y_test.cpu().numpy(),
        y_train=y_train.cpu().numpy(),
        test_preds=test_preds,
        train_preds=train_preds,
    )
    tracker.log(statistics_dict)

    # save data 
    save_data_dir = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir = add_to_dir(save_data_dir, model_id_string)
    save_data_dir = add_to_dir(save_data_dir, dataset_name)
    if only_test_on_special:
        save_data_dir = add_to_dir(save_data_dir, "holdout_MgB2_graphite_only")
    else:
        save_data_dir = add_to_dir(save_data_dir, "holdout_full_test_set_including_MgB2_graphite")
    save_data_dir = add_to_dir(save_data_dir, f"epochs{n_epochs}_lr{lr_str}_{wandb_run_name}")

    if train_preds is not None:
        np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
    learned_ls = model.covar_module.lengthscales.detach().cpu().numpy()
    np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", learned_ls)
    tracker.finish()
    if only_test_on_special:
        print("Only held out Two special points for testing")
    true_print = y_test.cpu().numpy().squeeze().tolist()[-2:]
    pred_print = test_preds.squeeze().tolist()[-2:]
    print("Special Two Point Predictions (MgB2 and graphite):")
    print("True:", true_print, "Preds:", pred_print)


def add_to_dir(dir_path, add_this):
    dir_path = dir_path + f"/{add_this}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        help="int id for dataset to train on",
        type=int,
        default=17,
        required=False,
    )
    parser.add_argument(
        "--n_epochs",
        help="int, num training epochs",
        type=int,
        default=2, 
        required=False,
    )
    parser.add_argument(
        "--n_ind_pts",
        help="int, num inducing points",
        type=int,
        default=1024, 
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
        "--bsz",
        help="int, train batch size",
        type=int,
        default=2048,
        required=False,
    )
    parser.add_argument(
        "--add_saas_ls_prior",
        help="bool, if true add saasbo stype HalfCatchy LS prior",
        type=str2bool,
        default=False, 
        required=False,
    )
    parser.add_argument(
        "--n_batches_emd_kernel",
        help="int, if None, no batching, if int given, split kernel comp into this many batches to avoid OOM",
        type=int,
        default=None, # None, None --> no batching, 2 --> feasibly avoid GPU OOM on largest 48G GPUs we have 
        required=False,
    )
    parser.add_argument(
        "--clear_cache",
        help=" bool, if true clear model cache after each training batch",
        type=str2bool,
        default=False, 
        required=False,
    )
    parser.add_argument(
        "--map_saas_tau",
        help="float, value for tau param for map saas, if None, learned from data",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--only_test_on_special",
        help=" bool, if true, only hold out special last two points as test set",
        type=str2bool,
        default=False,
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    assert dataset_name == "classification_data_whole_20250528"
    main(
        dataset_id=args.dataset_id, # for wandb log 
        dataset_name=dataset_name,
        n_epochs=args.n_epochs,
        lr=args.lr,
        add_saas_ls_prior=args.add_saas_ls_prior,
        map_saas_tau=args.map_saas_tau,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        n_ind_pts=args.n_ind_pts,
        bsz=args.bsz,
        clear_cache=args.clear_cache,
        only_test_on_special=args.only_test_on_special,
    )

# python3 train_class_gp.py --n_epochs 2 --n_ind_pts 1024 --lr 0.1 --bsz 2048 --add_saas_ls_prior False --n_batches_emd_kernel None --clear_cache False 