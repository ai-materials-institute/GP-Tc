import torch 
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 
import copy 
import gpytorch
import matplotlib.pyplot as plt 
import argparse 
import gc 
import sys 
sys.path.append("../")
from utils.create_wandb_tracker import create_tracker
from utils.str2bool_for_argparse import str2bool
from data_loading_scripts.load_data_09_26_25 import load_data_09_26_25
from utils.get_performance_stats import get_performance_stats
from utils.get_model_preds import get_model_predictions 
from models.new_sg_featurization_sg_emd_exact_gp import SgEmdExactGPModelV2


SAVE_DATA_DIRECTORY = "../save_model_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(
    hist_features_ord_id, # ord1 or ordall
    include_sg_features, # True/False 
    n_batches_emd_kernel=10,
    n_epochs=32,
    lr=0.1,
):
    wandb_config_dict = {k: v for k, v in locals().items()}
    assert hist_features_ord_id in ["ord1", "ordall"]
    assert include_sg_features in [True, False]
    X_train, X_test, y_train, y_test, data_shape, n_histogram, n_sg = load_data_09_26_25(
        hist_features_ord_id=hist_features_ord_id,
        include_sg_features=include_sg_features,
        device=device, 
    )
    wandb_config_dict["n_histogram"] = n_histogram
    wandb_config_dict["n_sg"] = n_sg
    wandb_config_dict["n_train"] = y_train.shape[0]
    wandb_config_dict["n_test"] = y_test.shape[0]
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name="sc-train-gp-09-26-25",
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model = SgEmdExactGPModelV2(
        train_x=X_train, 
        train_y=y_train, 
        likelihood=likelihood, 
        n_sg=n_sg,
        map_saas_tau=None,
        n_histogram=n_histogram, 
        data_shape=data_shape, 
        n_batches_emd_kernel=n_batches_emd_kernel,
        add_saas_ls_prior=False,
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
        optimizer.step()
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            dict_log = {
                "epoch":e + 1,
                "loss":loss.item(),
            }
            tracker.log(dict_log)
            del dict_log

    model._clear_cache()
    gc.collect()
    torch.cuda.empty_cache()
    model._clear_cache()
    train_preds, test_preds = get_model_predictions(
        model=model, 
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
    ) # outputs numpy arrays 
    y_train = y_train.cpu().numpy()
    y_test = y_test.cpu().numpy()
    statistics_dict = get_performance_stats(
        train_preds=train_preds,
        y_train=y_train,
        test_preds=test_preds,
        y_test=y_test,
    )
    tracker.log(statistics_dict)

    save_data_dir = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir = add_to_dir(save_data_dir, "sc-train-gp-09-26-25")
    if include_sg_features:
        use_sg_str = "_with_sg" 
    else:
        use_sg_str = "_no_sg"
    data_id_string = hist_features_ord_id + use_sg_str
    save_data_dir = add_to_dir(save_data_dir, data_id_string)
    save_data_dir = add_to_dir(save_data_dir, f"epochs{n_epochs}_{wandb_run_name}")

    if train_preds is not None:
        np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    # same emd kernel weights and length scales 
    np.save(f"{save_data_dir}/emd_kernel_weights.npy", model.covar_module.weights.detach().cpu().numpy())
    np.save(f"{save_data_dir}/emd_kernel_lengthscales.npy", model.covar_module.lengthscales.detach().cpu().numpy())
    # directly save sg graph features learned here too 
    if include_sg_features:
        sg_kernel_lengthscale = model.covar_module.sg_kernel.base_kernel.lengthscale.squeeze().detach().cpu().numpy() 
        np.save(f"{save_data_dir}/sg_kernel_lengthscale.npy", np.array(sg_kernel_lengthscale))

    test_r2_ = round(statistics_dict["test_r2"], 3)
    test_mae_ = round(statistics_dict["test_mae"], 3)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, test_preds)
    plt.xlabel("True Test Y Value")
    plt.ylabel("Exact GP Model's Predicted Value")
    plt.title(f"Test R2:{test_r2_}, Test MAE:{test_mae_}")
    plt.savefig(f"{save_data_dir}/test_result.png")
    plt.clf()

    if train_preds is not None:
        train_r2_ = round(statistics_dict["train_r2"], 3)
        train_mae_ = round(statistics_dict["train_mae"], 3)
        plt.figure(figsize=(10, 8))
        plt.scatter(y_train, train_preds)
        plt.xlabel("True Train Y Value")
        plt.ylabel("Exact GP Model's Predicted Value")
        plt.title(f"Train R2:{train_r2_}, Train MAE:{train_mae_}")
        plt.savefig(f"{save_data_dir}/train_result.png")
        plt.clf()

    tracker.finish()



def add_to_dir(dir_path, add_this):
    dir_path = dir_path + f"/{add_this}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path 
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--n_batches_emd_kernel",
        help="int, if None, no batching, if int given, split EMD kernel comp into this many batches to avoid OOM",
        type=int,
        default=10, # None, 
        required=False,
    )
    parser.add_argument(
        "--n_epochs",
        help="int, number of training epochs",
        type=int,
        default=32,
        required=False,
    )
    parser.add_argument(
        "--hist_features_ord_id",
        help="str must be either ord1 or ordall",
        type=str,
        default="ord1",
        required=False,
    )
    parser.add_argument(
        "--include_sg_features",
        help="bool, whether to also use sg features",
        type=str2bool,
        default=False, 
        required=False,
    )
    args = parser.parse_args() 
    main(
        hist_features_ord_id=args.hist_features_ord_id,
        include_sg_features=args.include_sg_features,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        n_epochs=args.n_epochs,
        lr=0.1,
    )

# python3 train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features False 
# python3 train_gp_09_26_25.py --hist_features_ord_id ord1 --include_sg_features True
# python3 train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features False 
# python3 train_gp_09_26_25.py --hist_features_ord_id ordall --include_sg_features True