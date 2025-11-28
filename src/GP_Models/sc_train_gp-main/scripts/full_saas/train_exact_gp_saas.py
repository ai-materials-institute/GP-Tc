import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME
from data_loading_scripts.load_data import load_data
import torch 
import copy 
import math 
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from utils.get_performance_stats import get_performance_stats
from utils.create_wandb_tracker import create_tracker
from botorch.fit import fit_fully_bayesian_model_nuts
from utils.str2bool_for_argparse import str2bool
from models.emd_exact_gp_saas import EMDExactGPModelSaasFullyBayesian
import argparse 
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["WANDB_SILENT"] = "True"
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DATA_DIRECTORY = "../save_model_data"

def main(
    dataset_name,
    thinning,
    max_tree_depth,
    warmup_steps,
    num_samples,
    num_samples_save_at_end=13,
    n_batches_emd_kernel=None,
    remove_ord1_features=False,
):
    ls_sampling_flag = "batch"
    model = "exact-gp-saas-try2" # for wandb config dict 
    wandb_config_dict = {k: v for k, v in locals().items()}
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
    )

    save_data_dir = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir = add_to_dir(save_data_dir, "saas_fully_bayesian")
    save_data_dir = add_to_dir(save_data_dir, dataset_name)
    save_data_dir = add_to_dir(save_data_dir, f"{wandb_run_name}")

    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data(
        dataset_name=dataset_name,
        device=device,
        remove_ord1_features=remove_ord1_features,
    )
    y_train = y_train.unsqueeze(-1)
    y_test = y_test.unsqueeze(-1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    # initialize model
    #  r"""Normalize the inputs to the unit cube. https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/input.py
    input_transform = Normalize(d=X_train.shape[-1]) 
    # output has been standardized to have zero mean and unit variance.
    #  r"""Standardize outcomes (zero mean, unit variance).: https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/outcome.py
    #   m: The output dimension.
    outcome_transform = Standardize(m=1)

    model = EMDExactGPModelSaasFullyBayesian(
        train_x=X_train, 
        train_y=y_train, 
        data_shape=data_shape, 
        n_histogram=n_histogram,
        path_save_ls_samples=None,
        input_transform=input_transform,
        outcome_transform=outcome_transform,
        n_batches_emd_kernel=n_batches_emd_kernel,
    ).to(device=device)
    # fit model w/ fit_fully_bayesian_model_nuts: https://github.com/pytorch/botorch/blob/main/botorch/fit.py#L339
    fit_fully_bayesian_model_nuts(
        model=model,
        max_tree_depth=max_tree_depth, 
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        thinning=thinning,
    )
    
    # lines to avoid GPU OOM during predictions 
    model._clear_cache()
    torch.cuda.empty_cache()
    model._clear_cache()
    # get preds, take 'mixture_mean' of GaussianMixturePosterior
    with torch.no_grad():
        posterior_test = model.posterior(X_test)
        test_preds = posterior_test.mixture_mean.detach().cpu().numpy()

        posterior_train = model.posterior(X_train)
        train_preds = posterior_train.mixture_mean.detach().cpu().numpy()

    statistics_dict = get_performance_stats(
        train_preds=train_preds.squeeze(),
        y_train=y_train.cpu().numpy().squeeze(),
        test_preds=test_preds.squeeze(),
        y_test=y_test.cpu().numpy().squeeze(),
    )
    tracker.log(statistics_dict)

    if train_preds is not None:
        np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, f"{save_data_dir}/model_state.pt")

    tracker.finish()

    # THEN: try to get lots of samples... 
    if False:
        model.pyro_model.path_save_ls_samples = f"{save_data_dir}/lengthscales_n_samples_by_n_histograms.npy"
        fit_fully_bayesian_model_nuts(
            model=model,
            max_tree_depth=max_tree_depth, 
            warmup_steps=0,
            num_samples=num_samples_save_at_end,
            thinning=1,
        )

    # Alternative V2: ** remove the assert 0 before testing out. 
    # indicated in wandb with ls_sampling_flag = "batch"
    n_iters = math.ceil(num_samples_save_at_end/num_samples)
    for iter in range(n_iters):
        model._clear_cache()
        torch.cuda.empty_cache()
        model._clear_cache()
        model.pyro_model.path_save_ls_samples = f"{save_data_dir}/lengthscales_n_samples_by_n_histograms_{iter + 1}.npy"
        fit_fully_bayesian_model_nuts(
            model=model,
            max_tree_depth=max_tree_depth, 
            warmup_steps=warmup_steps,
            num_samples=num_samples, # num_samples at a time, up to at least num_samples_save_at_end
            thinning=1,
        )


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
        choices=[1,2,3,4,6,7],
        default=4,
        required=False,
    )
    parser.add_argument(
        "--thinning",
        help="int thinning arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=16, # before 2/19/25 I used thinning > num_samples --> only one sample left! Bad! 
        required=False,
    )
    parser.add_argument(
        "--max_tree_depth",
        help="int max_tree_depth arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--warmup_steps",
        help="int warmup_steps arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=1, # botorch default = 512, # use lower i.e. 3 for debugging 
        required=False,
    )
    parser.add_argument(
        "--num_samples",
        help="int num_samples arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=6, # botorch default = 256, # use lower i.e. 3 for debugging, 16 --> avoid OOM on largest 48G GPUs we have 
        required=False,
    )
    parser.add_argument(
        "--n_batches_emd_kernel",
        help="int, if None, no batching, if int given, split kernel comp into this many batches to avoid OOM",
        type=int,
        default=10, # None, None --> no batching, 2 --> feasibly avoid GPU OOM on largest 48G GPUs we have 
        required=False,
    )
    parser.add_argument(
        "--num_samples_save_at_end",
        help="int, num ls samples to save at end of opt",
        type=int,
        default=20, 
        required=False,
    )
    parser.add_argument(
        "--remove_ord1_features",
        help="bool, if remove first 10 histogram features (order 1 features)",
        type=str2bool,
        default=True,
        required=False,
    )
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    main(
        dataset_name=dataset_name, 
        thinning=args.thinning,
        max_tree_depth=args.max_tree_depth,
        warmup_steps=args.warmup_steps,
        num_samples=args.num_samples,
        n_batches_emd_kernel=args.n_batches_emd_kernel,
        num_samples_save_at_end=args.num_samples_save_at_end,
        remove_ord1_features=args.remove_ord1_features,
    )

