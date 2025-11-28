import sys 
sys.path.append("../")
from constants import DATA_ID_TO_NAME
from data_loading_scripts.load_data_w_sg import load_data_with_sgs
from utils.str2bool_for_argparse import str2bool
import torch 
import copy 
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from utils.get_performance_stats import get_performance_stats
from utils.create_wandb_tracker import create_tracker
from botorch.fit import fit_fully_bayesian_model_nuts
from models.graph_emd_exact_gp_saas import GraphEMDExactGPModelSaasFullyBayesian
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
    sg_feature_rank,
    n_batches_emd_kernel=None,
    use_input_transform=False,
    remove_ord1_features=False,
    use_11d_symm_sg_features=False,
):
    if use_11d_symm_sg_features:
        assert 0, "not yet set up for use_11d_symm_sg_features=True version"
    model = "graph-emd-exact-gp-saas" 
    kernel_flag = "index-kernel" # flag for using index kernel to add sg features 
    wandb_config_dict = {k: v for k, v in locals().items()}
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
    )
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_with_sgs(
        dataset_name=dataset_name,
        device=device,
        remove_ord1_features=remove_ord1_features,
        use_11d_symm_sg_features=use_11d_symm_sg_features,
    )
    y_train = y_train.unsqueeze(-1)
    y_test = y_test.unsqueeze(-1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model_id_string = copy.deepcopy(model)
    
    # initialize model
    #  r"""Normalize the inputs to the unit cube. https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/input.py
    input_transform = None # using standard norm input transform might mess up the sg features, unsure though
    if use_input_transform:
        input_transform = Normalize(d=X_train.shape[-1]) 
    # output has been standardized to have zero mean and unit variance.
    #  r"""Standardize outcomes (zero mean, unit variance).: https://github.com/pytorch/botorch/blob/main/botorch/models/transforms/outcome.py
    #   m: The output dimension.
    outcome_transform = Standardize(m=1)
    model = GraphEMDExactGPModelSaasFullyBayesian(
        train_x=X_train, 
        train_y=y_train, 
        sg_feature_rank=sg_feature_rank,
        data_shape=data_shape, 
        n_histogram=n_histogram,
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

    save_data_dir = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir = add_to_dir(save_data_dir, model_id_string)
    save_data_dir = add_to_dir(save_data_dir, dataset_name)
    save_data_dir = add_to_dir(save_data_dir, f"{wandb_run_name}")

    if train_preds is not None:
        np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    
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
        choices=[3,5],
        default=1,
        required=False,
    )
    parser.add_argument(
        "--thinning",
        help="int thinning arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=16,
        required=False,
    )
    parser.add_argument(
        "--max_tree_depth",
        help="int max_tree_depth arg for fit_fully_bayesian_model_nuts",
        type=int,
        default=6,
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
        default=2, # botorch default = 256, # use lower i.e. 3 for debugging, 16 --> avoid OOM on largest 48G GPUs we have 
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
        "--sg_feature_rank",
        help="int, rank for sg feature index kernel",
        type=int,
        default=10, 
        required=False,
    )
    parser.add_argument(
        "--use_input_transform",
        help="bool, if true use Normalize input transform (standard for SAAS)",
        type=str2bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--remove_ord1_features",
        help="bool, if remove first 10 histogram features (order 1 features)",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--use_11d_symm_sg_features",
        help="bool, if True, use alternative 11D SYMM sg features",
        type=str2bool,
        default=False,
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
        use_input_transform=args.use_input_transform,
        sg_feature_rank=args.sg_feature_rank,
        remove_ord1_features=args.remove_ord1_features,
        use_11d_symm_sg_features=args.use_11d_symm_sg_features,
    )

