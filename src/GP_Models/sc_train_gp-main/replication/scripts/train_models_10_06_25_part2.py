import argparse
import copy
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.environ['WANDB_SILENT'] = 'True'

SC_TRAIN_ROOT = Path(__file__).resolve().parents[2]
REPLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(SC_TRAIN_ROOT) not in sys.path:
    sys.path.append(str(SC_TRAIN_ROOT))

from constants import DATA_ID_TO_NAME
from models.emd_exact_gp import EMDExactGPModel
from replication.data_loading_scripts.load_data_10_06_25_part2 import load_data_with_only_specific_features_10_25
from utils.create_wandb_tracker import create_tracker
from utils.get_model_preds import get_model_predictions
from utils.get_performance_stats import get_performance_stats


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DATA_DIRECTORY = str(REPLICATION_ROOT / 'save_model_data')


def main(features_to_keep_list, features_subset_str, save_data_dir, dataset_name, tracker, n_epochs=32):
    lr = 0.1
    n_batches_emd_kernel = 10

    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_data_with_only_specific_features_10_25(
        dataset_name=dataset_name,
        device=device,
        list_of_features_to_keep=features_to_keep_list,
    )
    plot_titles_string = f'Performance w Hist. Features: {features_subset_str}'

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

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    train_preds, test_preds = get_model_predictions(
        model=model,
        likelihood=likelihood,
        X_test=X_test,
        X_train=X_train,
    )
    y_test = y_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    statistics_dict = get_performance_stats(
        train_preds=train_preds,
        y_train=y_train,
        test_preds=test_preds,
        y_test=y_test,
    )
    statistics_dict = {key: float(value) for key, value in statistics_dict.items()}
    test_r2_ = statistics_dict['test_r2']
    test_mae_ = statistics_dict['test_mae']
    train_r2_ = statistics_dict['train_r2']
    train_mae_ = statistics_dict['train_mae']

    with open(f'{save_data_dir}/performance_stats.json', 'w') as file:
        json.dump(statistics_dict, file, indent=4)
    tracker.log(statistics_dict)

    np.save(f'{save_data_dir}/train_preds.npy', train_preds)
    np.save(f'{save_data_dir}/test_preds.npy', test_preds)
    torch.save(model.state_dict(), f'{save_data_dir}/model_state.pt')
    torch.save(likelihood.state_dict(), f'{save_data_dir}/likelihood_state.pt')
    np.save(f'{save_data_dir}/emd_kernel_weights.npy', model.covar_module.weights.detach().cpu().numpy())
    learned_ls = model.covar_module.lengthscales.detach().cpu().numpy()
    np.save(f'{save_data_dir}/emd_kernel_lengthscales.npy', learned_ls)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, test_preds)
    plt.xlabel('True Test Y Value')
    plt.ylabel("Exact GP Model's Predicted Value")
    plt.title('Test ' + plot_titles_string + f'\n R2:{test_r2_:.3f}, MAE:{test_mae_:.3f}')
    plt.savefig(f'{save_data_dir}/test_result.png')
    plt.clf()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, train_preds)
    plt.xlabel('True Train Y Value')
    plt.ylabel("Exact GP Model's Predicted Value")
    plt.title('Train ' + plot_titles_string + f'\nR2:{train_r2_:.3f}, MAE:{train_mae_:.3f}')
    plt.savefig(f'{save_data_dir}/train_result.png')
    plt.clf()

    gc.collect()
    torch.cuda.empty_cache()


def add_to_dir(dir_path, add_this):
    dir_path = dir_path + f'/{add_this}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, default=21, required=False)
    parser.add_argument('--n_epochs', type=int, default=32, required=False)
    parser.add_argument('--feature_subset_id', type=int, default=1, required=False)
    args = parser.parse_args()
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]

    if args.feature_subset_id == 1:
        features_to_keep_list = np.arange(0, 31).tolist()
        features_subset_str = '0-30'
    elif args.feature_subset_id == 2:
        features_to_keep_list = np.arange(10, 31).tolist()
        features_subset_str = '10-30'
    else:
        raise AssertionError('feature_subset_id must be 1 or 2')

    wandb_config_dict = {
        'dataset_id': args.dataset_id,
        'dataset_name': dataset_name,
        'n_epochs': args.n_epochs,
        'feature_subset_id': args.feature_subset_id,
        'features_subset': features_subset_str,
    }
    tracker, wandb_run_name = create_tracker(
        config_dict=wandb_config_dict,
        wandb_project_name='sc-train-gp-10-06-25',
    )
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, dataset_name)
    save_data_dir_toptop = add_to_dir(
        save_data_dir_toptop,
        f'{wandb_run_name}_epochs{args.n_epochs}_10_06_25_fts{features_subset_str}',
    )
    main(
        features_to_keep_list=features_to_keep_list,
        features_subset_str=features_subset_str,
        save_data_dir=save_data_dir_toptop,
        dataset_name=dataset_name,
        tracker=tracker,
        n_epochs=args.n_epochs,
    )

