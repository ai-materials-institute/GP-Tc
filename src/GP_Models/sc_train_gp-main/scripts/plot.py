import matplotlib.pyplot as plt 
import torch 
import sys 
import numpy as np 
import json 
import glob 
import os 
import argparse 
sys.path.append("../")
from utils.get_performance_stats import get_performance_stats
from utils.get_acc_stats_classification import get_performance_stats_classification
from data_loading_scripts.load_data import load_data
from data_loading_scripts.load_data_w_sg import load_data_with_sgs
from data_loading_scripts.load_data_classification import load_class_data
from constants import DATA_ID_TO_NAME
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_results(dataset_name, model_id_string=None):
    save_data_dirs_path = "../save_model_data"
    # if saas_model: 
    #     save_data_dirs_path = save_data_dirs_path + "/saas_fully_bayesian"
    if model_id_string is not None:
        save_data_dirs_path = save_data_dirs_path + f"/{model_id_string}"
    # TO-DO: change back to single star 
    save_data_dirs_path = save_data_dirs_path + f"/{dataset_name}/*/*/test_preds.npy"
    save_data_dirs = glob.glob(save_data_dirs_path)
    save_data_dirs = [dir1.replace("/test_preds.npy", "") for dir1 in save_data_dirs]
    if "graph" in model_id_string:
        load_data_func = load_data_with_sgs
        get_stats_func = get_performance_stats
    elif "class" in model_id_string:
        load_data_func = load_class_data
        get_stats_func = get_performance_stats_classification
    else:
        load_data_func = load_data
        get_stats_func = get_performance_stats

    for save_dir in save_data_dirs:
        _, _, y_train, y_test, data_shape, n_histogram = load_data_func(
            dataset_name=dataset_name,
            device=torch.device("cpu"),
        ) 
        # TO-DO: remove following two lines 
        try:
            os.remove(f"{save_dir}/test_result.png")
        except:
            pass
        try:
            os.remove(f"{save_dir}/train_result.png")
        except:
            pass
        y_train = y_train.cpu().numpy()
        y_test = y_test.cpu().numpy()
        train_preds_path = f"{save_dir}/train_preds.npy"
        if os.path.exists(train_preds_path):
            train_preds = np.load(train_preds_path )
        else:
            train_preds = None 

        test_preds = np.load(f"{save_dir}/test_preds.npy")
        
        statistics_dict = get_stats_func(
            train_preds=train_preds,
            y_train=y_train,
            test_preds=test_preds,
            y_test=y_test,
        ) 
        if "class" in model_id_string:
            with open(f"{save_dir}/performance.json", 'w') as file:
                json.dump(statistics_dict, file, indent=4) # indent is optional, for pretty formatting
        else:
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, test_preds)
            plt.xlabel("True Test Y Value")
            plt.ylabel("Exact GP Model's Predicted Value")
            plt.title(f"Test R2:{statistics_dict["test_r2"]:.3f}, Test MAE:{statistics_dict["test_mae"]:.3f}")
            plt.savefig(f"{save_dir}/test_result.png")
            plt.clf()

            if train_preds is not None:
                plt.figure(figsize=(10, 8))
                plt.scatter(y_train, train_preds)
                plt.xlabel("True Train Y Value")
                plt.ylabel("Exact GP Model's Predicted Value")
                plt.title(f"Train R2:{statistics_dict["train_r2"]:.3f}, Train MAE:{statistics_dict["train_mae"]:.3f}")
                plt.savefig(f"{save_dir}/train_result.png")
                plt.clf()
        
        if "saas" in model_id_string:
            if os.path.exists(f"{save_dir}/lengthscales_n_samples_by_n_histograms_1.npy"):
                all_ls_paths = glob.glob(f"{save_dir}/lengthscales_n_samples_by_n_histograms_*.npy")
                all_ls_samples = []
                total_n_samples = 0
                for ls_path in all_ls_paths:
                    ls_samples = np.load(ls_path)
                    total_n_samples += ls_samples.shape[0]
                    assert ls_samples.shape[1] == 31 # n histograms 
                    all_ls_samples.append(torch.tensor(ls_samples).float()) # N,31
                all_ls_samples = torch.cat(all_ls_samples, 0)
                assert all_ls_samples.shape[0] == total_n_samples
                assert all_ls_samples.shape[1] == 31 # n histograms 
                np.save(f"{save_dir}/lengthscales_{total_n_samples}samples.npy", all_ls_samples.numpy())
            if os.path.exists(f"{save_dir}/lengthscales_n_samples_by_n_histograms.npy"):
                ls_samples = np.load(f"{save_dir}/lengthscales_n_samples_by_n_histograms.npy")
                n_samples = ls_samples.shape[0]
                assert ls_samples.shape[1] == 31 # n histograms 
                np.save(f"{save_dir}/lengthscales_{n_samples}samples.npy", ls_samples)

        learned_ls_path = f"{save_dir}/emd_kernel_lengthscales.npy"
        if os.path.exists(learned_ls_path):
            learned_ls = np.load(learned_ls_path) 
            learned_ls = learned_ls.tolist()
            categories = [f"{i+1}" for i in range(len(learned_ls))]
            fig_width = len(learned_ls)//3 
            plt.figure(figsize=(fig_width, 4))
            plt.bar(categories, learned_ls, color='skyblue')
            plt.xlabel('Histogram Feature')
            plt.ylabel('Lengthscale Value')
            title_bar = f"EMD Learned Lengthscales"
            plt.title(title_bar)
            plt.savefig(f"{save_dir}/emd_learned_lengthscales.png")
            plt.clf()
        
        symm_graph_ls_path = f"{save_dir}/sg_kernel_lengthscale.npy"
        if os.path.exists(symm_graph_ls_path):
            learned_ls = np.load(symm_graph_ls_path) 
            learned_ls = learned_ls.tolist()
            categories = [f"{i+1}" for i in range(len(learned_ls))]
            fig_width = len(learned_ls)//2 
            plt.figure(figsize=(fig_width, 4))
            plt.bar(categories, learned_ls, color='skyblue')
            plt.xlabel('Symm SG Feature')
            plt.ylabel('Lengthscale Value')
            title_bar = f"Symm SG Learned Lengthscales"
            plt.title(title_bar)
            plt.savefig(f"{save_dir}/symm_sg_learned_lengthscales.png")
            plt.clf()




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
        "--model_id_string",
        help=" id string for model, dictates subdir in save_model_data dir",
        type=str,
        default=None, 
        required=False,
    ) 
    args = parser.parse_args() 
    dataset_name = DATA_ID_TO_NAME[args.dataset_id]
    plot_results(
        dataset_name=dataset_name,
        model_id_string=args.model_id_string,
    )
# python3 plot.py --dataset_id 13 --model_id_string class-gp