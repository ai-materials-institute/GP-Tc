import sys 
sys.path.append("../")
from data_loading_scripts.load_data_special_class_expt import load_special_example_class_data
import torch 
import gc 
import copy 
import pandas as pd 
import json 
import gpytorch
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

# TO-DO: 
# train and test on train/test data 
# do for all, order 2 only, top 10% or so of combos of 4 features only 

def main(
    tracker,
    save_data_dir_top,
    n_epochs=32, 
    n_batches_emd_kernel=None,
    order2_features_only=True,
    n_ind_pts=1024,
    add_saas_ls_prior=False,
    map_saas_tau=None,
    bsz=32,
    lr=0.1, 
    list_of_features_to_keep="all",
):
    X_train, X_test, y_train, y_test, data_shape, n_histogram = load_special_example_class_data(
        device=device, 
        list_of_features_to_keep=list_of_features_to_keep,
        order2_features_only=order2_features_only,
    )

    if list_of_features_to_keep == "all":
        if order2_features_only:
            features_kept_string = "all-ord2-features"
        else:
            features_kept_string = "all-features"
    else:
        features_kept_string = "features"
        for ft_kept in list_of_features_to_keep:
            features_kept_string = features_kept_string + f"-{ft_kept + 1}"
        
    print("Starting Run Keeping Features:", features_kept_string)
    
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
    # ** hack to to make my get preds func work w/ only 2 points 
    X_test_temp = [X_test]*bsz # [torch.Size([2, 840])]*1024
    X_test_temp = torch.cat(X_test_temp, 0) # torch.Size([2048, 840])

    train_preds, test_preds, train_preds_raw, test_preds_raw = get_class_model_preds(
        model=model, 
        likelihood=likelihood,
        X_test=X_test_temp,
        X_train=X_train,
        bsz=bsz,
        also_return_raw_preds=True,
    ) # outputs numpy arrays of classification preds 
    test_preds = test_preds[0:len(X_test)]
    test_preds_raw = test_preds_raw[0:len(X_test)]
    y_test = y_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    statistics_dict = get_performance_stats_classification(y_test=y_test,y_train=y_train,test_preds=test_preds,train_preds=train_preds,)
    dict_log = {
        "test_y1_true":y_test[0].item(),
        "test_y1_pred":test_preds[0].item(),
        "test_y1_pred_raw":test_preds_raw[0].item(),
        "test_y2_true":y_test[1].item(),
        "test_y2_pred":test_preds[1].item(),
        "test_y2_pred_raw":test_preds_raw[1].item(),
        "features_used":features_kept_string,
    }
    dict_log.update(statistics_dict)
    tracker.log(dict_log)

    save_data_dir = add_to_dir(save_data_dir_top, features_kept_string)
    np.save(f"{save_data_dir}/train_preds.npy", train_preds)
    np.save(f"{save_data_dir}/test_preds.npy", test_preds)
    np.save(f"{save_data_dir}/train_preds_raw.npy", train_preds_raw)
    np.save(f"{save_data_dir}/test_preds_raw.npy", test_preds_raw)
    torch.save(model.state_dict(), f"{save_data_dir}/model_state.pt")
    torch.save(likelihood.state_dict(), f"{save_data_dir}/likelihood_state.pt")
    with open(f"{save_data_dir}/performance.json", 'w') as file:
        json.dump(dict_log, file, indent=4) 

    # empty garbage, clear cache
    gc.collect()
    torch.cuda.empty_cache()
    tracker.finish()



def add_to_dir(dir_path, add_this):
    dir_path = dir_path + f"/{add_this}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path 


PATH_TO_BEST_COMBOS_FOUR_FEATURES = "../data_for_special_class_expt_4_16_25/classification_data_2025_4_12-all-combos-4-features-sorted-by-test-acc.csv"
if __name__ == "__main__":
    save_data_dir_toptop = copy.deepcopy(SAVE_DATA_DIRECTORY)
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"class-gp-special-expt-2-heldout-materials-with-raw")
    save_data_dir_toptop = add_to_dir(save_data_dir_toptop, f"epochs32")

    lists_of_features_to_keep = ["all", "all"]

    df = pd.read_csv(PATH_TO_BEST_COMBOS_FOUR_FEATURES) # (5985, 5)
    df = df[0:len(df)//100] #  /100 --> (59, 5) = top 1 percent     /10 --> (598, 5) = top 10 percent
    f1s = df["feature1"].values - 1 # (598,), -1 to get indexes starting at 0 
    f2s = df["feature2"].values - 1
    f3s = df["feature3"].values - 1
    f4s = df["feature4"].values - 1

    # test-acc,feature1,feature2,feature3,feature4
    for row in range(len(df)):
        four_features = [f1s[row].item(), f2s[row].item(), f3s[row].item(), f4s[row].item()]
        lists_of_features_to_keep.append(four_features)


    for ix, fts_to_keep in enumerate(lists_of_features_to_keep):
        tracker, wandb_run_name = create_tracker(
            config_dict={},
            wandb_project_name="sc-class-2-heldout-expt",
        )
        print(f"Wandb run: {wandb_run_name}, Keeping features:", fts_to_keep)

        order2_features_only_ = True 
        if ix == 0:
            order2_features_only_ = False # only on first round, do all 56 features 

        # TO-DO: 
        # train and test on train/test data 
        # do for all, order 2 only, top 10% or so of combos of 4 features only 

        # 2. list_of_features_to_keep
        main(
            tracker=tracker,
            save_data_dir_top=save_data_dir_toptop,
            n_epochs=32, 
            n_batches_emd_kernel=10,
            order2_features_only=order2_features_only_,
            n_ind_pts=1024,
            add_saas_ls_prior=False,
            map_saas_tau=None,
            bsz=1024,
            lr=0.05, 
            list_of_features_to_keep=fts_to_keep,
        )


