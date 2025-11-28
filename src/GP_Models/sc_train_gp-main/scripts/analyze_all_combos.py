import pandas as pd 
import numpy as np 
from itertools import combinations

def main(
    num_features=4,
    total_num_features=32, # 21 for only ord2 features 
    test_stat_string_id="test-r2", # "test-acc" for class gp
    wandb_data_filepath = f"../save_model_data/wandb-all-combos-4-features.csv"
):
    df = pd.read_csv(wandb_data_filepath) # (3887, 406)
    for label in df.keys():
        if ('MIN' in label) or ('MAX' in label) or ('_step' in label) or ('Step' in label):
            df = df.drop(label, axis=1)
    # df.shape (3887, 135) --> 27 runs 


    save_df = {
        test_stat_string_id:[]
    }
    for i in range(num_features):
        save_df[f"feature{i+1}"] = []

    for label in df.keys():
        if test_stat_string_id in label: 
            wandb_run_name = label.replace(f" - {test_stat_string_id}", "")
            test_stat = df[label].values # (3887,)
            bool_arr = ~np.isnan(test_stat)
            test_stat = test_stat[bool_arr]
            for i in range(num_features):
                fti = df[f"{wandb_run_name} - feature{i + 1}"].values.astype(int)
                fti = fti[bool_arr]
                assert fti.shape[0] == test_stat.shape[0]
                assert np.isnan(fti).sum() == 0 
                save_df[f"feature{i+1}"] = save_df[f"feature{i+1}"] + fti.tolist()
            save_df[test_stat_string_id] = save_df[test_stat_string_id] + test_stat.tolist()

    # create df for all results 
    save_df = pd.DataFrame.from_dict(save_df) 
    subset = [f"feature{i+1}" for i in range(num_features)] # remove repeated combos
    save_df = save_df.drop_duplicates(subset=subset) # (35960, 5) (32 choose 4 is 35960)
    print(save_df.shape)
    
    # Check for missing combos: 
    all_combos_done = save_df[subset].values # (35952, 4)
    all_combos_done = all_combos_done - 1 # as feature ids instead of feature numbers 
    all_combos_done = all_combos_done.tolist()
    all_combos = list(combinations(range(total_num_features), num_features)) # (35960, 4)
    all_combos = np.array(all_combos).tolist()
    missing_combos = []
    for combo_i in all_combos:
        if not (combo_i in all_combos_done):
            missing_combos.append(list(combo_i))
    print("Number of missing combos:", len(missing_combos), "Mising combos are:", missing_combos)
    assert len(missing_combos) == len(all_combos) - len(all_combos_done)

    # Save result 
    save_df = save_df.sort_values(by=test_stat_string_id, ascending=False) # sort by top test r2 
    save_df.to_csv(f"../save_model_data/all-combos-{num_features}-features-sorted-by-{test_stat_string_id}.csv", index=False)



if __name__ == "__main__":
    # NOTE: first download wandb data from plot with test stat, ft1,ft2,...,ftn
    # Example: used before to get all bf combos of features for TC prediction: 
    # main(
    #     num_features=4,
    #     total_num_features=32, 
    #     test_stat_string_id="test-r2", 
    #     wandb_data_filepath = f"../save_model_data/wandb-all-combos-4-features.csv"
    # )
    # For latest class gp version: 
    # TO-DO download wandb data, then run locally. 
    main(
        num_features=4,
        total_num_features=21, # 21 for only ord2 features 
        test_stat_string_id="test-acc", # "test-acc" for class gp
        wandb_data_filepath = f"../save_model_data/wandb-all-combos-4-features-class-gp.csv"
    )
