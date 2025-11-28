DATA_ID_TO_NAME = {
    1:"good_data_thre1%",
    2:"good_data_ord3_thre0.001%",
    3:"good_data_ord3_thre1%",
    4:"good_data_thre0.001%",
    5:"good_data_ord2_thre1%",
    6:"good_data_ord2_thre1%_with0",
    7:"good_data_ord2_thre0.001%_with0",
    8:"good_data_ord2_thre1%_with_symm", # sg features added as åextra histogram (32 histograms, treat like regular data w/ 32 histograms)
    9:"good_data_ord3_thre1%_2025_2_28", # some version of ord3 with 56 histograms instead of 64 
    10:"good_data_ord3_thre1%_with_symm", # sg features added as extra histogram (67 histograms, treat like regular data w/ 67 histograms)
    11:"good_data_ord2_thre1%_with_symm_ord2", # latest 3/7/25 
    12:"good_data_ord2_thre1%_with_symm_ord1+2", # latest v2 3/7/25 
    13:"classification_data_2025_3_22", # new class data, bigger, ord2 and ord3 features, SC=1
    14:"good_data_symm_33d_thre1%", # new sg featurization 1 
    15:"good_data_symm_thre1%", # new sg featurization 2 
    16:"classification_data_2025_4_12",
    17:"classification_data_whole_20250528",
    18:"classification_data_metal_3DSCnonsc_symm_20250723", # 07/23/25
    19:"good_data_symm_thre1%_no_MgB2",
    20:"classification_data_3DSCnonsc_symm_20251002",
    21:"regression_data_histogram&symmetry",
}
# NOTE: many of these datasets have been removed for storage purposes, add back if need again later (get from G Drive)
FOUR_BEST_ORD2_HIST_FEATURES = [3,16,18,20] # using 1 indexing as I did in slides: [4,17,19,21]

# For new experiment 08/01/2025
FOUR_BEST_ORD2_HIST_FEATURES_V2 = [3, 8, 16, 20]
