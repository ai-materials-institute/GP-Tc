from sklearn.metrics import confusion_matrix


def get_performance_stats_classification(
    y_test,
    y_train,
    test_preds,
    train_preds,
):
    statistics_dict = {}
    # for classificaiton, R2, MAE don't make much sense, especially R2
    # Instead we want acc, precision, recall 
    acc_test = (test_preds == y_test).sum().item() / len(test_preds)
    acc_train = (train_preds == y_train).sum().item() / len(train_preds)
    statistics_dict["test_acc"] = acc_test
    statistics_dict["train_acc"] = acc_train

    matrix = confusion_matrix(y_test, test_preds)
    # row 1 = true 1 
    # row 0 = true 0 
    # col 1 = pred 1 
    # col 0 = pred 0 
    recall_tc_1 = matrix[1,1] / (matrix[1,1] + matrix[1,0]) # acc when true tc is 1  
    precision_tc_1 = matrix[1,1] / (matrix[1,1] + matrix[0,1]) # acc when we precict tc 1 
    recall_tc_0 = matrix[0,0] / (matrix[0,0] + matrix[0,1]) # acc when true tc is 0 
    precision_tc_0 = matrix[0,0]/ (matrix[0,0] + matrix[1,0]) # acc when we precict tc 0 
    statistics_dict["test_recall_sc"] = recall_tc_1
    statistics_dict["test_precision_sc"] = precision_tc_1 
    statistics_dict["test_recall_nonsc"] = recall_tc_0
    statistics_dict["test_precision_nonsc"] = precision_tc_0 

    # repeat for train set 
    matrix = confusion_matrix(y_train, train_preds)
    recall_tc_1 = matrix[1,1] / (matrix[1,1] + matrix[1,0]) # acc when true tc is 1  
    precision_tc_1 = matrix[1,1] / (matrix[1,1] + matrix[0,1]) # acc when we precict tc 1 
    recall_tc_0 = matrix[0,0] / (matrix[0,0] + matrix[0,1]) # acc when true tc is 0 
    precision_tc_0 = matrix[0,0]/ (matrix[0,0] + matrix[1,0]) # acc when we precict tc 0 
    statistics_dict["train_recall_sc"] = recall_tc_1
    statistics_dict["train_precision_sc"] = precision_tc_1 
    statistics_dict["train_recall_nonsc"] = recall_tc_0
    statistics_dict["train_precision_nonsc"] = precision_tc_0 

    return statistics_dict

