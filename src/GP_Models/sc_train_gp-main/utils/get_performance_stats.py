from sklearn.metrics import make_scorer, mean_absolute_error, r2_score


def get_performance_stats(
    train_preds,
    y_train,
    test_preds,
    y_test,
):
    statistics_dict = {}

    if train_preds is not None:
        r2 = r2_score(y_train, train_preds)
        print("Train R² Score:", r2)
        statistics_dict["train_r2"] = r2

        mae = mean_absolute_error(y_train, train_preds)
        print("Train Mean Absolute Error (MAE):", mae)
        statistics_dict["train_mae"] = mae

    r2 = r2_score(y_test, test_preds)
    print("Test R² Score:", r2)
    statistics_dict["test_r2"] = r2

    mae = mean_absolute_error(y_test, test_preds)
    print("Test Mean Absolute Error (MAE):", mae)
    statistics_dict["test_mae"] = mae


    return statistics_dict
