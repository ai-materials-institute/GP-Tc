import torch 
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader


def get_class_model_preds(
    model, 
    likelihood,
    X_test,
    X_train,
    bsz,
    also_return_raw_preds=False,
):
    # test model in eval mode 
    model.eval()
    likelihood.eval()
    # data loaders for x train and test 
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=bsz, shuffle=False)
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=False)

    # get raw train preds 
    train_preds_raw = []
    for (train_x_batch,) in train_loader:
        with torch.no_grad():
            train_preds_batch = likelihood(model(train_x_batch)).mean.cpu()
        train_preds_raw.append(train_preds_batch)
    train_preds_raw = torch.cat(train_preds_raw).numpy()

    # get raw test preds 
    test_preds_raw = []
    for (test_x_batch,) in test_loader:
        with torch.no_grad():
            test_preds_batch = likelihood(model(test_x_batch)).mean.cpu()
        test_preds_raw.append(test_preds_batch)
    test_preds_raw = torch.cat(test_preds_raw).numpy()

    # covert raw preds to 0/1 classifications, where > 0.5 we predict 1 
    test_preds, train_preds = convert_raw_preds_to_class_preds(
        test_preds_raw=test_preds_raw, 
        train_preds_raw=train_preds_raw, 
    )
    
    if also_return_raw_preds:
        return train_preds, test_preds, train_preds_raw, test_preds_raw
    return train_preds, test_preds 

def convert_raw_preds_to_class_preds(test_preds_raw, train_preds_raw, ):
    test_preds = np.zeros(len(test_preds_raw))
    test_preds[test_preds_raw >= 0.5] = test_preds[test_preds_raw >= 0.5] + 1.0
    train_preds = np.zeros(len(train_preds_raw))
    train_preds[train_preds_raw >= 0.5] = train_preds[train_preds_raw >= 0.5] + 1.0
    return test_preds, train_preds 
