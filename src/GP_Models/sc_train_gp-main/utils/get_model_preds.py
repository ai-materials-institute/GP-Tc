import gpytorch 
import torch 

def get_model_predictions(
    model, 
    likelihood,
    X_test,
    X_train,
):
    # test model 
    model.eval()
    likelihood.eval()
    with torch.no_grad(): # , gpytorch.settings.fast_pred_var():
        try:
            test_preds = likelihood(model(X_test)).mean.cpu().numpy()
        except:
            X_test = X_test.cpu()
            model._clear_cache()
            model.train().eval()
            model._clear_cache()

            likelihood = likelihood.to(device=torch.device("cpu")) 
            model = model.to(device=torch.device("cpu")) 
            model.covar_module = model.covar_module.to(device=torch.device("cpu")) 
            likelihood = likelihood.to(device=torch.device("cpu")) 

            model._clear_cache()
            model.train().eval()
            model._clear_cache()

            out1 = model(X_test)
            test_preds = likelihood(out1)
            test_preds = test_preds.mean.cpu().numpy()
        try:
            train_preds = likelihood(model(X_train)).mean.cpu().numpy()
        except: 
            train_preds = None # avoid OOM failure for train preds 
    return train_preds, test_preds 

