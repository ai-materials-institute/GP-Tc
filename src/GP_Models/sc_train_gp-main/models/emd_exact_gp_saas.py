from gpytorch.means.mean import Mean
import numpy as np 
# from gpytorch.kernels.kernel import dist, Kernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.constraints import GreaterThan
from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from gpytorch.likelihoods.likelihood import Likelihood
# from botorch.models.utils import validate_input_scaling
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
import pyro
import torch 
from torch import Tensor
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian import SaasPyroModel
import sys
sys.path.append("../")
from kernels.emd_kernel import AdditiveEMDKernel



def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)

# def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
#     """Compute kernel distances."""
#     scaled_X = X / lengthscale
#     return dist(scaled_X, scaled_X, x1_eq_x2=True)

# def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
#     """Matern-5/2 kernel."""
#     _sqrt5 = math.sqrt(5)
#     dist = compute_dists(X=X, lengthscale=lengthscale)
#     sqrt5_dist = _sqrt5 * dist
#     return sqrt5_dist.add(1 + 5 / 3 * (dist**2)) * torch.exp(-sqrt5_dist)


# https://github.com/pytorch/botorch/blob/main/botorch/models/fully_bayesian.py#L189
class SaasPyroModelEMD(SaasPyroModel):
    r""" SaasPyroModel with EMD kernel
        Default SaasPyroModel hard-codes Matern Kernel 
    """

    def set_n_histogram_and_data_shape(
        self, 
        n_histogram,
        data_shape,
        path_save_ls_samples,
        n_batches_emd_kernel=None,
    ) -> None:
        # NOTE change #1 here:
        self.n_histogram = n_histogram
        self.data_shape = data_shape
        self.n_batches = n_batches_emd_kernel
        self.path_save_ls_samples = path_save_ls_samples

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        # lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.n_histogram, **tkwargs)
        if self.train_Y.shape[-2] > 0:
            # Do not attempt to sample Y if the data is empty.
            # This leads to errors with empty data.
            # replace self.ard_num_dims w/ n_histograms 
            # NOTE: change #2 here: 
            covar_module = AdditiveEMDKernel(
                n_histogram=self.n_histogram, 
                data_shape=self.data_shape,
                n_batches=self.n_batches,
                map_saas_tau=None,
                # n inputs: 11971600, bsz:1024, n batches:11692
            )
            # set covar module weights and lengthscales so comp happens how we want in compute_emd_more_memory_efficient
            temp_lengthscale = reshape_and_detach(target=covar_module.lengthscales,new_value=lengthscale,)
            covar_module._set_lengthscales(temp_lengthscale)
            temp_outputscale = torch.tensor([outputscale]*self.n_histogram)
            # torch.cat([torch.tensor(outputscale)]*self.n_histogram, 0 ) 
            temp_outputscale = reshape_and_detach(target=covar_module.weights,new_value=temp_outputscale,)
            covar_module._set_weights(temp_outputscale)
            # get emd K 
            K = covar_module.compute_emd_more_memory_efficient(self.train_X, self.train_X)
            K = K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
            
            # if False: # perv version 
                # EMD_matrix = covar_module.compute_emd(self.train_X, self.train_X)
                # EMD_ls = EMD_matrix / lengthscale.view(1, 1, -1)
                # K1 = torch.exp(-EMD_ls)
                # scaled_K = K1 * outputscale.view(1, 1, -1)
                # scaled_K = scaled_K.sum(-1)
                # K = scaled_K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
            if False:
                scaled_X = self.train_X / lengthscale
                EMD_matrix = covar_module.compute_emd(scaled_X, scaled_X) # torch.Size([3460, 3460, 31])
                K = torch.exp(-EMD_matrix) # torch.Size([3460, 3460, 31])
                K = K.sum(-1) # torch.Size([3460, 3460])
            # XXX K = matern52_kernel(X=self.train_X, lengthscale=lengthscale) # XXX above lines replace this
            # XXX K = outputscale * K + noise * torch.eye(self.train_X.shape[0], **tkwargs)  # torch.Size([3460, 3460])
            pyro.sample(
                "Y",
                pyro.distributions.MultivariateNormal(
                    loc=mean.view(-1).expand(self.train_X.shape[0]),
                    covariance_matrix=K,
                ),
                obs=self.train_Y.squeeze(-1),
            )
            torch.cuda.empty_cache()


    def load_mcmc_samples(
        self, mcmc_samples: dict[str, Tensor]
    ) -> tuple[Mean, Kernel, Likelihood]:
        # NOTE: now running with path_save_ls_samples != none once at end with tons of samples 
        if self.path_save_ls_samples is not None:
            save_ls_samples = mcmc_samples["lengthscale"].detach().cpu().numpy()
            print("saving ls samples:", save_ls_samples.shape, "at:", self.path_save_ls_samples )
            np.save(self.path_save_ls_samples, save_ls_samples) # (N_samples, 31)
            # assert 0, "Aborting after saving samples"
            
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"]) # 3 
        batch_shape = torch.Size([num_mcmc_samples]) # torch.Size([3])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        # NOTE: change #3 here: 
        # XXX covar_module = ScaleKernel(
        #     base_kernel=MaternKernel(
        #         ard_num_dims=self.ard_num_dims,
        #         batch_shape=batch_shape,
        #     ),
        #     batch_shape=batch_shape,
        # ).to(**tkwargs)
        covar_module = AdditiveEMDKernel(
            n_histogram=self.n_histogram, 
            data_shape=self.data_shape,
            n_batches=self.n_batches,
            add_saas_ls_prior=False,
            map_saas_tau=None,
        )
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        # NOTE: change #4 here: (no base_kernel)
        # XXX covar_module.base_kernel.lengthscale = reshape_and_detach(
        #     target=covar_module.base_kernel.lengthscale,
        #     new_value=mcmc_samples["lengthscale"],
        # )
        # mcmc_samples["lengthscale"].shape # torch.Size([1, 31])
        # covar_module.lengthscales.shape  # torch.Size([31])
        # with thinning < n_samples: (thinning=1, n_samples=3)
        # mcmc_samples["lengthscale"].shape: torch.Size([3, 31])
        # covar_module.lengthscales.shape # torch.Size([31])
        # mcmc_samples["lengthscale"].mean(0).shape : torch.Size([31])

        
        if num_mcmc_samples > 1:
            new_lengthscale = reshape_and_detach(
                target=covar_module.lengthscales,
                new_value=mcmc_samples["lengthscale"].mean(0), # ??? 
            )
        else:
            new_lengthscale = reshape_and_detach(
                target=covar_module.lengthscales,
                new_value=mcmc_samples["lengthscale"],
            )
        covar_module._set_lengthscales(new_lengthscale)
        # covar_module._set_lengthscales(mcmc_samples["lengthscale"]) XXX 31*3=93, does not match shape 31 
            
        # fit_fully_bayesian_model_nuts: https://github.com/pytorch/botorch/blob/main/botorch/fit.py#L339
        # Saas pyro model: https://github.com/pytorch/botorch/blob/main/botorch/models/fully_bayesian.py#L189

        # NOTE: change #5 here: (no outputscale)
        # covar_module.outputscale = reshape_and_detach(
        #     target=covar_module.outputscale,
        #     new_value=mcmc_samples["outputscale"],
        # )
        # mcmc_samples["outputscale"].shape  torch.Size([1])
        # torch.cat([mcmc_samples["outputscale"]]*self.n_histogram, 0 ).shape # torch.Size([31])
        # covar_module.weights.shape   torch.Size([31])
        # w/ n_samples > 1: 
        # mcmc_samples["outputscale"].shape torch.Size([3]
        # mcmc_samples["outputscale"].mean().shape # torch.Size([])
        # 
        if num_mcmc_samples > 1:
            new_values_tensor = torch.tensor([mcmc_samples["outputscale"].mean()]*self.n_histogram)  # torch.Size([31])
        else:
            new_values_tensor = torch.cat([mcmc_samples["outputscale"]]*self.n_histogram, 0 ) 
        new_outputscale = reshape_and_detach(target=covar_module.weights,new_value=new_values_tensor,) # torch.Size([31])
        covar_module._set_weights(new_outputscale)

        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


# See here: SaasFullyBayesianSingleTaskGP
# https://github.com/pytorch/botorch/blob/main/botorch/models/fully_bayesian.py
class EMDExactGPModelSaasFullyBayesian(SaasFullyBayesianSingleTaskGP):
    def __init__(
            self, 
            train_x, 
            train_y, 
            path_save_ls_samples=None,
            n_histogram=31, 
            data_shape=(31, 20, 2),
            input_transform=None,
            outcome_transform=None,
            n_batches_emd_kernel=None, # if int given, split kernel comp into this many batches
        ):
        pyro_model = SaasPyroModelEMD()
        pyro_model.set_n_histogram_and_data_shape(
            n_histogram=n_histogram,
            data_shape=data_shape,
            n_batches_emd_kernel=n_batches_emd_kernel,
            path_save_ls_samples=path_save_ls_samples,
        )
        super(EMDExactGPModelSaasFullyBayesian, self).__init__(
            train_X=train_x, 
            train_Y=train_y,
            train_Yvar=None,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            pyro_model=pyro_model,
        )

        
# Model assumes that the inputs have been normalized to [0, 1]^d and that
#   the output has been standardized to have zero mean and unit variance. You can
#   either normalize and standardize the data before constructing the model or use
#   an `input_transform` and `outcome_transform`. 

#   You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
# #     isn't compatible with `fit_gpytorch_mll`.

# Example:
#     >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
#     >>> fit_fully_bayesian_model_nuts(saas_gp)
#     >>> posterior = saas_gp.posterior(test_X)