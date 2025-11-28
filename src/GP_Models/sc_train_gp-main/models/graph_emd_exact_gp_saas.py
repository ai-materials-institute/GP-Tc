from gpytorch.means.mean import Mean
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
from kernels.graph_emd_kernel import AdditiveSgEMDKernel



def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)

# https://github.com/pytorch/botorch/blob/main/botorch/models/fully_bayesian.py#L189
class SaasPyroModelGraphEMD(SaasPyroModel):
    r""" SaasPyroModel with EMD kernel
        Default SaasPyroModel hard-codes Matern Kernel 
    """

    def set_n_histogram_and_data_shape(
        self, 
        n_histogram,
        data_shape,
        sg_feature_rank,
        n_batches_emd_kernel=None,
    ) -> None:
        # NOTE change #1 here:
        self.n_histogram = n_histogram
        self.data_shape = data_shape
        self.n_batches = n_batches_emd_kernel
        self.sg_feature_rank = sg_feature_rank

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
            # NOTE: change #2 here: 
            covar_module = AdditiveSgEMDKernel(
                n_histogram=self.n_histogram, 
                data_shape=self.data_shape,
                n_batches=self.n_batches,
                map_saas_tau=None,
                sg_feature_rank=self.sg_feature_rank,
                add_saas_ls_prior=False,
                use_11d_symm_sg_features=False,
            )
            # set covar module weights and lengthscales so comp happens how we want in compute_emd_more_memory_efficient
            temp_lengthscale = reshape_and_detach(target=covar_module.lengthscales,new_value=lengthscale,)
            covar_module._set_lengthscales(temp_lengthscale)
            temp_outputscale = torch.tensor([outputscale]*self.n_histogram)
            # torch.cat([torch.tensor(outputscale)]*self.n_histogram, 0 ) 
            temp_outputscale = reshape_and_detach(target=covar_module.weights,new_value=temp_outputscale,)
            covar_module._set_weights(temp_outputscale)
            # Use histogram features to construct EMD matrix (assume first feature is SG ID)
            emd_K = covar_module.compute_emd_more_memory_efficient(self.train_X[:,1:], self.train_X[:,1:])
            # Multiply by index kernel for SG Graph features 
            final_K = emd_K * covar_module.index_kernel(self.train_X[:,0] - 1, self.train_X[:,0] - 1).to(self.train_X.device)
            # add noise 
            K = final_K + noise * torch.eye(self.train_X.shape[0], **tkwargs)
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
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        # NOTE: change #3 here: 
        # XXX covar_module = ScaleKernel(
        #     base_kernel=MaternKernel(
        #         ard_num_dims=self.ard_num_dims,
        #         batch_shape=batch_shape,
        #     ),
        #     batch_shape=batch_shape,
        # ).to(**tkwargs)
        covar_module = AdditiveSgEMDKernel(
            n_histogram=self.n_histogram, 
            data_shape=self.data_shape,
            n_batches=self.n_batches,
            map_saas_tau=None,
            sg_feature_rank=self.sg_feature_rank,
            add_saas_ls_prior=False,
            use_11d_symm_sg_features=False,
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
        new_lengthscale = reshape_and_detach(
            target=covar_module.lengthscales,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module._set_lengthscales(new_lengthscale)

        # NOTE: change #5 here: (no outputscale)
        # covar_module.outputscale = reshape_and_detach(
        #     target=covar_module.outputscale,
        #     new_value=mcmc_samples["outputscale"],
        # )
        # mcmc_samples["outputscale"].shape  torch.Size([1])
        # covar_module.weights.shape   torch.Size([32])
        # XXX new_values_tensor = torch.cat([mcmc_samples["outputscale"]]*(self.n_histogram+1), 0 ) 
        new_values_tensor = torch.cat([mcmc_samples["outputscale"]]*self.n_histogram, 0 ) 
        new_outputscale = reshape_and_detach(target=covar_module.weights,new_value=new_values_tensor,)
        covar_module._set_weights(new_outputscale)

        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


# See here: SaasFullyBayesianSingleTaskGP
# https://github.com/pytorch/botorch/blob/main/botorch/models/fully_bayesian.py
class GraphEMDExactGPModelSaasFullyBayesian(SaasFullyBayesianSingleTaskGP):
    def __init__(
            self, 
            train_x, 
            train_y, 
            sg_feature_rank,
            n_histogram=31, 
            data_shape=(31, 20, 2),
            input_transform=None,
            outcome_transform=None,
            n_batches_emd_kernel=None, # if int given, split kernel comp into this many batches
        ):
        pyro_model = SaasPyroModelGraphEMD()
        pyro_model.set_n_histogram_and_data_shape(
            n_histogram=n_histogram,
            data_shape=data_shape,
            n_batches_emd_kernel=n_batches_emd_kernel,
            sg_feature_rank=sg_feature_rank,
        )
        super(GraphEMDExactGPModelSaasFullyBayesian, self).__init__(
            train_X=train_x, 
            train_Y=train_y,
            train_Yvar=None,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            pyro_model=pyro_model,
        )
