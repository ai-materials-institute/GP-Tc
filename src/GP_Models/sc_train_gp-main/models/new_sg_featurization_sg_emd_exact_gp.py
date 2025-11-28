import gpytorch
import sys
sys.path.append("../")
from kernels.new_sg_featurization_sg_emd_kernel import SgEMDKernelV2
from kernels.botorch_map_saas_prior_utils import add_saas_prior


class SgEmdExactGPModelV2(gpytorch.models.ExactGP):
    # supports n_histogram=0 --> sg features only
    # assumes for X's passed in, sg features come first, then hist features 
    def __init__(
        self, 
        train_x, 
        train_y, 
        likelihood, 
        n_sg,
        map_saas_tau,
        n_histogram=31, 
        data_shape=(31, 20, 2), 
        n_batches_emd_kernel=None,
        add_saas_ls_prior=False,
    ):
        super(SgEmdExactGPModelV2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.covar_module = SgEMDKernelV2(
            n_sg, # num sg features (sg features always first in X data)
            n_histogram=n_histogram,  # num histogram 
            data_shape=data_shape, # hist features data shape 
            n_batches=n_batches_emd_kernel,  # num batches for emd kernel 
        )
        if add_saas_ls_prior:
            self.covar_module = add_saas_prior(
                base_kernel=self.covar_module,
                tau=map_saas_tau,
            )

    def forward(self, x): 
        mean_f = self.mean_module(x)
        covar_histogram_and_sg_graph = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_f, covar_histogram_and_sg_graph )
