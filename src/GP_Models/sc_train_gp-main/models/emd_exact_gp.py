import gpytorch
import sys
sys.path.append("../")
from kernels.emd_kernel import AdditiveEMDKernel
from kernels.botorch_map_saas_prior_utils import add_saas_prior

class EMDExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self, 
        train_x, 
        train_y, 
        likelihood, 
        map_saas_tau,
        n_histogram=31, 
        data_shape=(31, 20, 2), 
        n_batches_emd_kernel=None,
        add_saas_ls_prior=False,
    ):
        super(EMDExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = AdditiveEMDKernel(
            n_histogram=n_histogram, 
            data_shape=data_shape,
            n_batches=n_batches_emd_kernel,
        )
        if add_saas_ls_prior:
            self.covar_module = add_saas_prior(
                base_kernel=self.covar_module,
                tau=map_saas_tau,
            )

    def forward(self, x): 
        mean_f = self.mean_module(x)
        covar_histogram = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_f, covar_histogram)
