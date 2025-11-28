import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from kernels.botorch_map_saas_prior_utils import add_saas_prior
import sys
sys.path.append("../")
from kernels.emd_kernel import AdditiveEMDKernel


# https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html
class EmdGpClassificationModel(ApproximateGP):
    def __init__(
        self, 
        train_x, 
        n_histogram=21, 
        data_shape=(21, 20, 2), 
        n_batches_emd_kernel=None,
        map_saas_tau=None,
        add_saas_ls_prior=False,
    ):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(EmdGpClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = AdditiveEMDKernel(
            n_histogram=n_histogram, 
            data_shape=data_shape,
            n_batches=n_batches_emd_kernel,
            unsqueeze_diag=False,
        )
        if add_saas_ls_prior:
            self.covar_module = add_saas_prior(
                base_kernel=self.covar_module,
                tau=map_saas_tau,
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


