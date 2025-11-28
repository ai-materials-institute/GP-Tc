import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from kernels.new_sg_featurization_sg_emd_kernel import SgEMDKernelV2
import sys
sys.path.append("../")

# https://docs.gpytorch.ai/en/v1.6.0/examples/04_Variational_and_Approximate_GPs/Non_Gaussian_Likelihoods.html
class EmdSgGpClassificationModel(ApproximateGP):
    def __init__(
        self, 
        train_x, 
        n_histogram=21, 
        n_sg=11,
        data_shape=(21, 20, 2), 
        n_batches_emd_kernel=None,
    ):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(EmdSgGpClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SgEMDKernelV2(
            n_sg=n_sg, # num sg features (sg features always first in X data)
            n_histogram=n_histogram,  # num histogram 
            data_shape=data_shape, # hist features data shape 
            n_batches=n_batches_emd_kernel,  # num batches for emd kernel 
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


