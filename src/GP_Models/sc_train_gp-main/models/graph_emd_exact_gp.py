import gpytorch
import sys
sys.path.append("../")
from kernels.graph_emd_kernel import AdditiveSgEMDKernel
from kernels.botorch_map_saas_prior_utils import add_saas_prior

class SgEmdExactGP(gpytorch.models.ExactGP):
    def __init__(
        self, 
        train_x, 
        train_y, 
        likelihood, 
        sg_feature_rank, 
        map_saas_tau,
        n_histogram=31, 
        data_shape=(31, 20, 2), 
        n_batches_emd_kernel=None,
        add_saas_ls_prior=False,
        use_11d_symm_sg_features=False,
    ):
        super(SgEmdExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = AdditiveSgEMDKernel(
            n_histogram=n_histogram, 
            data_shape=data_shape,
            n_batches=n_batches_emd_kernel,
            sg_feature_rank=sg_feature_rank,
            use_11d_symm_sg_features=use_11d_symm_sg_features,
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
