from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel
from gpytorch.priors import Prior
from gpytorch.constraints import Positive
from gpytorch.utils.memoize import cached
import torch 
import ot 
import math 

class SgEMDKernelV2(Kernel):

    def __init__(
        self, 
        n_sg, # num sg features (sg features always first in X data)
        n_histogram=0,  # num histogram 
        data_shape=None, # hist features data shape 
        n_batches=None,  # num batches for emd kernel 
        weights_prior: Prior = None, 
        **kwargs,
    ):
        super(SgEMDKernelV2, self).__init__(has_lengthscale=False, **kwargs)
        self.n_sg = n_sg
        if n_sg > 0:
            self.sg_kernel = ScaleKernel(RBFKernel(ard_num_dims=self.n_sg))
        self.n_batches = n_batches
        self.data_shape = data_shape
        self.n_histogram = n_histogram
        # option for n_histogram = 0 --> just a regular RBF kernel over the sg features 
        if self.n_histogram > 0:
            self.raw_weights = torch.nn.Parameter(torch.rand(n_histogram)*0.1)
            self.raw_lengthscales = torch.nn.Parameter(torch.rand(n_histogram)*0.1)
            if weights_prior is not None:
                self.register_prior("weights_prior", weights_prior, lambda: self.weights, "weights")
            self.register_constraint("raw_weights", Positive())
            self.register_constraint("raw_lengthscales", Positive())

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        self._set_weights(value)

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

    @property
    def lengthscales(self):
        return self.raw_lengthscales_constraint.transform(self.raw_lengthscales)

    @lengthscales.setter
    def lengthscales(self, value):
        self._set_lengthscales(value)

    def _set_lengthscales(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_lengthscales=self.raw_lengthscales_constraint.inverse_transform(value))
    
    @cached
    def compute_emd_more_memory_efficient(
        self,
        X1: torch.Tensor, 
        X2: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Precompute EMD distances between all pairs of samples in X1 and X2.

        Parameters:
        -----------
        X1, X2 : torch tensor of shape (n_samples, n_histograms, max_n_bins, 2)
            Input data arrays.

        Returns:
        --------
        torch tensor of shape (n1_samples, n2_samples, n_histograms)
            Precomputed EMD distances.
        """
        
        X1 = X1.reshape(-1, *self.data_shape)
        X2 = X2.reshape(-1, *self.data_shape)
        X1_X2 = False 
        if torch.equal(X1,X2):
            X1_X2 = True 
        
        n1_samples = X1.shape[0]
        n2_samples = X2.shape[0]
        n_histograms = X1.shape[1]
        # XXX emd_matrix = torch.zeros((n1_samples, n2_samples, n_histograms),dtype=torch.float).to(X1.device)

        def get_valid_bin_lengths(h_array):
            # Determine the actual number of bins for each histogram type
            bin_lengths = []
            for h in range(n_histograms):
                valid_bins = torch.sum((h_array[:, h, :, 0] != -1.0) | (h_array[:, h, :, 1] != -1.0), axis=1)
                # Assume bins with midpoint -1 and count -1 are padding
                bin_lengths.append(int(valid_bins.max()))
            return bin_lengths

        # Determine the actual number of bins for each histogram type
        bin_lengths_1 = get_valid_bin_lengths(X1)
        bin_lengths_2 = get_valid_bin_lengths(X2)
        if (bin_lengths_1 != bin_lengths_2):
            raise ValueError(f"Inconsistent number of histogram bins")
        else:
            bin_lengths=bin_lengths_1

        X1_bins = X1[:, :, :, 0] 
        X1_counts = X1[:, :, :, 1] 
        X2_bins = X2[:, :, :, 0] 
        X2_counts = X2[:, :, :, 1] 
        # XXX emd_matrix[:, :, :] = 1e7
        for k in range(n_histograms):
            bin_len = bin_lengths[k]
            # x1 bins and counts 
            X1_bins_k = X1_bins[:,k]
            X1_counts_k = X1_counts[:,k]
            X1_counts_k = X1_counts_k / X1_counts_k.sum(dim=-1, keepdim=True)  
            X1_bins_k = X1_bins_k[:,:bin_len]
            X1_counts_k = X1_counts_k[:, :bin_len] 
            # x2 bins and counts 
            X2_bins_k = X2_bins[:,k]
            X2_counts_k = X2_counts[:,k]
            X2_counts_k = X2_counts_k / X2_counts_k.sum(dim=-1, keepdim=True)  
            X2_bins_k = X2_bins_k[:,:bin_len]
            X2_counts_k = X2_counts_k[:, :bin_len] 
            # repeat x1, x2 bins and counts to pass into OT 
            X1_bins_k_repeated = torch.cat([X1_bins_k]*n2_samples, 0 ) 
            X2_bins_k_repeated = X2_bins_k.repeat_interleave(n1_samples, 0)
            X1_counts_k_repeated = torch.cat([X1_counts_k]*n2_samples, 0 ) # GPU OOM HERE once with saas w/ too high num samples
            X2_counts_k_repeated = X2_counts_k.repeat_interleave(n1_samples, 0)
            # pass into OT to get wasserstein distances 
            if self.n_batches is None:
                emd = ot.wasserstein_1d(X1_bins_k_repeated.T, X2_bins_k_repeated.T, u_weights=X1_counts_k_repeated.T, v_weights=X2_counts_k_repeated.T) 
            else:
                n_inputs = X1_bins_k_repeated.shape[0]
                bsz = math.ceil(n_inputs/self.n_batches)
                bsz = max(bsz, 1) # double check bsz is non-zero 
                emds = []
                for batch_n in range(self.n_batches):
                    start_ix, stop_ix = batch_n*bsz, (batch_n+1)*bsz
                    X1_bins_batch = X1_bins_k_repeated[start_ix:stop_ix]
                    X2_bins_batch = X2_bins_k_repeated[start_ix:stop_ix]
                    X1_counts_batch = X1_counts_k_repeated[start_ix:stop_ix]
                    X2_counts_batch = X2_counts_k_repeated[start_ix:stop_ix]
                    emd_batch = ot.wasserstein_1d(X1_bins_batch.T, X2_bins_batch.T, u_weights=X1_counts_batch.T, v_weights=X2_counts_batch.T) 
                    emds.append(emd_batch)
                emd = torch.cat(emds) 
            emd = emd.reshape(n2_samples, n1_samples).T 
            if X1_X2:
                for i in range(emd.shape[0]): emd[i,i] = 0.0 
            emd_K_ix = torch.exp(-(emd / self.lengthscales[k])) * self.weights[k]
            if k == 0:
                emd_K = emd_K_ix
            else:
                emd_K = emd_K + emd_K_ix

        return emd_K 


    def forward(self, x1, x2, diag=False, **params):
        n_x1_dims = len(x1.shape)
        n_x2_dims = len(x2.shape)
        if (n_x1_dims != 2) or (n_x2_dims !=  2):
            x1 = x1.squeeze()
            x2 = x2.squeeze()  
            assert len(x1.shape) == 2
            assert len(x2.shape) == 2

        if self.n_sg > 0:
            kernel = self.sg_kernel(x1[:,0:self.n_sg], x2[:,0:self.n_sg])
            # if n hist = 0, using only sg features, all xs are the n_sg sg features 
            if self.n_histogram > 0:
                # multiply sg kernel by emd kernel
                kernel = kernel * self.compute_emd_more_memory_efficient(x1[:,self.n_sg:], x2[:,self.n_sg:])
        else:
            kernel = self.compute_emd_more_memory_efficient(x1[:,self.n_sg:], x2[:,self.n_sg:])

        if diag:
            return kernel.diagonal(dim1=-2, dim2=-1) # .unsqueeze(0)
        else:
            return kernel


