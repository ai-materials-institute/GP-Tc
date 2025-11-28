import numpy as np
from typing import List

class Histo_Array_Scaler:
    """
    Scaler for hist_array to standardize bin midpoints across samples.

    This class computes the mean and standard deviation of the bin midpoints
    for each histogram type and scales the bin midpoints accordingly.

    Attributes:
    -----------
    mean_ : np.ndarray
        Mean of the bin midpoints for each histogram type.
    std_ : np.ndarray
        Standard deviation of the bin midpoints for each histogram type.
    """

    def fit(self, hist_array, bin_lengths: List[int] = None):
        """
        Compute the mean and std to be used for later scaling.

        Parameters:
        -----------
        hist_array : np.ndarray of shape (n_samples, n_histograms, max_n_bins, 2)
            The data used to compute the mean and standard deviation
            used for later scaling along the bin midpoints.

        Returns:
        --------
        self : object
            Fitted scaler.
        """
        n_samples, n_histograms, max_n_bins, _ = hist_array.shape
        self.mean_ = np.zeros(n_histograms)
        self.std_ = np.zeros(n_histograms)
        self.n_histograms=n_histograms
        self.bin_lengths = bin_lengths
        
        
        # Determine the actual number of bins for each histogram type
        if(self.bin_lengths== None):
            self.bin_lengths = self.get_valid_bin_lengths(hist_array)
            
        if (self.n_histograms != len(self.bin_lengths)):
            raise ValueError(f"Inconsistent number of histograms")


        for h in range(self.n_histograms):
            bin_midpoints = hist_array[:, h, 0:self.bin_lengths[h], 0] 
            bin_heights=hist_array[:, h, 0:self.bin_lengths[h], 1] 
            self.mean_[h] = np.sum(bin_midpoints*bin_heights)/np.sum(bin_heights)

            self.std_[h] = np.sqrt(np.sum((bin_midpoints**2)*bin_heights)/np.sum(bin_heights) -self.mean_[h]**2)
        return self

    def transform(self, hist_array):
        """
        Perform standardization by centering and scaling bin midpoints.

        Parameters:
        -----------
        hist_array : np.ndarray of shape (n_samples, n_histograms, max_n_bins, 2)
            The data that should be transformed.

        Returns:
        --------
        hist_array_scaled : np.ndarray
            Scaled data.
        """
        n_samples, n_histograms, max_n_bins, _ = hist_array.shape
        if(self.n_histograms!=n_histograms):
            raise ValueError(f"wrong histogram number in train: {self.n_histograms}, and test: {n_histograms}")


        hist_array_scaled = hist_array.copy()
        bin_lengths=self.get_valid_bin_lengths(hist_array)
        for h in range(n_histograms):
            bin_len=bin_lengths[h]
            bin_midpoints = hist_array[:, h, 0:bin_len, 0]
            hist_array_scaled[:, h, 0:bin_len, 0] = (bin_midpoints - self.mean_[h]) / self.std_[h]
        return hist_array_scaled

    
    
    
    def fit_transform(self, hist_array):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        hist_array : np.ndarray
            The data to fit, then transform.

        Returns:
        --------
        hist_array_scaled : np.ndarray
            Transformed data.
        """
        return self.fit(hist_array).transform(hist_array)

    def get_valid_bin_lengths(self,h_array):
        # Determine the actual number of bins for each histogram type
        bin_lengths = []
        for h in range(self.n_histograms):
            # Assume bins with midpoint -1 and count -1 are padding
            valid_bins = np.sum((h_array[:, h, :, 0] != -1.0) | (h_array[:, h, :, 1] != -1.0), axis=1)
            bin_lengths.append(int(valid_bins.max()))
        return bin_lengths