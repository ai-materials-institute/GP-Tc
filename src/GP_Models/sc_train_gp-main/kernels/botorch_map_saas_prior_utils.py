# Map saas code copied from: https://github.com/pytorch/botorch/blob/main/botorch/models/map_saas.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
# from botorch.exceptions import UnsupportedError
# from botorch.models.gp_regression import SingleTaskGP
# from botorch.models.transforms.input import InputTransform
# from botorch.models.transforms.outcome import OutcomeTransform
# from botorch.utils.constraints import LogTransformedInterval
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel # , AdditiveKernel, MaternKernel, ScaleKernel
# from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
# from gpytorch.means import ConstantMean
from gpytorch.priors import HalfCauchyPrior # , NormalPrior, GammaPrior
from torch import Tensor
# from torch.distributions.half_cauchy import HalfCauchy
from torch.nn import Parameter


EPS = 1e-8


class SaasPriorHelper:
    """Helper class for specifying parameter and setting closures."""

    def __init__(self, tau: float | None = None):
        """Instantiates a new helper object.

        Args:
            tau: Value of the global shrinkage parameter. If `None`, the tau will be
                a free parameter and inferred from the data.
        """
        self._tau = torch.as_tensor(tau) if tau is not None else None

    def tau(self, m: Kernel) -> Tensor:
        """The global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a lengthscales.

        Returns:
            The global shrinkage parameter of the SAAS prior.
        """
        return (
            self._tau.to(m.lengthscales)
            if self._tau is not None
            else m.raw_tau_constraint.transform(m.raw_tau)
        )

    def inv_lengthscales_prior_param_or_closure(self, m: Kernel) -> Tensor:
        """Closure to compute the scaled inverse lengthscales parameter (`tau / l^2`)
        to which the SAAS prior is applied.

        Args:
            m: A kernel object equipped with a lengthscales.

        Returns:
            The scaled inverse lengthscales parameter.
        """
        tau = self.tau(m)
        return tau.view(*tau.shape, 1, 1) / (m.lengthscales**2)

    def inv_lengthscales_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        """Closure to set the inverse lengthscales prior parameter.

        Args:
            m: A kernel object equipped with a lengthscales.
            value: The value of the scaled inverse lengthscales parameter, (`tau / l^2`),
                used to recover and set the lengthscales of the kernel.
        """
        # Lengthscales is batch x m x 1 x d, update tau to avoid unwanted broadcasting.
        tau = self.tau(m)
        tau = tau.view(*tau.shape, 1, 1)
        lb = m.raw_lengthscales_constraint.lower_bound.to(tau)
        ub = m.raw_lengthscales_constraint.upper_bound.to(tau)
        m._set_lengthscales((tau / value.to(tau)).sqrt().clamp(lb + EPS, ub - EPS))

    def tau_prior_param_or_closure(self, m: Kernel) -> Tensor:
        """Closure to compute the global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a `raw_tau` parameter.

        Returns:
            The transformed global shrinkage parameter `tau`.
        """
        return m.raw_tau_constraint.transform(m.raw_tau)

    def tau_prior_setting_closure(self, m: Kernel, value: Tensor) -> None:
        """Closure to set the global shrinkage parameter `tau`.

        Args:
            m: A kernel object equipped with a `raw_tau` parameter.
            value: The value of the global shrinkage parameter.
        """
        lb = m.raw_tau_constraint.lower_bound.to(m.raw_tau)
        ub = m.raw_tau_constraint.upper_bound.to(m.raw_tau)
        m.raw_tau.data.fill_(
            m.raw_tau_constraint.inverse_transform(
                value.to(m.raw_tau).clamp(lb + EPS, ub - EPS)
            ).item()
        )


def add_saas_prior(
    base_kernel: Kernel,
    tau: float | None = None,
    # log_scale: bool = True,
) -> Kernel:
    """Add a SAAS prior to a given base_kernel.

    The SAAS prior is given by tau / lengthscales^2 ~ HC(1.0). If tau is None,
    we place an additional HC(0.1) prior on tau similar to the original SAAS prior
    that relies on inference with NUTS.

    Example:
        >>> matern_kernel = MaternKernel(...)
        >>> add_saas_prior(matern_kernel, tau=None)  # Add a SAAS prior

    Args:
        base_kernel: Base kernel that has a lengthscales and uses ARD.
            Note that this function modifies the kernel object in place.
        tau: Value of the global shrinkage. If `None`, infer the global
            shrinkage parameter.
        log_scale: Set to `True` if the lengthscales and tau should be optimized on
            a log-scale without any domain rescaling. That is, we will learn
            `raw_lengthscales := log(lengthscales)` and this hyperparameter needs to
            satisfy the corresponding bound constraints. Setting this to `True` will
            generally improve the numerical stability, but requires an optimizer that
            can handle bound constraints, e.g., L-BFGS-B.

    Returns:
        Base kernel with SAAS priors added.
    """
    tkwargs = {"device": base_kernel.device, "dtype": base_kernel.dtype}

    batch_shape = base_kernel.raw_lengthscales.shape[:-2] # *** AttributeError: 'AdditiveEMDKernel' object has no attribute 'raw_lengthscales'
    # IntervalClass = LogTransformedInterval if log_scale else Interval # <class 'botorch.utils.constraints.LogTransformedInterval'>
    IntervalClass = Interval
    base_kernel.register_constraint(
        param_name="raw_lengthscales",
        constraint=IntervalClass(0.01, 1e4, initial_value=1),
        replace=True,
    ) # *** RuntimeError: Attempting to register constraint for nonexistent parameter.
    prior_helper = SaasPriorHelper(tau=tau)
    if tau is None:  # Place a HC(0.1) prior on tau
        base_kernel.register_parameter(
            name="raw_tau",
            parameter=Parameter(torch.full(batch_shape, 0.1, **tkwargs)),
        )
        base_kernel.register_constraint(
            param_name="raw_tau",
            constraint=IntervalClass(1e-3, 10, initial_value=0.1),
            replace=True,
        )
        base_kernel.register_prior(
            name="tau_prior",
            prior=HalfCauchyPrior(torch.tensor(0.1, **tkwargs)),
            param_or_closure=prior_helper.tau_prior_param_or_closure,
            setting_closure=prior_helper.tau_prior_setting_closure,
        )
    # Place a HC(1) prior on tau / lengthscales^2
    base_kernel.register_prior(
        name="inv_lengthscales_prior",
        prior=HalfCauchyPrior(torch.tensor(1.0, **tkwargs)),
        param_or_closure=prior_helper.inv_lengthscales_prior_param_or_closure,
        setting_closure=prior_helper.inv_lengthscales_prior_setting_closure,
    )
    return base_kernel

