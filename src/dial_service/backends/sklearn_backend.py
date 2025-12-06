"""NOTE: This file should not be imported in application code except dynamically via the get_backend_module function in __init__.py ."""

import inspect

import numpy as np
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Kernel,
    Matern,
    WhiteKernel,
)

from ..utilities import strategies
from . import AbstractBackend

_KERNELS_SKLEARN = {'rbf': RBF, 'matern': Matern, 'linear': DotProduct}

_SAMPLERS_SKLEARN = {
    'uncertainty': strategies.greedy_sampling,
    'upper_confidence_bound': strategies.greedy_sampling,
    'upper_confidence_bound_nomad': strategies.greedy_sampling,
    'expected_improvement': strategies.greedy_sampling,
    'confidence_bound': strategies.greedy_sampling,
    'polymer_acl_sampler': strategies.batch_sampling_acl,
}


def _filter_kwargs_for(cls, params: dict) -> dict:
    """Keep only kwargs that `cls.__init__` actually accepts."""
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters) - {'self', 'args', 'kwargs'}
    return {k: v for k, v in params.items() if k in allowed}


class SklearnBackend(
    AbstractBackend[GaussianProcessRegressor, Kernel, tuple[np.ndarray, np.ndarray]]
):
    @staticmethod
    def get_kernel(data):
        kernel_name = data.kernel.lower()
        if kernel_name not in _KERNELS_SKLEARN:
            msg = f'Unknown kernel {kernel_name}'
            raise ValueError(msg)
        _params = {} if data.kernel_args is None else data.kernel_args

        if 'length_scale' not in _params:
            length_per_dimension = (
                data.extra_args.get('length_per_dimension') if data.extra_args else False
            )
            _params['length_scale'] = [1.0] * len(data.X_train[0]) if length_per_dimension else 1.0

        base_kernel_cls = _KERNELS_SKLEARN[kernel_name]
        const_params = _filter_kwargs_for(ConstantKernel, _params)
        white_params = _filter_kwargs_for(WhiteKernel, _params)
        base_params = _filter_kwargs_for(base_kernel_cls, _params)

        constant_kernel = ConstantKernel(**const_params)
        base_kernel = base_kernel_cls(**base_params)
        white_kernel = WhiteKernel(**white_params)

        # TODO : Generalized expression from client
        return constant_kernel * base_kernel + white_kernel

    @staticmethod
    def train_model(data):
        """Create a model with training."""
        if data.backend_args is None:
            _extra_args = {}
        else:
            _extra_args = data.backend_args.copy()  # Ensure it's a dictionary
            if 'alpha' in _extra_args and not isinstance(_extra_args['alpha'], np.ndarray):
                # Process alpha as a numpy array
                _extra_args['alpha'] = np.array(_extra_args['alpha'])

        model = GaussianProcessRegressor(
            kernel=SklearnBackend.get_kernel(data), n_restarts_optimizer=1000, **_extra_args
        )
        model.fit(data.X_train, data.Y_train)
        return model

    @staticmethod
    def intialize_model(data):
        """Create a model without training."""
        if data.backend_args is None:
            _extra_args = {}
        else:
            _extra_args = data.backend_args.copy()  # Ensure it's a dictionary
            if 'alpha' in _extra_args and not isinstance(_extra_args['alpha'], np.ndarray):
                # Process alpha as a numpy array
                _extra_args['alpha'] = np.array(_extra_args['alpha'])

        return GaussianProcessRegressor(
            kernel=SklearnBackend.get_kernel(data), n_restarts_optimizer=1000, **_extra_args
        )

    @staticmethod
    def predict(model, data):
        dim = data.X_train.shape[1]

        derivative_type = data.extra_args.get('derivative_type', 0) if data.extra_args else 0
        if derivative_type == 0:
            means, stddevs = model.predict(data.x_predict.reshape(-1, dim), return_std=True)
            return means, data.stddev * stddevs

        if derivative_type == 2:
            means, stddevs = compute_posterior_f_double_prime(
                model, data.x_predict.reshape(-1, dim)
            )
            return means, stddevs

        msg = f'Order {derivative_type} is not supported. Supported orders are 0 and 2.'
        raise ValueError(msg)

    @staticmethod
    def sample(module, model, data):
        strategy_name = data.strategy.lower()

        if strategy_name not in _SAMPLERS_SKLEARN:
            msg = f'Unknown strategy {strategy_name}'
            raise ValueError(msg)

        return _SAMPLERS_SKLEARN[strategy_name](module, model, data)

    @staticmethod
    def samples(module, model, data):
        strategy_name = data.strategy.lower()

        if strategy_name not in _SAMPLERS_SKLEARN:
            msg = f'Unknown strategy {strategy_name}'
            raise ValueError(msg)

        samples = _SAMPLERS_SKLEARN[strategy_name](module, model, data)
        return [[float(x)] for x in samples]


def compute_posterior_f_double_prime(gpr, x_predict):
    """
    Compute the posterior mean and variance of the second derivative f''(x)
    using a trained GaussianProcessRegressor in sklearn.

    Parameters:
    - gpr: Trained sklearn GaussianProcessRegressor model
    - x_predict: Test points where f''(x) is evaluated (shape: NxD)

    Returns:
    - posterior_mean: Mean of f''(x) at x_predict
    - posterior_variance: Variance of f''(x) at x_predict
    """
    # Extract trained parameters
    kernel = gpr.kernel_

    if not isinstance(kernel, RBF):
        msg = 'Second derivative prediction is only implemented for the RBF kernel.'
        raise NotImplementedError(msg)

    X_train = gpr.X_train_
    alpha = gpr.alpha_
    length_scale = kernel.length_scale

    # Compute second derivative covariance using RBF kernel
    def rbf_second_derivative(x1, x2, length_scale):
        """Computes k''(x1, x2) (second derivative) for the RBF kernel"""
        diff = x1 - x2
        squared_distance = np.sum(diff**2)
        exp_term = np.exp(-0.5 * squared_distance / length_scale**2)
        return (squared_distance / length_scale**4 - 1 / length_scale**2) * exp_term

    def rbf_fourth_derivative(x1, x2, length_scale):
        """Computes k''''(x1, x2) (fourth derivative) for the RBF kernel"""
        diff = x1 - x2
        squared_distance = np.sum(diff**2)
        exp_term = np.exp(-0.5 * squared_distance / length_scale**2)

        term1 = (diff**4) / length_scale**8
        term2 = -6 * (diff**2) / length_scale**6
        term3 = 3 / length_scale**4
        return (term1 + term2 + term3) * exp_term

    # Compute k''(x_predict, X_train)
    k_2_0 = np.array(
        [
            [rbf_second_derivative(x_test, x_train, length_scale) for x_train in X_train]
            for x_test in x_predict
        ]
    )

    # Compute posterior mean: E[f''(x*) | X, y]
    posterior_mean = k_2_0 @ alpha

    # Compute prior covariance of f''(x_predict) (4th-order derivative)
    k_2_2 = np.array(
        [[rbf_fourth_derivative(x1, x2, length_scale) for x2 in x_predict] for x1 in x_predict]
    ).squeeze()
    # Compute posterior variance: Var[f''(x*) | X]
    K = gpr.L_ @ gpr.L_.T  # Get covariance matrix K(X, X)
    solved_term = sp.linalg.solve(K, k_2_0.T, assume_a='pos')  # Solve K^{-1} k_2_0
    posterior_variance = k_2_2 - (k_2_0 @ solved_term)
    posterior_sd = np.sqrt(np.diag(posterior_variance))

    return posterior_mean, posterior_sd
