"""NOTE: This file should not be imported in application code except dynamically via the get_backend_module function in __init__.py ."""

import numpy as np
import scipy as sp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel, Matern

from . import AbstractBackend

_KERNELS_SKLEARN = {'rbf': RBF, 'matern': Matern}


class SklearnBackend(
    AbstractBackend[GaussianProcessRegressor, Kernel, tuple[np.ndarray, np.ndarray]]
):
    @staticmethod
    def get_kernel(data):
        kernel_name = data.kernel.lower()
        if kernel_name not in _KERNELS_SKLEARN:
            msg = f'Unknown kernel {kernel_name}'
            raise ValueError(msg)
        length_scale = [1.0] * len(data.X_train[0]) if data.length_per_dimension else 1.0
        return _KERNELS_SKLEARN[kernel_name](length_scale=length_scale)

    @staticmethod
    def train_model(data):
        model = GaussianProcessRegressor(
            kernel=SklearnBackend.get_kernel(data), n_restarts_optimizer=1000
        )
        model.fit(data.X_train, data.Y_train)
        return model

    @staticmethod
    def predict(model, data):
        dim = data.X_train.shape[1]
        means, stddevs = compute_posterior_f_double_prime(model, data.x_predict.reshape(-1, dim))
        return means, data.stddev * stddevs

    @staticmethod
    def sample(module, model, data):
        strategy_name = data.strategy.lower()

        sample_func = None
        match strategy_name:
            case (
                'uncertainty'
                | 'upper_confidence_bound'
                | 'expected_improvement'
                | 'confidence_bound'
            ):
                from dial_service.utilities.strategies import greedy_sampling

                sample_func = greedy_sampling
            case _:
                msg = f'Unknown strategy {strategy_name}'
                raise ValueError(msg)

        return sample_func(module, model, data)


def compute_posterior_f_double_prime(gpr: GaussianProcessRegressor, X_test: np.ndarray):
    """
    Compute the posterior mean and variance of the second derivative f''(x)
    using a trained GaussianProcessRegressor in sklearn.

    Parameters:
    - gpr: Trained sklearn GaussianProcessRegressor model
    - X_test: Test points where f''(x) is evaluated (shape: NxD)

    Returns:
    - posterior_mean: Mean of f''(x) at X_test
    - posterior_variance: Variance of f''(x) at X_test
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

    # Compute k''(X_test, X_train)
    k_2_0 = np.array(
        [
            [rbf_second_derivative(x_test, x_train, length_scale) for x_train in X_train]
            for x_test in X_test
        ]
    )

    # Compute posterior mean: E[f''(x*) | X, y]
    posterior_mean = k_2_0 @ alpha

    # Compute prior covariance of f''(X_test) (4th-order derivative)
    k_2_2 = np.array(
        [[rbf_fourth_derivative(x1, x2, length_scale) for x2 in X_test] for x1 in X_test]
    )
    k_2_2 = np.array(
        [[rbf_fourth_derivative(x1, x2, length_scale) for x2 in X_test] for x1 in X_test]
    ).squeeze()
    # Compute posterior variance: Var[f''(x*) | X]
    K = gpr.L_ @ gpr.L_.T  # Get covariance matrix K(X, X)
    solved_term = sp.linalg.solve(K, k_2_0.T, assume_a='pos')  # Solve K^{-1} k_2_0
    posterior_variance = k_2_2 - (k_2_0 @ solved_term)
    posterior_sd = np.sqrt(np.diag(posterior_variance))
    return posterior_mean, posterior_sd
