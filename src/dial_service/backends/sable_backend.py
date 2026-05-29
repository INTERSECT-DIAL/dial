"""NOTE: This file should not be imported in application code except dynamically via the get_backend_module function in __init__.py ."""

import numpy as np
from sable import DiscretizedSurrogateModel, ScaledRBFModel

from dial_dataclass import Normal

from ..utilities import strategies
from . import AbstractBackend

_KERNELS_SABLE = {'rbf': ScaledRBFModel}

_SAMPLERS_SABLE = {
    'uncertainty': strategies.greedy_sampling,
    'upper_confidence_bound': strategies.greedy_sampling,
    'upper_confidence_bound_nomad': strategies.greedy_sampling,
    'expected_improvement': strategies.greedy_sampling,
    'confidence_bound': strategies.greedy_sampling,
    'polymer_acl_sampler': strategies.batch_sampling_acl,
}


def _get_model_kwargs(data) -> dict:
    backend_args = {} if data.backend_args is None else data.backend_args
    return {
        'n_features': backend_args.get('n_features', 10000),
        'alpha': backend_args.get('alpha', 0.3),
        'p': backend_args.get('p', 1.25),
        'n_iter_irls': backend_args.get('n_iter_irls', 100),
    }


def _get_observation_errors(data, n_observations: int) -> np.ndarray:
    backend_args = {} if data.backend_args is None else data.backend_args

    if isinstance(data.statistics_y, Normal):
        y_err = data.Yerr_train
        y_err_arr = np.asarray(y_err, dtype=float).reshape(-1)
        if y_err_arr.size == 1:
            y_err_arr = np.full(n_observations, float(y_err_arr[0]), dtype=float)
    else:
        # if y_err is not provided through the statistics, use the old fallback for compatibility
        # TODO: remove if no longer needed
        y_err = backend_args.get('y_err', backend_args.get('noise_level', 1e-6))

        y_err_arr = np.asarray(y_err, dtype=float).reshape(-1)
        if y_err_arr.size == 1:
            y_err_arr = np.full(n_observations, float(y_err_arr[0]), dtype=float)

        if data.preprocess_standardize and len(data.Y_raw) > 0:
            scale = np.std(np.asarray(data.Y_raw, dtype=float))
            if scale > 0:
                y_err_arr = y_err_arr / scale

    return y_err_arr


class SABLEBackend(
    AbstractBackend[DiscretizedSurrogateModel, ScaledRBFModel, tuple[np.ndarray, np.ndarray]]
):
    @staticmethod
    def get_kernel(data):
        kernel_args = {} if data.kernel_args is None else data.kernel_args
        return _KERNELS_SABLE[data.kernel.lower()](
            x_dimension=data.dim_x,
            x_range=kernel_args.get('x_range', (0.0, 1.0)),
            sigma_range=kernel_args.get('sigma_range', (1e-2, 1.0)),
            gamma=kernel_args.get('gamma', 0.01),
        )

    @staticmethod
    def train_model(data):
        """Create and train a SABLE surrogate model."""
        model = SABLEBackend.initialize_model(data)
        y_err = _get_observation_errors(data, len(data.Y_train))
        model.fit(data.X_train, data.Y_train, y_err=y_err)
        return model

    @staticmethod
    def initialize_model(data):
        """Create a SABLE surrogate model without training."""
        return DiscretizedSurrogateModel(
            featuremodel=SABLEBackend.get_kernel(data),
            **_get_model_kwargs(data),
        )

    @staticmethod
    def predict(model, data):
        x_query = np.asarray(data.x_predict, dtype=float).reshape(-1, data.dim_x)
        means, stddevs = model.predict(x_query)
        return np.asarray(means, dtype=float).reshape(-1), np.asarray(stddevs, dtype=float).reshape(
            -1
        )

    @staticmethod
    def sample(module, model, data):
        return _SAMPLERS_SABLE[data.strategy.lower()](module, model, data)

    @staticmethod
    def samples(module, model, data):
        samples = _SAMPLERS_SABLE[data.strategy.lower()](module, model, data)
        return [[float(x)] for x in samples]
