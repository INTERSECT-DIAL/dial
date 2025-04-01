"""This file is meant to consist of stateless functions that get called by the dialed service itself.

The idea is that these functions can easily be unit-tested (or called in a JupyterNotebook, etc.) without having to set up backing service logic.
"""

import gpax
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from .logger import logger
from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)

gpax.utils.enable_x64()

_KERNELS_SKLEARN = {'rbf': RBF, 'matern': Matern}
_KERNELS_GPAX = {'rbf': 'RBF', 'matern': 'matern'}


# trains a model based on the user's data
def _train_model(data: ServersideInputBase):  # -> GaussianProcessRegressor or -> gpax.ExactGP:
    backend_name = data.backend.lower()

    if backend_name == 'gpax':
        if data.seed == -1:
            rng_key_train, rng_key_predict = gpax.utils.get_keys()
        else:
            rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
        # Initialize and train a variational inference GP model
        gp_model = gpax.viGP(len(data.bounds), _kernel(data), guide='delta')
        gp_model.fit(
            rng_key_train,
            data.X_train,
            data.Y_train,
            num_steps=250,
            step_size=0.05,
            print_summary=False,
            progress_bar=False,
        )
        return gp_model

    model = GaussianProcessRegressor(kernel=_kernel(data), n_restarts_optimizer=250)
    model.fit(data.X_train, data.Y_train)
    return model


# parses the user's requested kernel
def _kernel(data: ServersideInputBase):
    kernel_name = data.kernel.lower()
    backend_name = data.backend.lower()

    if backend_name == 'gpax':
        if kernel_name not in _KERNELS_GPAX:
            msg = f'Unknown kernel {kernel_name}'
            raise ValueError(msg)
        return _KERNELS_GPAX[kernel_name]

    if kernel_name not in _KERNELS_SKLEARN:
        msg = f'Unknown kernel {kernel_name}'
        raise ValueError(msg)
    length_scale = [1.0] * len(data.X_train[0]) if data.length_per_dimension else 1.0
    return _KERNELS_SKLEARN[kernel_name](length_scale=length_scale)


# looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
def _create_n_dim_grid(data: ServersideInputBase, points_per_dim: int | list[int]):
    if isinstance(points_per_dim, int):
        points_per_dim = [points_per_dim] * len(data.bounds)
    meshgrid = np.meshgrid(
        *(
            np.linspace(low, high, pts)
            for (low, high), pts in zip(data.bounds, points_per_dim, strict=False)
        ),
        indexing='ij',
    )
    return np.column_stack([arr.flatten() for arr in meshgrid])


def _random_in_bounds(data: ServersideInputBase, rng: np.random.RandomState):
    return [rng.uniform(low, high) for low, high in data.bounds]


def _hypercube(
    data: ServersideInputBase, num_points: int, rng: np.random.RandomState
) -> list[list[float]]:
    coordinates = []
    for low, high in data.bounds:
        # for each dimension, generate a list of spaced coordinates and shuffle it:
        step = (high - low) / num_points
        coordinates.append(
            [rng.uniform(low + i * step, low + (i + 1) * step) for i in range(num_points)]
        )
        rng.shuffle(coordinates[-1])
    # add the points:
    return [list(point) for point in zip(*coordinates, strict=False)]


# pure functional implementation of message, without MongoDB calls
def internal_get_next_point(data: ServersideInputSingle) -> list[float]:
    rng = np.random.RandomState(None if data.seed == -1 else data.seed)

    # If it's random point, we don't need to train a model or anything else
    if data.strategy == 'random':
        return _random_in_bounds(data, rng)

    model = _train_model(data)

    # Generate the function that gives -1 * the "value" of measuring a point with the given mean & stddev
    negative_value = None
    match data.strategy:
        case 'uncertainty':

            def negative_value(_mean: float, stddev: float):
                return -stddev
        case 'expected_improvement':

            def negative_value(mean: float, stddev: float):
                if stddev == 0:
                    return 0

                z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
                return -stddev * (z * norm.cdf(z) + norm.pdf(z))
        case 'confidence_bound':
            # calculate the z score that corresponds to the confidence bound:
            Z_VALUE = norm.ppf(
                0.5 + data.confidence_bound / 2
            )  # need this because this is the "inverse CDF" which is one-tailed, but we want two-tailed

            def negative_value(mean: float, stddev: float):
                # y_is_good: maximize mean + z*stddev, so minimize -mean - z*stdev
                # y_is_bad:  minimize mean - z*stddev
                return -Z_VALUE * stddev + mean * (-1 if data.y_is_good else 1)

    # Create a function that can be minimized over the bounds:
    def to_minimize(x: np.ndarray):
        if data.backend == 'gpax':
            if data.seed == -1:
                rng_key_train, rng_key_predict = gpax.utils.get_keys()
            else:
                rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
            mean, y_var = model.predict(
                rng_key_predict, x.reshape(1, -1)
            )  # output is y_pred, y_var
            mean, sigma = (
                mean[0],
                data.stddev * y_var[0],
            )  # it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            return negative_value(mean, sigma)

        mean, sigma = model.predict(x.reshape(1, -1), return_std=True)
        mean, sigma = (
            mean[0],
            data.stddev * sigma[0],
        )  # it returns arrays, so fix that.  Also turn sigma into stddev of prediction
        return negative_value(mean, sigma)

    if data.discrete_measurements:
        measurement_grid = []
        row_values = np.linspace(
            data.bounds[0][0], data.bounds[0][1], data.discrete_measurement_grid_size[0]
        )
        col_values = np.linspace(
            data.bounds[1][0], data.bounds[1][1], data.discrete_measurement_grid_size[1]
        )
        for row in range(data.discrete_measurement_grid_size[0]):
            for col in range(data.discrete_measurement_grid_size[1]):
                measurement_grid.append([row_values[row], col_values[col]])

        backend_name = data.backend.lower()
        if backend_name == 'gpax':
            if data.seed == -1:
                rng_key_train, rng_key_predict = gpax.utils.get_keys()
            else:
                rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
            y_pred, y_var = model.predict(rng_key_predict, measurement_grid)
            stddevs = data.stddev * y_var  # turn sigma into stddev of prediction

        else:
            means, stddevs = model.predict(measurement_grid, return_std=True)
            stddevs *= data.stddev  # turn sigma into stddev of prediction

        response_surface = negative_value(means, stddevs)

        index = np.int64(np.argmin(response_surface))
        # selected point
        return measurement_grid[index]

    guess = min(np.array(_hypercube(data, data.optimization_points, rng)), key=to_minimize)
    selected_point = minimize(to_minimize, guess, bounds=data.bounds, method='L-BFGS-B').x.tolist()
    logger.debug('selected point with non-discrete measurements')
    logger.debug(selected_point)
    return selected_point


# pure functional implementation of message, without MongoDB calls
def internal_get_next_points(data: ServersideInputMultiple) -> list[list[float]]:
    rng = np.random.RandomState(None if data.seed == -1 else data.seed)
    # model = self._train_model(data) #this will be needed when we add qEI/constant liars
    output_points = None
    match data.strategy:
        case 'random':
            output_points = [_random_in_bounds(data, rng) for _ in range(data.points)]
        case 'hypercube':
            output_points = _hypercube(data, data.points, rng)
    return output_points


# pure functional implementation of message, without MongoDB calls
def internal_get_surrogate_values(data: ServersideInputPrediction) -> list[list[float]]:
    if data.backend == 'gpax':
        model = _train_model(data)
        if data.seed == -1:
            rng_key_train, rng_key_predict = gpax.utils.get_keys()
        else:
            rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
        y_pred, y_var = model.predict(rng_key_predict, data.x_predict)
        stddevs = data.stddev * y_var  # turn sigma into stddev of prediction
        # undo preprocessing:
        means = data.inverse_transform(y_pred)
        transformed_stddevs = data.inverse_transform(stddevs)
        return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]

    model = _train_model(data)
    means, stddevs = model.predict(data.x_predict, return_std=True)
    stddevs *= data.stddev  # turn sigma into stddev of prediction
    # undo preprocessing:
    means = data.inverse_transform(means)
    transformed_stddevs = data.inverse_transform(stddevs)
    return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]
