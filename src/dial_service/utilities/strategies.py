import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..backends import AbstractBackend
from ..serverside_data import (
    ServersideInputSingle,
)

logger = logging.getLogger(__name__)


def random_in_bounds(bounds: list[list[float]], rng: np.random.RandomState):
    return [rng.uniform(low, high) for low, high in bounds]


def get_negative_value_function(data):
    match data.strategy:
        case 'uncertainty':
            return lambda _mean, stddev: -stddev
        case 'expected_improvement':
            return lambda mean, stddev: _expected_improvement(mean, stddev, data)
        case 'upper_confidence_bound':
            return lambda mean, stddev: -(mean + 1 * stddev)
        case 'confidence_bound':
            z_value = norm.ppf(0.5 + data.confidence_bound / 2)
            return lambda mean, stddev: _confidence_bound(mean, stddev, z_value, data)
        case _:
            msg = f'Unknown strategy {data.strategy}'
            raise ValueError(msg)


def _expected_improvement(mean, stddev, data):
    if stddev == 0:
        return 0
    z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
    return -stddev * (z * norm.cdf(z) + norm.pdf(z))


def _confidence_bound(mean, stddev, z_value, data):
    return -z_value * stddev + mean * (-1 if data.y_is_good else 1)


def hypercube(
    bounds: list[list[float]], num_points: int, rng: np.random.RandomState
) -> list[list[float]]:
    coordinates = []
    for low, high in bounds:
        # for each dimension, generate a list of spaced coordinates and shuffle it:
        step = (high - low) / num_points
        coordinates.append(
            [rng.uniform(low + i * step, low + (i + 1) * step) for i in range(num_points)]
        )
        rng.shuffle(coordinates[-1])
    # add the points:
    return [list(point) for point in zip(*coordinates, strict=False)]


def _create_measurement_grid(data: ServersideInputSingle):
    """
    Create a grid of measurement points for discrete optimization.

    Args:
        data (ServersideInputBase): Input data containing bounds and grid size.

    Returns:
        list[list[float]]: A grid of measurement points.
    """
    row_values = np.linspace(
        data.bounds[0][0], data.bounds[0][1], data.discrete_measurement_grid_size[0]
    )
    col_values = np.linspace(
        data.bounds[1][0], data.bounds[1][1], data.discrete_measurement_grid_size[1]
    )
    return [[row, col] for row in row_values for col in col_values]


def greedy_sampling(backend_module: AbstractBackend, model, data: ServersideInputSingle):
    negative_value = get_negative_value_function(data)

    def to_minimize(_x: np.ndarray):
        mean, sigma = backend_module.predict(model, data)
        return negative_value(mean, sigma)

    if data.discrete_measurements:
        _measurement_grid = _create_measurement_grid(data)
        means, stddevs = backend_module.predict(model, _measurement_grid, data)
        response_surface = negative_value(means, stddevs)
        index = np.int64(np.argmin(response_surface))
        selected_point = _measurement_grid[index]
        logger.debug('selected point with discrete measurements')
        logger.debug(selected_point)
        return selected_point

    guess = min(
        np.array(hypercube(data.bounds, data.optimization_points, data.numpy_rng)), key=to_minimize
    )
    selected_point = minimize(to_minimize, guess, bounds=data.bounds, method='L-BFGS-B').x.tolist()
    logger.debug(selected_point)

    return selected_point
