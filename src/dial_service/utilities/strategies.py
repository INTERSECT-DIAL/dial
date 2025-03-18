import logging
import random
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from ..serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)

logger = logging.getLogger(__name__)

def get_negative_value_function(data):
    match data.strategy:
        case 'uncertainty':
            return lambda mean, stddev: -stddev
        case 'expected_improvement':
            return lambda mean, stddev: _expected_improvement(mean, stddev, data)
        case 'upper_confidence_bound':
            return lambda mean, stddev: -(mean + 1*stddev)
        case 'confidence_bound':
            z_value = norm.ppf(0.5 + data.confidence_bound / 2)
            return lambda mean, stddev: _confidence_bound(mean, stddev, z_value, data)
        case _:
            raise ValueError(f"Unknown strategy {data.strategy}")

def _expected_improvement(mean, stddev, data):
    if stddev == 0:
        return 0
    z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
    return -stddev * (z * norm.cdf(z) + norm.pdf(z))

def _confidence_bound(mean, stddev, z_value, data):
    return -z_value * stddev + mean * (-1 if data.y_is_good else 1)


def _hypercube(data: ServersideInputBase, num_points) -> list[list[float]]:
    coordinates = []
    for low, high in data.bounds:
        # for each dimension, generate a list of spaced coordinates and shuffle it:
        step = (high - low) / num_points
        coordinates.append(
            [random.uniform(low + i * step, low + (i + 1) * step) for i in range(num_points)]  # noqa: S311
            # (TODO - probably OK to use cryptographically insecure random numbers, but check this first)
        )
        random.shuffle(coordinates[-1])
    # add the points:
    return [list(point) for point in zip(*coordinates, strict=False)]


def _create_measurement_grid(data):
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


def greedy_sampling(backend_module, model, data):
    negative_value = get_negative_value_function(data)

    def to_minimize(x: np.ndarray):
        mean, sigma = backend_module.predict(model, x, data)
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

    guess = min(np.array(_hypercube(data, data.optimization_points)), key=to_minimize)
    selected_point = minimize(to_minimize, guess, bounds=data.bounds, method='L-BFGS-B').x.tolist()

    return selected_point
