import numpy as np
import random
import logging
from scipy.optimize import minimize

from ..serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .rewards import get_negative_value_function

logger = logging.getLogger(__name__)


def _random_in_bounds(data: ServersideInputBase):
    return [random.uniform(low, high) for low, high in data.bounds]  # noqa: S311
    # (TODO - probably OK to use cryptographically insecure random numbers, but check this first)


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
