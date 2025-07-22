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


def uncertainty_sampling(_mean, stddev, _data):
    return -stddev


def upper_confidence_bound(mean, stddev, data):
    _params = data.strategy_args
    y_is_good = data.y_is_good
    _direction = 1 if y_is_good else -1

    if _params is None:
        return _direction * mean + stddev

    return _direction * _params['exploit'] * mean + _params['explore'] * stddev

def upper_confidence_bound_nomad(mean, stddev, data):
    _params = data.strategy_args
    y_is_good = data.y_is_good
    _direction = 1 if y_is_good else -1

    _radius = .025
    _center = data.X_train[-1] + _radius/5
    _delta = (data.x_predict - _center) / _radius
    _delta = np.where(np.abs(_delta) < 1, 0.0, _delta)
    _distances = _delta**2
    _penalty_factor = np.exp(-0.02 * _distances).flatten()

    if _params is None:
        return _penalty_factor*(_direction * mean + stddev)

    return _penalty_factor*(_direction * _params['exploit'] * mean + _params['explore'] * stddev)

def expected_improvement(mean, stddev, data):
    _params = data.strategy_args
    y_is_good = data.y_is_good

    if stddev < 1e-8:
        return 0.0
    z = (mean - data.Y_best) / stddev * (1 if y_is_good else -1)
    return -stddev * (z * norm.cdf(z) + norm.pdf(z))


def confidence_bound(mean, stddev, data):
    y_is_good = data.y_is_good
    z_value = norm.ppf(0.5 + data.confidence_bound / 2)

    return -z_value * stddev + mean * (-1 if y_is_good else 1)


STRATEGIES = {
    'uncertainty': uncertainty_sampling,
    'upper_confidence_bound': upper_confidence_bound,
    'upper_confidence_bound_nomad': upper_confidence_bound_nomad,
    'expected_improvement': expected_improvement,
    'confidence_bound': confidence_bound,
}


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


def create_measurement_grid(data: ServersideInputSingle):
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
    try:
        strategy_ = STRATEGIES[data.strategy]
    except KeyError as exc:
        msg = f'Invalid strategy: {data.strategy}'
        raise ValueError(msg) from exc

    def to_minimize(_x: np.ndarray):
        data.x_predict = _x
        mean, sigma = backend_module.predict(model, data)
        return -strategy_(mean, sigma, data)

    if data.discrete_measurements:
        _measurement_grid = create_measurement_grid(data)
        means, stddevs = backend_module.predict(model, _measurement_grid, data)
        response_surface = -strategy_(means, stddevs)
        index = np.int64(np.argmin(response_surface))
        selected_point = _measurement_grid[index]
        logger.debug('selected point with discrete measurements')
        logger.debug(selected_point)
        return selected_point

    n_restarts = 10
    init_array = np.array(hypercube(data.bounds, n_restarts, data.numpy_rng))
    best_score = np.inf
    selected_point = None
    for x_init in init_array:
        res = minimize(
            to_minimize,
            x_init,
            bounds=data.bounds,
            options={'eps': 1e-6, 'gtol': 1e-10, 'ftol': 1e-12},
            method='L-BFGS-B',
        )
        if res.fun < best_score:
            best_score = res.fun
            selected_point = res.x

    return selected_point.tolist()
