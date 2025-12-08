import itertools
import logging

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..backends import AbstractBackend
from ..serverside_data import ServersideInputMultiple, ServersideInputSingle

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

    _radius = 0.025
    _center = data.X_train[-1] + _radius / 5
    _delta = (data.x_predict - _center) / _radius
    _delta = np.where(np.abs(_delta) < 1, 0.0, _delta)
    _distances = _delta**2
    _penalty_factor = np.exp(-0.02 * _distances).flatten()

    if _params is None:
        return _penalty_factor * (_direction * mean + stddev)

    return _penalty_factor * (_direction * _params['exploit'] * mean + _params['explore'] * stddev)


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
    axes = [
        np.linspace(low, high, n)
        for (low, high), n in zip(data.bounds, data.discrete_measurement_grid_size, strict=False)
    ]

    # 2. Cartesian product → grid points
    return [list(point) for point in itertools.product(*axes)]


def greedy_sampling(backend_module: AbstractBackend, model, data: ServersideInputSingle):
    try:
        strategy_ = STRATEGIES[data.strategy]
    except KeyError as exc:
        msg = f'Invalid strategy: {data.strategy}'
        raise ValueError(msg) from exc

    def to_minimize(_x: np.ndarray):
        data.set_x_predict(_x)
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


def batch_sampling_acl(backend_module: AbstractBackend, model, data: ServersideInputMultiple):
    """
    Greedy batch selection using GP std and multiple penalties:

      score(t) =
          sd_dev(t)
        - lambda_time * (t / t_max)^2
        - lambda_near_train * near_train_penalty(t)
        - lambda_near_batch * near_batch_penalty(t)
        - lambda_batchT * ΔT(t) / t_max

    Where:
      - near_train_penalty(t) is large if t is within radius_train of x_train
      - near_batch_penalty(t) is large if t is within radius_batch of already chosen batch points
      - ΔT(t) = max(current_batch_t_max, t) - current_batch_t_max (parallel reactor cost)
    """

    x_grid = create_measurement_grid(data)
    x_grid = np.array(x_grid)

    data.set_x_predict(x_grid)
    mean, sd_dev = backend_module.predict(model, data)
    x_train = data.X_raw
    _params = data.strategy_args

    batch_size = data.points
    lambda_time = _params['lambda_time']  # penalty on large t
    lambda_near_train = _params['lambda_near_train']  # penalty on being close to existing points
    lambda_near_batch = _params['lambda_near_batch']  # penalty on being close to other batch points
    lambda_batchT = _params[
        'lambda_batchT'
    ]  # penalty on extending max t in batch (parallel reactors)
    radius_train_factor = _params['radius_train_factor']  # neighborhood size as fraction of t_max
    radius_batch_factor = _params['radius_batch_factor']
    eps = _params['eps']

    xg = x_grid.ravel()  # shape (N,)
    xt = x_train.ravel()  # shape (n,)
    t_max = np.max(xg)

    # Avoid divide-by-zero if t_max == 0
    if t_max <= 0:
        t_max = 1.0

    # Precompute distance to existing training points
    if xt.size > 0:
        # (N, n) distances -> min over n
        dist_to_train = np.min(np.abs(xg[:, None] - xt[None, :]), axis=1)
    else:
        dist_to_train = np.full_like(xg, fill_value=t_max)

    radius_train = radius_train_factor * t_max
    radius_batch = radius_batch_factor * t_max

    # Base mask: exclude points already in training set
    base_mask = np.ones_like(xg, dtype=bool)
    for x in xt:
        base_mask &= np.abs(xg - x) > eps

    batch_idx = []
    current_t_max = 0.0

    for _ in range(batch_size):
        # Start from allowed candidates (not in training)
        mask = base_mask.copy()
        # Also exclude already chosen batch points
        for j in batch_idx:
            mask &= np.abs(xg - xg[j]) > eps

        if not np.any(mask):
            break  # nothing left to pick

        # --- penalties shared across candidates ---

        # 1) Time penalty: larger t → heavier penalty (quadratic)
        penalty_time = lambda_time * (xg / t_max) ** 2

        # 2) Penalty for being close to existing training points
        #    Linear ramp inside radius_train, 0 outside
        if radius_train > 0:
            near_train = np.maximum(0.0, (radius_train - dist_to_train) / radius_train)
        else:
            near_train = 0.0
        penalty_train = lambda_near_train * near_train

        # 3) Penalty for extending batch max time (parallel reactors)
        delta_T = np.maximum(current_t_max, xg) - current_t_max
        penalty_batchT = lambda_batchT * (delta_T / t_max)

        # 4) Penalty for being close to existing batch points (diversity term)
        if batch_idx:
            dist_to_batch = np.min(
                np.abs(xg[:, None] - xg[np.array(batch_idx)][None, :]),
                axis=1,
            )
            if radius_batch > 0:
                near_batch = np.maximum(0.0, (radius_batch - dist_to_batch) / radius_batch)
            else:
                near_batch = 0.0
            penalty_batch = lambda_near_batch * near_batch
        else:
            penalty_batch = 0.0

        # --- final score ---
        score = sd_dev - penalty_time - penalty_train - penalty_batch - penalty_batchT

        # Remove invalid points from consideration
        score[~mask] = -np.inf

        j_star = np.argmax(score)
        if not np.isfinite(score[j_star]):
            break

        batch_idx.append(j_star)
        current_t_max = max(current_t_max, xg[j_star])

    return xg[batch_idx]
