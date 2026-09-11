import copy
import itertools
import logging
from numbers import Real

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from ..backends import AbstractBackend
from ..serverside_data import ServersideInputMultiple, ServersideInputSingle

logger = logging.getLogger(__name__)


def random_in_bounds(bounds: list[list[float]], rng: np.random.RandomState):
    return [rng.uniform(low, high) for low, high in bounds]


def uncertainty_sampling(_mean, stddev, _data):
    """Measure of uncertainty (stddev) for maximization"""
    return stddev


def upper_confidence_bound(mean, stddev, data):
    """Upper confidence bound for maximization
    If y_is_good = False, multiply mean by -1.
    """
    _params = data.strategy_args
    y_is_good = data.y_is_good
    _direction = 1 if y_is_good else -1

    if _params is None:
        return _direction * mean + stddev

    return _direction * _params['exploit'] * mean + _params['explore'] * stddev


def upper_confidence_bound_nomad(mean, stddev, data):
    """Upper confidence bound (NOMAD specific version) for maximization
    Masks the values around the last measurement point to force exploration.
    If y_is_good = False, multiply mean by -1.
    """
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
    """Expected Improvement (EI) for maximization
    If y_is_good = False, multiply mean and data value by -1.
    """
    _params = data.strategy_args
    y_is_good = data.y_is_good
    _direction = 1 if y_is_good else -1

    # guard against small or negative stddev
    stddev = np.maximum(stddev, 1e-15)

    z = (mean - data.Y_best) / stddev * _direction
    return stddev * (z * norm.cdf(z) + norm.pdf(z))


def confidence_bound(mean, stddev, data):
    """Confidence bound for maximization
    The same as upper_confidence_bound with exploit = 1., explore = norm.ppf(0.5 + data.confidence_bound / 2)
    If y_is_good = False, multiply mean by -1.
    """
    y_is_good = data.y_is_good
    _direction = 1 if y_is_good else -1
    z_value = norm.ppf(0.5 + data.confidence_bound / 2)

    return _direction * mean + z_value * stddev


STRATEGIES = {
    'uncertainty': uncertainty_sampling,
    'upper_confidence_bound': upper_confidence_bound,
    'upper_confidence_bound_nomad': upper_confidence_bound_nomad,
    'expected_improvement': expected_improvement,
    'confidence_bound': confidence_bound,
}


###############################################################################
# Surrogate-free indexed strategies


def domain_center(data, indices=None):  # noqa: ARG001
    return [[0.5 * (data.bounds[i][1] + data.bounds[i][0]) for i in range(data.dim_x)]]


def domain_corners(data, indices: list[int]):
    points = []
    for index in indices:
        # convert flat index into coordinate indices for each dimension
        coo_indices = np.unravel_index(index, [2] * data.dim_x)  # 2 corners per dimension
        points.append([data.bounds[i][coo_indices[i]] for i in range(data.dim_x)])
    return points


def uniform_grid(data, indices: list[int]):
    grid_size = data.strategy_args['grid_size']

    # grid spacing in each dimension
    steps = [
        (data.bounds[i][1] - data.bounds[i][0]) / max(1, grid_size[i] - 1)
        for i in range(data.dim_x)
    ]

    points = []
    for index in indices:
        # convert flat index into coordinate indices for each dimension
        coo_indices = np.unravel_index(index, grid_size)
        points.append(
            [
                data.bounds[i][0] + coo_indices[i] * steps[i]
                if grid_size[i] > 1
                else 0.5 * (data.bounds[i][1] + data.bounds[i][0])
                for i in range(data.dim_x)
            ]
        )
    return points


def chebyshev_grid(data, indices: list[int]):
    grid_size = data.strategy_args['grid_size']

    points = []
    for index in indices:
        # convert flat index into coordinate indices for each dimension
        coo_indices = np.unravel_index(index, grid_size)
        # Chebyshev nodes in each dimension, or the midpoint if only one point is specified
        x = [
            np.cos(coo_indices[i] * np.pi / (grid_size[i] - 1))
            if grid_size[i] > 1
            else 0.5 * (data.bounds[i][1] + data.bounds[i][0])
            for i in range(data.dim_x)
        ]
        points.append(
            [
                0.5 * (data.bounds[i][1] - data.bounds[i][0]) * (x[i] + 1) + data.bounds[i][0]
                for i in range(data.dim_x)
            ]
        )
    return points


def latin_hypercube(data, indices: list[int]):
    """For latin hypercube grid_size is the number of intervals per dimension"""
    grid_size = data.strategy_args['grid_size']

    points = []
    for index in indices:
        # convert flat index into interval indices for each dimension
        interval_indices = np.unravel_index(index, grid_size)
        points.append(
            [
                data.numpy_rng.uniform(
                    data.bounds[i][0]
                    + interval_indices[i] * (data.bounds[i][1] - data.bounds[i][0]) / grid_size[i],
                    data.bounds[i][0]
                    + (interval_indices[i] + 1)
                    * (data.bounds[i][1] - data.bounds[i][0])
                    / grid_size[i],
                )
                for i in range(data.dim_x)
            ]
        )
    return points


INDEXED_STRATEGIES = {
    'center': domain_center,
    'corners': domain_corners,
    'grid': uniform_grid,
    'chebyshev': chebyshev_grid,
    'latin_hypercube': latin_hypercube,
}

MAX_INDEXED_POINTS = {
    'center': lambda data: 1,  # noqa: ARG005
    'corners': lambda data: 2**data.dim_x,
    'grid': lambda data: np.prod(data.strategy_args['grid_size']),
    'chebyshev': lambda data: np.prod(data.strategy_args['grid_size']),
    'latin_hypercube': lambda data: np.prod(data.strategy_args['grid_size']),
}

###############################################################################


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


def create_measurement_grid(data: ServersideInputSingle | ServersideInputMultiple):
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


def greedy_sampling(
    backend_module: AbstractBackend, model, data: ServersideInputSingle | ServersideInputMultiple
):
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
        response_surface = to_minimize(_measurement_grid)
        index = np.int64(np.argmin(response_surface))
        selected_point = _measurement_grid[index]
        logger.debug('selected point with discrete measurements')
        logger.debug(selected_point)
        return selected_point

    n_restarts = 25
    init_array = np.array(hypercube(data.bounds, n_restarts, data.numpy_rng))
    best_score = np.inf
    selected_point = None
    # out_list = []
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
        # out_list.append((x_init.tolist(), res.x.tolist(), res.fun))

    logger.debug('selected point with optimization')
    logger.debug('score and point: %f, %s', best_score, str(selected_point))
    # print(f'optimized: {best_score}, {selected_point}:', '\n',
    #       '\n'.join([str(out) for out in out_list]))

    return selected_point.tolist()


def indexed_selection(data: ServersideInputSingle | ServersideInputMultiple):
    try:
        strategy_ = INDEXED_STRATEGIES[data.strategy]
    except KeyError as exc:
        msg = f'Invalid strategy: {data.strategy}'
        raise ValueError(msg) from exc

    start_index = data.strategy_args.get('start_index', 0) if data.strategy_args is not None else 0
    index = (len(data.dataset_y) - start_index) % MAX_INDEXED_POINTS[data.strategy](data)
    selected_point = strategy_(data, [index])[0]

    return selected_point


def _prepare_batch_sampling(backend_module: AbstractBackend, model, data: ServersideInputMultiple):
    if data.strategy in INDEXED_STRATEGIES:  # noqa: SIM108
        # for indexed strategies, a planning model is not needed.
        planning_model = None
    else:
        # make a deep copy of the model and build a predictor from it
        # making a deep copy allows to safely use backend_module.update_model,
        # without adding fake data
        planning_model = copy.deepcopy(model)

    # since we accept None in the dataclass, provide the believer as a default
    batch_strategy = data.batch_strategy or 'believer'

    if batch_strategy == 'liar':
        default_liar = 'mean'
        liar_setting = (
            data.strategy_args.get('liar_value', default_liar)
            if data.strategy_args is not None
            else default_liar
        )

        if isinstance(liar_setting, Real):
            # use a median liar value, and replace the y entry by the provided value
            liar_value = np.median(data.dataset_y, axis=0)
            liar_value[data.labels_y.index(data.statistics_y.loc)] = float(liar_setting)
        elif isinstance(liar_setting, list):
            liar_value = np.array(liar_setting, dtype=float).reshape((1, data.dim_y))
        elif isinstance(liar_setting, str):
            # data.dataset_y has shape (n_data, dim_y)
            # apply the strategies along axis 0 only
            match liar_setting:
                case 'mean':
                    liar_value = np.mean(data.dataset_y, axis=0)
                case 'max':
                    liar_value = np.max(data.dataset_y, axis=0)
                case 'min':
                    liar_value = np.min(data.dataset_y, axis=0)
                case 'random':
                    liar_value = data.numpy_rng.uniform(
                        np.min(data.dataset_y, axis=0), np.max(data.dataset_y, axis=0)
                    )
                case 'median':
                    liar_value = np.median(data.dataset_y, axis=0)
                case _:
                    liar_value = np.mean(data.dataset_y, axis=0)

        # this can not happen, how would we get a callable object through dial dataclass pydantic validation?
        # data.strategy_args is type dict[str, float | int | bool | list[int | float]]
        # elif callable(liar_setting):
        #    liar_value = liar_setting(data.dataset_y)

        def predictor(point: np.ndarray, model) -> np.ndarray:  # noqa: ARG001
            return liar_value

    elif batch_strategy == 'believer':
        # There is only the Kriging believer. In the future, there could be more
        believer_setting = (
            data.strategy_args.get('believer_type', 'kriging')
            if data.strategy_args is not None
            else 'kriging'
        )

        if believer_setting == 'kriging':
            # even for a believer strategy, we need to lie about the yerr values
            # could make this configureable, default to median right now
            liar_prediction = np.median(data.dataset_y, axis=0).reshape(data.dim_y)

            def expand_pred_to_dataset_y(mean: float):
                """Generate a dataset_y column from liar_prediction, with mean entry replaced by given mean."""
                dataset_y_col = liar_prediction.copy()
                pos_y = data.labels_y.index(data.statistics_y.loc)
                dataset_y_col[pos_y] = mean
                return dataset_y_col

            def predictor(point: np.ndarray, model) -> np.ndarray:
                """Return a raw dataset_y column predicted from the provided model at point."""
                # this will internally scale point from bounds to the unit cube
                data.set_x_predict(point)
                means, stddevs_ = backend_module.predict(model, data)
                # apply the inverse transform to get a 'raw' value suitable for dataset_y
                means, stddevs_ = data.inverse_transform_Y(means, stddevs_)
                return expand_pred_to_dataset_y(means.item())
        else:
            msg = f'Invalid beliver setting: {believer_setting}'
            raise ValueError(msg)
    else:
        msg = f'Invalid batch strategy: {data.batch_strategy}'
        raise ValueError(msg)

    return predictor, planning_model


def batch_sampling(
    backend_module: AbstractBackend, model, data: ServersideInputMultiple
) -> list[list[float]]:
    """
    Greedy batch selection using the liar or believer strategies.
    """

    if data.points <= 0:
        return []

    predictor, planning_model = _prepare_batch_sampling(backend_module, model, data)

    selected_points: list[list[float]] = []
    initial_x = data.dataset_x.copy()
    initial_y = data.dataset_y.copy()

    if 'predictor' not in locals():
        msg = f'Invalid batch strategy: {data.batch_strategy}'
        raise ValueError(msg)

    try:
        for _ in range(data.points):
            if data.strategy in INDEXED_STRATEGIES:
                next_point = indexed_selection(data)
            else:
                next_point = greedy_sampling(backend_module, planning_model, data)
            selected_points.append([float(v) for v in next_point])

            next_x = np.array(next_point, dtype=float).reshape(1, data.dim_x)
            next_y = predictor(next_x, model=planning_model).reshape(1, data.dim_y)

            data.dataset_x = np.concatenate([data.dataset_x, next_x])
            data.dataset_y = np.concatenate([data.dataset_y, next_y])

            if data.strategy not in INDEXED_STRATEGIES:
                # update_model updates the planning model, not modifying the original model
                planning_model = backend_module.update_model(planning_model, data)
    finally:
        # Restore original state so pseudo-observations never leak outside this method.
        data.dataset_x = initial_x
        data.dataset_y = initial_y

    return selected_points


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

    if data.dim_x > 1:
        msg = f'strategy batch_sampling_acl supports only one input dimension, but {data.dim_x=}'
        raise ValueError(msg)

    x_grid = create_measurement_grid(data)
    x_grid = np.array(x_grid)

    data.set_x_predict(x_grid)
    _, sd_dev = backend_module.predict(model, data)
    x_train = data.dataset_x  # get raw x data without scaling

    # set some default parameters
    _params = {
        'lambda_time': 0.0,
        'lambda_near_train': 1.0,
        'lambda_near_batch': 1.0,
        'lambda_batchT': 0.0,
        'radius_train_factor': 0.1,
        'radius_batch_factor': 0.1,
        'eps': 1.0e-3,
    }
    # update with provided values
    _params.update(data.strategy_args or {})

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
