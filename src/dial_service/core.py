"""This file is meant to consist of stateless functions that get called by the dialed service itself.

The idea is that these functions can easily be unit-tested (or called in a JupyterNotebook, etc.) without having to set up backing service logic.
"""

from typing import Any

import numpy as np

from .backends import get_backend_module
from .logger import logger
from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .utilities.strategies import (
    create_measurement_grid,
    hypercube,
    random_in_bounds,
)


# pure functional implementation of message, without MongoDB calls
def get_next_point(data: ServersideInputSingle, model: Any) -> list[float]:
    """Gets the next point for optimization based on the provided strategy.

    Model parameter should be a pretrained model, you can usually call core.train_model with the same data parameter if you don't yet have a model.

    Args:
        client_data (DialInputSingle): Input data containing bounds, strategy, and other parameters.

    Returns:
        list[float]: The selected point for the next iteration.
    """
    # If it's random point, we don't need to train a model or anything else
    if data.strategy == 'random':
        if data.discrete_measurements:
            _measurement_grid = create_measurement_grid(data)
            index = np.random.choice(len(_measurement_grid))
            # selected_point = data.numpy_rng.choice(_measurement_grid)
            selected_point = _measurement_grid[index]
            logger.debug('selected point with random strategy and discrete measurements')
            logger.debug(selected_point)
            return selected_point
        return random_in_bounds(data.bounds, data.numpy_rng)

    if data.strategy == 'hypercube':
        if not data.discrete_measurements:
            # use sane default
            data.discrete_measurement_grid_size = [4] * data.dim_x
        _measurement_grid = create_measurement_grid(data)
        index = data.point_index % len(_measurement_grid)
        selected_point = _measurement_grid[index]
        logger.debug('selected point with hypercube strategy')
        logger.debug(selected_point)
        return selected_point

    backend = data.backend.lower()
    module = get_backend_module(backend)
    selected_point = module.sample(module, model, data)

    logger.debug('selected point with non-discrete measurements: %s', selected_point)
    return selected_point


# pure functional implementation of message, without MongoDB calls
def get_next_points(data: ServersideInputMultiple, model: Any) -> list[list[float]]:
    """
    Get multiple next points for optimization based on the provided strategy.

    Args:
        client_data (DialInputMultiple): Input data containing bounds, strategy, and other parameters.

    Returns:
        list[list[float]]: A list of selected points for the next iteration.
    """
    # model = self._train_model(data) #this will be needed when we add qEI/constant liars
    output_points = None
    match data.strategy:
        case 'random':
            output_points = [
                random_in_bounds(data.bounds, data.numpy_rng) for _ in range(data.points)
            ]
            return output_points
        case 'hypercube':
            output_points = hypercube(data.bounds, data.points, data.numpy_rng)
            return output_points

    backend = data.backend.lower()
    module = get_backend_module(backend)
    output_points = module.samples(module, model, data)

    return output_points


# pure functional implementation of message, without MongoDB calls
def get_surrogate_values(
    data: ServersideInputPrediction, model: Any
) -> tuple[list[float], list[float], list[float], float]:
    """
    Get surrogate model predictions for given input points.

    Model parameter should be a pretrained model, you can usually call core.train_model with the same data parameter if you don't yet have a model.

    Args:
        client_data (DialInputPredictions): Input data containing prediction points and model parameters.

    Returns:
        tuple[list[float], list[float], list[float], float]: A tuple containing means, transformed standard deviations, raw standard deviations, and a float value.
    """
    backend = data.backend.lower()
    module = get_backend_module(backend)
    means, stddevs = module.predict(model, data)
    means = data.inverse_transform(means)
    transformed_stddevs = data.inverse_transform(stddevs, is_stddev=True)
    average = np.sqrt(np.mean(np.asarray(transformed_stddevs) ** 2))
    return (means.tolist(), transformed_stddevs.tolist(), stddevs.tolist(), float(average))


def train_model(data: ServersideInputBase) -> Any:
    """
    Trains a model and returns it
    """
    backend = data.backend.lower()
    module = get_backend_module(backend)
    return module.train_model(data)


def initialize_model(data: ServersideInputBase) -> Any:
    """
    Creates an untrained model and returns it
    """
    backend = data.backend.lower()
    module = get_backend_module(backend)
    return module.initialize_model(data)
