"""This file is meant to consist of stateless functions that get called by the dialed service itself.

The idea is that these functions can easily be unit-tested (or called in a JupyterNotebook, etc.) without having to set up backing service logic.
"""

from typing import Any

from .backends import get_backend_module
from .logger import logger
from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .utilities.strategies import hypercube, random_in_bounds


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
        return random_in_bounds(data.bounds, data.numpy_rng)

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

    return output_points  # noqa: RET504


# pure functional implementation of message, without MongoDB calls
def get_surrogate_values(data: ServersideInputPrediction, model: Any) -> list[list[float]]:
    """
    Get surrogate model predictions for given input points.

    Model parameter should be a pretrained model, you can usually call core.train_model with the same data parameter if you don't yet have a model.

    Args:
        client_data (DialInputPredictions): Input data containing prediction points and model parameters.

    Returns:
        list[list[float]]: A list containing means, transformed standard deviations, and raw standard deviations.
    """
    backend = data.backend.lower()
    module = get_backend_module(backend)
    means, stddevs = module.predict(model, data)
    means = data.inverse_transform(means)
    transformed_stddevs = data.inverse_transform(stddevs, is_stddev=True)
    return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]


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
