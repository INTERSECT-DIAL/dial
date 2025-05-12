"""This file is meant to consist of stateless functions that get called by the dialed service itself.

The idea is that these functions can easily be unit-tested (or called in a JupyterNotebook, etc.) without having to set up backing service logic.
"""

from .backends import get_backend_module
from .logger import logger
from .serverside_data import (
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .utilities.strategies import hypercube, random_in_bounds


# pure functional implementation of message, without MongoDB calls
def get_next_point(data: ServersideInputSingle) -> list[float]:
    """Trains a model, and then gets the next point for optimization based on the provided strategy.

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
    model = module.train_model(data)
    selected_point = module.sample(module, model, data)

    logger.debug('selected point with non-discrete measurements: %s', selected_point)
    return selected_point


# pure functional implementation of message, without MongoDB calls
def get_next_points(data: ServersideInputMultiple) -> list[list[float]]:
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
        case 'hypercube':
            output_points = hypercube(data.bounds, data.points, data.numpy_rng)
    return output_points


# pure functional implementation of message, without MongoDB calls
def get_surrogate_values(data: ServersideInputPrediction) -> list[list[float]]:
    """
    Get surrogate model predictions for given input points.

    Args:
        client_data (DialInputPredictions): Input data containing prediction points and model parameters.

    Returns:
        list[list[float]]: A list containing means, transformed standard deviations, and raw standard deviations.
    """
    backend = data.backend.lower()
    module = get_backend_module(backend)
    model = module.train_model(data)
    means, stddevs = module.predict(model, data)
    means = data.inverse_transform(means)
    transformed_stddevs = data.inverse_transform(stddevs, is_stddev=True)
    return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]
