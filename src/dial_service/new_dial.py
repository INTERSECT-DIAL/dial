import logging
import random
import numpy as np
from scipy.optimize import minimize

from ..dial_dataclass import DialInputMultiple, DialInputPredictions, DialInputSingle
from intersect_sdk import IntersectBaseCapabilityImplementation, intersect_message, intersect_status

from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)
from .utilities.sampling import greedy_sampling

logger = logging.getLogger(__name__)


class DialCapabilityImplementation(IntersectBaseCapabilityImplementation):
    """
    Implementation of the Dial capability.
    """

    intersect_sdk_capability_name = 'dial'

    _BACKENDS = {
        'gpax': 'src.dial_service.backends.gpax_backend',
        'sklearn': 'src.dial_service.backends.sklearn_backend',
    }

    def _get_backend_module(self, backend: str):
        if backend not in self._BACKENDS:
            raise ValueError(f'Unknown backend {backend}')
        module_name = self._BACKENDS[backend]
        module = __import__(module_name, fromlist=[''])
        return module

    def _train_model(self, data: ServersideInputBase):
        backend = data.backend.lower()
        module = self._get_backend_module(backend)
        return module.train_model(data)

    def _kernel(self, data: ServersideInputBase):
        backend = data.backend.lower()
        module = self._get_backend_module(backend)
        return module.get_kernel(data)

    # def _sampler(self, data: ServersideInputBase):
    #     # backend = data.backend.lower()
    #     # module = self._get_backend_module('greedy_search')
    #     return greedy_sampling

    def _random_in_bounds(self, data: ServersideInputBase):
        return [random.uniform(low, high) for low, high in data.bounds]  # noqa: S311
        # (TODO - probably OK to use cryptographically insecure random numbers, but check this first)

    def _hypercube(self, data: ServersideInputBase, num_points) -> list[list[float]]:
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

    @intersect_message()
    def get_next_point(self, client_data: DialInputSingle) -> list[float]:
        """
        Get the next point for optimization based on the provided strategy.

        Args:
            client_data (DialInputSingle): Input data containing bounds, strategy, and other parameters.

        Returns:
            list[float]: The selected point for the next iteration.
        """
        data = ServersideInputSingle(client_data)
        if data.strategy == 'random':
            return [random.uniform(low, high) for low, high in data.bounds]

        model = self._train_model(data)
        backend = data.backend.lower()
        module = self._get_backend_module(backend)
        selected_point = greedy_sampling(module, model, data)

        logger.debug('selected point with non-discrete measurements')
        logger.debug(selected_point)
        return selected_point

    def _create_measurement_grid(self, data):
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

    @intersect_message
    def get_next_points(self, client_data: DialInputMultiple) -> list[list[float]]:
        """
        Get multiple next points for optimization based on the provided strategy.

        Args:
            client_data (DialInputMultiple): Input data containing bounds, strategy, and other parameters.

        Returns:
            list[list[float]]: A list of selected points for the next iteration.
        """
        data = ServersideInputMultiple(client_data)

        if data.seed != -1:
            random.seed(data.seed)
            np.random.seed(data.seed)
        # model = self._train_model(data) #this will be needed when we add qEI/constant liars
        output_points = None
        match data.strategy:
            case 'random':
                output_points = [self._random_in_bounds(data) for _ in range(data.points)]
            case 'hypercube':
                output_points = self._hypercube(data, data.points)
        return output_points

    @intersect_message
    def get_surrogate_values(self, client_data: DialInputPredictions) -> list[list[float]]:
        """
        Get surrogate model predictions for given input points.

        Args:
            client_data (DialInputPredictions): Input data containing prediction points and model parameters.

        Returns:
            list[list[float]]: A list containing means, transformed standard deviations, and raw standard deviations.
        """
        data = ServersideInputPrediction(client_data)
        model = self._train_model(data)
        backend = data.backend.lower()
        module = self._get_backend_module(backend)
        means, stddevs = module.predict(model, data.x_predict, data)
        means = data.inverse_transform(means)
        transformed_stddevs = data.inverse_transform(stddevs)
        return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]

    @intersect_status()
    def status(self) -> str:
        """
        Get the status of the service.

        Returns:
            str: The status of the service ('Up').
        """
        return 'Up'
