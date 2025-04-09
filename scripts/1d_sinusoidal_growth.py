import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)
from scipy.stats import qmc

from dial_dataclass import DialInputPredictions, DialInputSingle

mpl.use('agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def latin_hypercube_sample(n, search_space, seed=10):
    """
    Generates a Latin Hypercube Sample for a d-dimensional space with
    bounds for each dimension.

    Args:
        d (int): The number of dimensions.
        n (int): The number of samples to generate.
        search_space (np.ndarray): A 2D array of shape (d, 2) where each row represents
                                   the (min, max) bounds for each dimension.

    Returns:
        np.ndarray: A (n x d) array of Latin Hypercube samples within the given bounds.
    """
    # Step 1: Create an array where each row corresponds to one dimension

    d = search_space.shape[0]

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    sample = sampler.random(n=n)
    scaled_sample = np.empty_like(sample)

    for i in range(d):
        min_val, max_val = search_space[i]

        if n == 1:
            scaled_sample[:, i] = min_val + (max_val - min_val) * sample[:, i]
        else:
            # Scale the values in the i-th dimension based on the search space
            scaled_sample[:, i] = min_val + (sample[:, i] - np.min(sample[:, i])) / (
                np.max(sample[:, i]) - np.min(sample[:, i])
            ) * (max_val - min_val)

    return scaled_sample


def sinusoidal_growth(x: np.ndarray) -> np.ndarray:
    """
    Compute the sinusoidal growth function.

    Parameters:
    x (np.ndarray): Input array.

    Returns:
    np.ndarray: Output array after applying the sinusoidal growth function.
    """
    result = x + np.sin(6 * x)
    logger.debug(result)
    return result


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination

        self.bounds = np.array([[-1, 1]])  # (min max) pairs in each dimension
        self.num_dims = len(self.bounds)

        x_train = latin_hypercube_sample(5, self.bounds)
        self.dataset_y = sinusoidal_growth(x_train)
        self.dataset_x = x_train.reshape(-1, 1).tolist()
        self.dataset_y = self.dataset_y.reshape(-1).tolist()

        print(self.dataset_y)

        # Generate a (meshgrid_size x meshgrid_size) grid for predictions and graphing:
        self.meshgrid_size = 101
        self.grid_points = [
            np.linspace(dim_bounds[0], dim_bounds[1], self.meshgrid_size)
            for dim_bounds in self.bounds
        ]
        self.meshgrids = np.meshgrid(*self.grid_points, indexing='ij')
        self.points_to_predict = np.hstack([mg.reshape(-1, 1) for mg in self.meshgrids])

    # create a message to send to the server
    def assemble_message(self, operation: str) -> IntersectClientCallback:
        payload = None
        if operation == 'get_next_point':
            payload = DialInputSingle(
                strategy='upper_confidence_bound',
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel='rbf',
                # allow the matern to use separate length scales for the two parameters
                length_per_dimension=True,
                y_is_good=False,  # we wish to minimize y (the error)
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
            )
        else:
            payload = DialInputPredictions(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                points_to_predict=self.points_to_predict,
                kernel='rbf',
                length_per_dimension=True,
                y_is_good=False,
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
            )
        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=f'dial.{operation}',
                    payload=payload,
                )
            ]
        )

    # The callback function.  This is called whenever the server responds to our message.
    # This could instead be implemented by defining a callback method (and passing it later), but here we chose to directly make the object callable.
    def __call__(
        self, _source: str, operation: str, _has_error: bool, payload: INTERSECT_JSON_VALUE
    ) -> IntersectClientCallback:
        if (
            operation == 'dial.get_surrogate_values'
        ):  # if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
            self.variance_grid = np.array(payload[1]).reshape((self.meshgrid_size,) * self.num_dims)
            self.mean_grid = np.array(payload[0]).reshape((self.meshgrid_size,) * self.num_dims)
            return self.assemble_message(
                'get_next_point'
            )  # returning a message automatically sends it to the server
        # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
        self.x_EI = payload
        self.graph()
        if len(self.dataset_x) == 25:
            minpos = np.argmin(self.dataset_y)

            y_opt = self.dataset_y[minpos]
            optimal_coords = self.dataset_x[minpos]
            coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
            print(
                f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}', end='\n', flush=True
            )
            raise Exception  # noqa: TRY002 (INTERSECT interaction mechanism, do not need custom exception)
        self.add_data()
        return self.assemble_message('get_surrogate_values')

    def graph(self):
        plt.clf()

        plt.plot(self.points_to_predict, self.mean_grid)

        plt.xlabel('Temperature #1')

        # add black dots for data points and a red marker for the recommendation:
        X_train = np.array(self.dataset_x)
        Y_train = np.array(self.dataset_y)

        plt.scatter(X_train[:, 0], Y_train, color='black', marker='o')
        plt.fill_between(
            self.points_to_predict[:, 0],
            self.mean_grid + 2 * self.variance_grid,
            self.mean_grid - 2 * self.variance_grid,
            alpha=0.5,
        )
        plt.savefig('graph.png')

    # calculates the rosenbrock at a certain spot and adds it to our dataset
    def add_data(self):
        coordinates = self.x_EI
        coord_str = ', '.join([f'{coord:.2f}' for coord in coordinates])
        print(f'Running simulation at ({coord_str}): ', end='', flush=True)
        y = sinusoidal_growth(*coordinates)
        print(f'{y:.3f}')
        self.dataset_x.append(coordinates)
        self.dataset_y.append(y)


if __name__ == '__main__':
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('DIAL_CONFIG_FILE', Path(__file__).parents[1] / 'remote-conf.json'),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)

    active_learning = ActiveLearningOrchestrator(
        service_destination=HierarchyConfig(
            **from_config_file['intersect-hierarchy']
        ).hierarchy_string('.')
    )
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message('get_surrogate_values'),
        **from_config_file['intersect'],
    )
    # use the orchestator to create the client
    client = IntersectClient(
        config=config,
        # the callback (here we use a callable object, as discussed above)
        user_callback=active_learning,
    )
    # This will run the send message -> wait for response -> callback -> repeat cycle until we have 25 points (and then raise an Exception)
    default_intersect_lifecycle_loop(
        client,
    )
