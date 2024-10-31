import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('agg')

import numpy as np

# from scipy.stats import qmc
from boalaas_dataclass import BOALaaSInputPredictions, BOALaaSInputSingle
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rosenbrock(
    x0, x1
):  # Represents simulation error (vs experimental data) as a function of 2 simulation parameters
    return (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2


class ActiveLearningOrchestrator:
    def __init__(self):
        self.bounds = [[-2, 2], [-2, 2]]
        self.num_dims = len(self.bounds)
        # For development, we use constant, explicitly written self.dataset_x
        self.dataset_x = [
            [0.9317758694133622, -0.23597335497782845],
            [-0.7569874398003542, -0.76891211613756],
            [-0.38457336507729645, -1.1327391183311766],
            [-0.9293590899359039, 0.25039725076881014],
            [1.984696498789749, -1.7147926093003538],
            [1.2001856430453541, 1.572387611848939],
            [0.5080666898409634, -1.566722183270571],
            [-1.871124738716507, 1.9022651997285078],
            [-1.572941300813352, 1.0014173171150125],
            [0.033053333077524005, 0.44682040004191537],
        ]
        # In practice, we use the following Latin Hypercube Sampling to generate self.dataset_x:
        # self.rng = np.random.default_rng(seed=42)
        # self.lhs_sampler = qmc.LatinHypercube(d=self.num_dims, seed=self.rng)
        # self.unscaled_lhs = self.lhs_sampler.random(n=10)
        # self.l_bounds = [bound[0] for bound in self.bounds]
        # self.u_bounds = [bound[1] for bound in self.bounds]
        # self.dataset_x = qmc.scale(self.unscaled_lhs, self.l_bounds, self.u_bounds).tolist()
        # Evaluate the rosenbrock function at the initial LHS points:
        self.dataset_y = [rosenbrock(*x) for x in self.dataset_x]
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
            payload = BOALaaSInputSingle(
                strategy='expected_improvement',
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel='rbf',
                length_per_dimension=True,  # allow the matern to use separate length scales for the two parameters
                y_is_good=False,  # we wish to minimize y (the error)
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
            )
        else:
            payload = BOALaaSInputPredictions(
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
                    destination='neeter-active-learning-organization.neeter-active-learning-facility.neeter-active-learning-system.neeter-active-learning-subsystem.neeter-active-learning-service',
                    operation=f'BOALaaS.{operation}',
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
            operation == 'BOALaaS.get_surrogate_values'
        ):  # if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
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
        if self.num_dims == 2:
            plt.clf()
            data = np.maximum(
                np.array(self.mean_grid), 0.11
            )  # the predicted means can be <0 which causes white patches in the graph; this fixes that
            plt.contourf(
                self.meshgrids[0],
                self.meshgrids[1],
                data,
                levels=np.logspace(-2, 4, 101),
                norm='log',
                extend='both',
            )
            cbar = plt.colorbar()
            cbar.set_ticks(np.logspace(-2, 4, 7))
            cbar.set_label('Simulation Result')
            plt.xlabel('Simulation Parameter #1')
            plt.ylabel('Simulation Parameter #2')
            # add black dots for data points and a red marker for the recommendation:
            X_train = np.array(self.dataset_x)
            plt.scatter(X_train[:, 0], X_train[:, 1], color='black', marker='o')
            plt.scatter([self.x_EI[0]], [self.x_EI[1]], color='red', marker='o')
            plt.scatter(
                [self.x_EI[0]], [self.x_EI[1]], color='none', edgecolors='red', marker='o', s=300
            )
            plt.savefig('graph.png')
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            message = (
                'Number of dimensions is not equal to two -\nBayesian Optimization plot is not available.\n'
                'Add plotting to the graph(self) function in\nautomated_client.py to generate a custom plot.'
            )
            ax.text(0.5, 0.5, message, fontsize=18, ha='center', va='center', wrap=True)
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig('graph.png')

    # calculates the rosenbrock at a certain spot and adds it to our dataset
    def add_data(self):
        coordinates = self.x_EI
        coord_str = ', '.join([f'{coord:.2f}' for coord in coordinates])
        print(f'Running simulation at ({coord_str}): ', end='', flush=True)
        y = rosenbrock(*coordinates)
        print(f'{y:.3f}')
        self.dataset_x.append(coordinates)
        self.dataset_y.append(y)


if __name__ == '__main__':
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('NEETER_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)

    active_learning = ActiveLearningOrchestrator()
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message('get_surrogate_values'),
        **from_config_file,
    )

    # use the orchestator to create the client
    client = IntersectClient(
        config=config,
        user_callback=active_learning,  # the callback (here we use a callable object, as discussed above)
    )

    # This will run the send message -> wait for response -> callback -> repeat cycle until we have 25 points (and then raise an Exception)
    default_intersect_lifecycle_loop(
        client,
    )
