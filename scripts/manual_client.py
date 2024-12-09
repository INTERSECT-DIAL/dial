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
from dial_dataclass import DialInputPredictions, DialInputSingle
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This is a replication of the workflow in the Maryland nature paper.  In this workflow, we have two tuneable "knobs" on a chemical process: Duration (20-300 ms) and Temperature (1200-2000 K).  We seek to maximize the yield (amount of desired chemical produced).
We start with 25 data points, then call the service, which calculates the point with the maximial EI (Expected Improvement).  This point is recommended to the user, who would then run a experiment, giving the 26th data point.
This process repeats until killed.

Overall, this represents a manual workflow: The user has the client open in a terminal, and runs the experiment/simulation/whatever is producing results, which is not connected to INTERSECT.

Note that the service must be started first, then the client.
"""


def read_float(
    prompt: str,
):  # read a floating point value from the user, keep asking until they give a valid one
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print('Invalid floating point input', file=sys.stderr)


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination

        self.dataset_x = [
            [20.0, 1200.0],
            [20.0, 1400.0],
            [20.0, 1600.0],
            [20.0, 1800.0],
            [20.0, 2000.0],
            [55.0, 1200.0],
            [55.0, 1400.0],
            [55.0, 1600.0],
            [55.0, 1800.0],
            [55.0, 2000.0],
            [70.0, 1600.0],
            [70.0, 1800.0],
            [70.0, 2000.0],
            [110.0, 1400.0],
            [110.0, 1600.0],
            [110.0, 1800.0],
            [110.0, 2000.0],
            [220.0, 1600.0],
            [220.0, 1800.0],
            [220.0, 2000.0],
            [330.0, 1200.0],
            [330.0, 1800.0],
            [330.0, 2000.0],
            [165.0, 1800.0],
            [165.0, 2000.0],
        ]
        self.dataset_y = [
            0.4023,
            0.8534,
            2.9733,
            4.7806,
            5.5944,
            0.191,
            1.1508,
            3.3015,
            6.51,
            9.8264,
            5.28,
            6.35,
            9.68,
            2.5194,
            7.1148,
            9.8176,
            9.7903,
            5.55,
            7.04,
            11.52,
            1.3018,
            7.15,
            10.42,
            6.77,
            11.51,
        ]
        self.bounds = [[20, 330], [1200, 2000]]
        # generate a 101x101 grid for predictions and graphing:
        self.xx, self.yy = np.meshgrid(
            np.linspace(self.bounds[0][0], self.bounds[0][1], 101),
            np.linspace(self.bounds[1][0], self.bounds[1][1], 101),
            indexing='ij',
        )
        self.points_to_predict = np.hstack([self.xx.reshape(-1, 1), self.yy.reshape(-1, 1)])

    # create a message to send to the server
    def assemble_message(self, operation: str) -> IntersectClientCallback:
        payload = None
        if operation == 'get_next_point':
            payload = DialInputSingle(
                strategy='expected_improvement',
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel='matern',
                length_per_dimension=True,  # allow the matern to use separate length scales for temp and duration
                y_is_good=True,  # we wish to maximize y (the yield)
                backend='sklearn',
                seed=-1,
            )
        else:
            payload = DialInputPredictions(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                points_to_predict=self.points_to_predict,
                kernel='matern',
                length_per_dimension=True,  # allow the matern to use separate length scales for temp and duration
                y_is_good=True,  # we wish to maximize y (the yield)
                backend='sklearn',
                seed=-1,
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
            self.mean_grid = np.array(payload[0]).reshape(self.xx.shape)
            return self.assemble_message(
                'get_next_point'
            )  # returning a message automatically sends it to the server
        # if we receive an EI recommendation, record it, show the user the current graph, and ask the user for the results of their experiment:
        self.x_EI = payload
        self.graph()
        self.add_data()
        return self.assemble_message('get_surrogate_values')

    # makes a color graph of the predicted yields, with markers for the training data and EI-recommended point:
    def graph(self):
        plt.clf()
        plt.contourf(
            self.xx, self.yy, self.mean_grid, levels=np.linspace(0, 12, 101), extend='both'
        )
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(0, 12, 7))
        cbar.set_label('C2H4 Yield (%)')
        plt.xlabel('Pulse Duration (ms)')
        plt.ylabel('Peak Temperature (K)')
        # add black dots for data points and a red marker for the recommendation:
        X_train = np.array(self.dataset_x)
        plt.scatter(X_train[:, 0], X_train[:, 1], color='black', marker='o')
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color='red', marker='o')
        plt.scatter(
            [self.x_EI[0]], [self.x_EI[1]], color='none', edgecolors='red', marker='o', s=300
        )
        plt.savefig('graph.png')

    # asks the user for a data point (an experimental result) and adds it to our dataset
    def add_data(self):
        print(f'\nEI recommends running at: {self.x_EI[0]:.1f} ms, {self.x_EI[1]:.1f} K')
        print('\nEnter Experimental Data:', file=sys.stderr)
        x0 = read_float('Duration (ms): ')
        x1 = read_float('Temperature (K): ')
        y = read_float('Yield (%): ')
        self.dataset_x.append([x0, x1])
        self.dataset_y.append(y)


if __name__ == '__main__':
    # Create configuration class, which handles user input validation - see the IntersectClientConfig class documentation for more info
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get(
            'DIAL_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'
        ),
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)

    # Create our orchestrator
    active_learning = ActiveLearningOrchestrator(
        service_destination=HierarchyConfig(
            **from_config_file['intersect-hierarchy']
        ).hierarchy_string('.')
    )
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message(
            'get_surrogate_values'
        ),  # the initial message to send
        **from_config_file['intersect'],
    )

    # use the orchestator to create the client
    client = IntersectClient(
        config=config,
        user_callback=active_learning,  # the callback (here we use a callable object, as discussed above)
    )

    # This will run the send message -> wait for response -> callback -> repeat cycle until the user raises an Exception with Ctrl+C
    print('Beginning user input cycle, use CTRL+D to exit', file=sys.stderr)
    default_intersect_lifecycle_loop(
        client,
    )
