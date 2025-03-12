import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('agg')

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

# from scipy.stats import qmc
from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParams,
    DialWorkflowDatasetUpdate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rosenbrock(
    x0, x1
):  # Represents simulation error (vs experimental data) as a function of 2 simulation parameters
    """TODO - calls to this function will eventually be replaced by calls to an INTERSECT service."""
    return (1 - x0) ** 2 + 100 * (x1 - x0**2) ** 2


"""
def generate_dataset_x(num_dims):
    # In practice, we use the following Latin Hypercube Sampling to generate self.dataset_x:
    # rng = np.random.default_rng(seed=42)
    # lhs_sampler = qmc.LatinHypercube(d=self.num_dims, seed=self.rng)
    # unscaled_lhs = lhs_sampler.random(n=10)
    # l_bounds = [bound[0] for bound in INITIAL_BOUNDS]
    # u_bounds = [bound[1] for bound in INITIAL_BOUNDS]
    # return qmc.scale(unscaled_lhs, l_bounds, u_bounds).tolist()
"""

# default inputs
INITIAL_BOUNDS = [[-2.0, 2.0], [-2.0, 2.0]]
NUM_DIMS = len(INITIAL_BOUNDS)
INITIAL_DATASET_X = [
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
"""Example of an explicitly-written dataset_x"""
# TODO initialize this by a call to a Rosenbrock microservice instead (call a "bulk processing" function)
INITIAL_DATASET_Y = [rosenbrock(*x) for x in INITIAL_DATASET_X]
MESHGRID_SIZE = 101
INITIAL_MESHGRIDS = np.meshgrid(
    *[np.linspace(dim_bounds[0], dim_bounds[1], MESHGRID_SIZE) for dim_bounds in INITIAL_BOUNDS],
    indexing='ij',
)
INITIAL_POINTS_TO_PREDICT = np.hstack([mg.reshape(-1, 1) for mg in INITIAL_MESHGRIDS])


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination

        # This value gets populated from the return value of initializing the workflow
        self.workflow_id = ''

        # The full dataset object state only needs to exist for the purposes of generating the graph and determining a stop-workflow order
        # if we don't care about "step by step" data, we technically do NOT need to save these as stateful, as we can get the data at the end by calling "dial.get_workflow_data"
        self.dataset_x = INITIAL_DATASET_X
        self.dataset_y = INITIAL_DATASET_Y

    # create a message to send to the server
    def assemble_message(self, operation: str, **kwargs: Any) -> IntersectClientCallback:
        if operation == 'initialize_workflow':
            payload = DialWorkflowCreationParams(
                dataset_x=INITIAL_DATASET_X,
                dataset_y=INITIAL_DATASET_Y,
                bounds=INITIAL_BOUNDS,
                kernel='rbf',
                length_per_dimension=True,  # allow the matern to use separate length scales for the two parameters
                y_is_good=False,  # we wish to minimize y (the error)
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
            )
        elif operation == 'update_workflow_with_data':
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == 'get_next_point':
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy='expected_improvement',
            )
        elif operation == 'get_surrogate_values':
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=INITIAL_POINTS_TO_PREDICT,
            )
        else:
            err_msg = f'Invalid operation {operation}'
            raise Exception(err_msg)  # noqa: TRY002
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
        self,
        _source: str,
        operation: str,
        _has_error: bool,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback:
        if _has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            raise Exception  # noqa: TRY002 (break INTERSECT loop)
        if operation == 'dial.initialize_workflow':
            self.workflow_id: str = payload
            return self.assemble_message('get_surrogate_values')
        if operation == 'dial.update_workflow_with_data':
            return self.assemble_message('get_surrogate_values')
        if (
            operation == 'dial.get_surrogate_values'
        ):  # if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
            self.mean_grid = np.array(payload[0]).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            return self.assemble_message(
                'get_next_point'
            )  # returning a message automatically sends it to the server

        if operation == 'dial.get_next_point':
            # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
            self.graph(payload)
            if len(self.dataset_x) == 25:
                minpos = np.argmin(self.dataset_y)
                y_opt = self.dataset_y[minpos]
                optimal_coords = self.dataset_x[minpos]
                coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
                print(
                    f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}',
                    end='\n',
                    flush=True,
                )
                raise Exception  # noqa: TRY002 (INTERSECT interaction mechanism, do not need custom exception)
            return self.add_data(payload)

        err_msg = f'Unknown operation received: {operation}'
        raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)

    def graph(self, x_EI: list[float]):
        if NUM_DIMS == 2:
            plt.clf()
            data = np.maximum(
                np.array(self.mean_grid), 0.11
            )  # the predicted means can be <0 which causes white patches in the graph; this fixes that
            plt.contourf(
                INITIAL_MESHGRIDS[0],
                INITIAL_MESHGRIDS[1],
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
            plt.scatter([x_EI[0]], [x_EI[1]], color='red', marker='o')
            plt.scatter(
                [x_EI[0]],
                [x_EI[1]],
                color='none',
                edgecolors='red',
                marker='o',
                s=300,
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
    def add_data(self, coordinates: list[float]):
        coord_str = ', '.join([f'{coord:.2f}' for coord in coordinates])
        print(f'Running simulation at ({coord_str}): ', end='', flush=True)
        y = rosenbrock(*coordinates)
        print(f'{y:.3f}')
        self.dataset_x.append(coordinates)
        self.dataset_y.append(y)
        return self.assemble_message('update_workflow_with_data', next_x=coordinates, next_y=y)


if __name__ == '__main__':
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('DIAL_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'),
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
        initial_message_event_config=active_learning.assemble_message('initialize_workflow'),
        **from_config_file['intersect'],
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
