import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from intersect_sdk import (
    INTERSECT_RESPONSE_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

# from scipy.stats import qmc
from dial_dataclass import (
    DialInputMultipleOtherStrategy,
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
)

mpl.use('agg')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rosenbrock(x, y) -> np.ndarray:
    x = np.asarray(x)
    y = np.asarray(y)
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


# default inputs
BOUNDS = [[-2.0, 2.0], [-2.0, 2.0]]
NUM_DIMS = len(BOUNDS)

MESHGRID_SIZE = 201
SURROGATE_MESHGRID = np.meshgrid(
    *[np.linspace(dim_bounds[0], dim_bounds[1], MESHGRID_SIZE) for dim_bounds in BOUNDS],
    indexing='ij',
)
POINTS_TO_PREDICT = np.hstack([mg.reshape(-1, 1) for mg in SURROGATE_MESHGRID]).tolist()

GROUND_TRUTH_X = np.hstack([mg.reshape(-1, 1) for mg in SURROGATE_MESHGRID])
GROUND_TRUTH_Y = rosenbrock(SURROGATE_MESHGRID[0], SURROGATE_MESHGRID[1]).tolist()


NUM_ITERATIONS = 100


SAMPLING_GRID_SIZE = [5, 5]
STRATEGY_SCHEDULE = [
    {'strategy': 'center', 'num_samples': 1},
    {'strategy': 'corners', 'num_samples': 2**NUM_DIMS},
    # {'strategy': 'grid',      'num_samples': np.prod(SAMPLING_GRID_SIZE), 'args': {'grid_size': SAMPLING_GRID_SIZE}},
    # {'strategy': 'chebyshev', 'num_samples': np.prod(SAMPLING_GRID_SIZE), 'args': {'grid_size': SAMPLING_GRID_SIZE}},
    {
        'strategy': 'latin_hypercube',
        'num_samples': np.prod(SAMPLING_GRID_SIZE),
        'args': {'grid_size': SAMPLING_GRID_SIZE},
    },
    {'strategy': 'expected_improvement'},
]

# KERNEL HYPERPARAMETERS
LENGTH_SCALE = 0.2
NOISE_LEVEL = 10e-6
CONSTANT_VALUE = 1.0


class ActiveLearningOrchestrator:
    @property
    def current_strategy(self):
        if not hasattr(self, '_current_strategy_id'):
            self._current_strategy_id = 0
            self._strategy_break_points = np.cumsum(
                [s.get('num_samples', 1e6) for s in STRATEGY_SCHEDULE]
            )

        if len(self.dataset_y) < self._strategy_break_points[self._current_strategy_id]:
            return STRATEGY_SCHEDULE[self._current_strategy_id]
        self._current_strategy_id += 1
        if self._current_strategy_id < len(self._strategy_break_points):
            return STRATEGY_SCHEDULE[self._current_strategy_id]
        return {'strategy': 'expected_improvement'}

    def __init__(self, service_destination: str, rosenbrock_destination: str):
        self.service_destination = service_destination
        self.rosenbrock_destination = rosenbrock_destination

        # This value gets populated from the return value of initializing the workflow
        self.workflow_id = ''

        # The full dataset object state only needs to exist for the purposes of generating the graph and determining a stop-workflow order
        # if we don't care about "step by step" data, we technically do NOT need to save these as stateful, as we can get the data at the end by calling "dial.get_workflow_data"
        self.dataset_x = []
        self.dataset_y: list[float] = []

    # create a message to send to the server
    def assemble_message(self, operation: str, **kwargs: Any) -> IntersectClientCallback:
        print(f'assembling dial message: {operation}')
        if operation == 'initialize_workflow':
            payload = DialWorkflowCreationParamsClient(
                # init_strategy=self.init_strategy,
                # init_npoints=self.init_npoints,
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=BOUNDS,
                dim_x=NUM_DIMS,  # Explicitly set the dimension based on the bounds
                kernel='rbf',
                kernel_args={
                    'length_scale': LENGTH_SCALE,
                    'length_scale_bounds': 'fixed',
                    'noise_level': NOISE_LEVEL,
                    'noise_level_bounds': 'fixed',
                    'constant_value': CONSTANT_VALUE,
                    'constant_value_bounds': 'fixed',
                },
                length_per_dimension=False,  # allow the matern to use separate length scales for the two parameters
                y_is_good=False,  # we wish to minimize y (the error)
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
            )
            print(payload)
        elif operation == 'update_workflow_with_data':
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == 'get_next_point':
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy=self.current_strategy['strategy'],
                strategy_args=self.current_strategy.get('args', None),
                bounds=BOUNDS,
            )
        elif operation == 'get_next_points':
            payload = DialInputMultipleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy=self.current_strategy['strategy'],
                strategy_args=self.current_strategy.get('args', None),
                points=self.batch_size,
                bounds=BOUNDS,
            )
        elif operation == 'get_surrogate_values':
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=POINTS_TO_PREDICT,
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

    def assemble_rosenbrock_message(self, operation: str) -> IntersectClientCallback:
        print(f'assembling rosenbrock message: {operation}')
        if operation == 'rosenbrock':
            last_x = self.dataset_x[-1]
            payload = {
                'x': last_x[0],
                'y': last_x[1],
            }
        elif operation == 'rosenbrock_bulk':
            payload = [{'x': x[0], 'y': x[1]} for x in self.dataset_x]
        else:
            err_msg = f'Invalid operation {operation}'
            raise Exception(err_msg)  # noqa: TRY002
        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.rosenbrock_destination,
                    operation=f'Rosenbrock.{operation}',
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
        has_error: bool,
        payload: INTERSECT_RESPONSE_VALUE,
    ) -> IntersectClientCallback:
        if has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            msg = f'Error in operation {operation}: payload = {payload}'
            raise Exception(msg)  # noqa: TRY002 (break INTERSECT loop)
        if operation == 'Rosenbrock.rosenbrock':
            # this operation gets called periodically
            self.dataset_y.append(payload)
            coord_str = ', '.join([f'{coord:.2f}' for coord in self.next_point])
            print(f'Running simulation at ({coord_str}): {payload:.3f}')
            if len(self.dataset_x) >= NUM_ITERATIONS:
                minpos = np.argmin(self.dataset_y)
                y_opt = self.dataset_y[minpos]
                optimal_coords = self.dataset_x[minpos]
                self.graph(optimal_coords, True)
                coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
                print(
                    f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}',
                    end='\n',
                    flush=True,
                )
                msg = 'Client simulation completed successfully.'
                raise Exception(msg)  # noqa: TRY002 (INTERSECT interaction mechanism, do not need custom exception)
            return self.assemble_message(
                'update_workflow_with_data', next_x=self.dataset_x[-1], next_y=payload
            )
        if operation == 'Rosenbrock.rosenbrock_bulk':
            # this operation only gets called at the very beginning of the workflow
            self.dataset_y: list[float] = payload
            return self.assemble_message('initialize_workflow')
        if operation == 'dial.initialize_workflow':
            self.workflow_id: str = payload
            # return self.assemble_message('get_surrogate_values')
            return self.assemble_message('get_next_point')
        if operation == 'dial.update_workflow_with_data':
            return self.assemble_message('get_surrogate_values')
        if operation == 'dial.get_surrogate_values':
            data = payload['data'][0]
            self.surrogate_y = np.array(data).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            return self.assemble_message('get_next_point')
        if operation == 'dial.get_next_point':
            # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
            next_point = payload['data']
            if hasattr(self, 'surrogate_y'):
                self.graph(next_point)
            self.next_point = next_point
            self.dataset_x.append(next_point)
            return self.assemble_rosenbrock_message('rosenbrock')
        if operation == 'dial.get_next_points':
            # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
            next_points = payload['data']
            for next_point in next_points:
                self.graph(next_point)
                self.dataset_x.append(next_point)
                coord_str = ', '.join([f'{coord:.2f}' for coord in next_point])
                print(f'Running simulation at ({coord_str}): ', end='', flush=True)
            return self.assemble_rosenbrock_message('rosenbrock')

        err_msg = f'Unknown operation received: {operation}'
        raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)

    def graph(self, x_EI: list[float], final: bool = False):
        if NUM_DIMS == 2:
            plt.figure(figsize=(12, 6))

            # ground truth Rosenbrock function
            plt.subplot(1, 2, 1)
            plt.contourf(
                SURROGATE_MESHGRID[0],
                SURROGATE_MESHGRID[1],
                GROUND_TRUTH_Y,
                levels=np.logspace(-2, 4, 101),
                norm='log',
                extend='both',
            )
            plt.contour(
                SURROGATE_MESHGRID[0],
                SURROGATE_MESHGRID[1],
                GROUND_TRUTH_Y,
                levels=np.logspace(-2, 4, 10),
                norm='log',
                extend='both',
                colors='k',
            )
            plt.scatter(1, 1, color='red', marker='*', s=200, zorder=10)
            plt.title('Ground Truth Rosenbrock Function')

            # surrogate model prediction
            plt.subplot(1, 2, 2)
            data = np.maximum(
                np.array(self.surrogate_y), 0.11
            )  # the predicted means can be <0 which causes white patches in the graph; this fixes that
            plt.contourf(
                SURROGATE_MESHGRID[0],
                SURROGATE_MESHGRID[1],
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
            plt.title('Surrogate')
            # add black dots for data points and a red marker for the recommendation:
            if len(self.dataset_x) > 0:
                X_train = np.array(self.dataset_x)
                plt.scatter(X_train[:, 0], X_train[:, 1], color='black', marker='o')
                plt.scatter(1.0, 1.0, s=300, color='None', edgecolors='black', marker='o')

                minpos = np.argmin(self.dataset_y)
                optimal_coords = self.dataset_x[minpos]

                plt.scatter(optimal_coords[0], optimal_coords[1], color='black', marker='*', s=200)
            if final:
                final_x = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
                plt.suptitle(
                    f'Best point estimate so far is x=({final_x}), y={self.dataset_y[minpos]:.3f}, true minimum is at x=(1.00, 1.00), y=0.00'
                )
            else:
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
            plt.close()
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
            fig.savefig('graph.png')


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
        ).hierarchy_string('.'),
        rosenbrock_destination=HierarchyConfig(
            **from_config_file['rosenbrock-hierarchy']
        ).hierarchy_string('.'),
    )
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message('initialize_workflow'),
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
