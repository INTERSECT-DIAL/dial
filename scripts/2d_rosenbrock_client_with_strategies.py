import argparse
import bisect
import json
import logging
import math
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
    DialWorkflowDatasetUpdates,
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


NUM_ITERATIONS = 200


SAMPLING_GRID_SIZE = [5, 5]
STRATEGY_SCHEDULE = [
    {'strategy': 'center', 'num_samples': 1},
    {'strategy': 'corners', 'num_samples': 2**NUM_DIMS},
    # # {'strategy': 'grid',      'num_samples': np.prod(SAMPLING_GRID_SIZE), 'args': {'grid_size': SAMPLING_GRID_SIZE}},
    # {'strategy': 'chebyshev', 'num_samples': np.prod(SAMPLING_GRID_SIZE), 'args': {'grid_size': SAMPLING_GRID_SIZE}},
    # {
    #     'strategy': 'latin_hypercube',
    #     'num_samples': np.prod(SAMPLING_GRID_SIZE),
    #     'args': {'grid_size': SAMPLING_GRID_SIZE},
    # },
    # {
    #     'batch_strategy': 'liar',
    #     'strategy': 'corners',
    #     'num_samples': 4,
    #     'args': {'liar_value': 0},
    # },
    {
        'batch_strategy': 'liar',
        'strategy': 'grid',
        'num_samples': 25,
        'args': {'liar_value': 'median', 'grid_size': [4, 4]},
    },
    {
        'batch_strategy': 'liar',
        'strategy': 'chebyshev',
        'num_samples': 25,
        'args': {'liar_value': 100, 'grid_size': [5, 5]},
    },
    {
        'batch_strategy': 'believer',
        'strategy': 'latin_hypercube',
        'num_samples': 25,
        'args': {'grid_size': [5, 5]},
    },
    {
        'batch_strategy': 'liar',
        'strategy': 'expected_improvement',
        'num_samples': 10,
        'args': {'liar_value': 0},
    },
    {
        'batch_strategy': 'liar',
        'strategy': 'expected_improvement',
        'num_samples': 10,
        'args': {'liar_value': 'mean'},
    },
    {
        'batch_strategy': 'believer',
        'strategy': 'expected_improvement',
        'num_samples': 10,
        'args': {'liar_value': 'mean'},
    },
    {'strategy': 'expected_improvement', 'num_samples': 10},
    {
        'batch_strategy': 'believer',
        'strategy': 'expected_improvement',
        'num_samples': 30,
        'args': {'liar_value': 'mean'},
    },
    {
        'batch_strategy': 'liar',
        'strategy': 'expected_improvement',
        'num_samples': 10,
        'args': {'liar_value': 'min'},
    },
    {'strategy': 'expected_improvement'},
]

# KERNEL HYPERPARAMETERS
LENGTH_SCALE = 0.2
NOISE_LEVEL = 10e-6
CONSTANT_VALUE = 1.0


class Scheduler:
    def __init__(self, strategy_schedule: list[dict[str, Any]]):
        self.strategy_schedule = strategy_schedule
        self.strategy_break_points = np.cumsum(
            [s.get('num_samples', 1e6) for s in strategy_schedule], dtype=int
        )

    def __call__(self, sample_index):
        index = bisect.bisect_right(self.strategy_break_points, sample_index)
        return self.strategy_schedule[min(index, len(self.strategy_schedule) - 1)]

    def get_strategy_index(self, sample_index):
        index = bisect.bisect_right(self.strategy_break_points, sample_index)
        return min(index, len(self.strategy_schedule) - 1)


class Plotter:
    """Class to handle plotting of the surrogate model and optimization progress."""

    def __init__(self, scheduler: Scheduler, max_cols: int = 4):
        """Initialize with a grid of subplots based on the number of strategies"""
        self.save_path = 'graph.png'
        self.scheduler = scheduler

        num_plots = len(scheduler.strategy_schedule) + 1
        num_cols = min(max_cols, num_plots)
        num_rows = math.ceil(num_plots / num_cols)

        self._graph_fig, graph_axes = plt.subplots(
            num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows), squeeze=False
        )
        self._graph_axes = graph_axes.ravel()

        # Hide empty subplot positions
        for unused_ax in self._graph_axes[num_plots:]:
            unused_ax.set_visible(False)

        self._graph_colorbars = {}

        break_points = [0, *list(scheduler.strategy_break_points)]
        for i, ax in enumerate(self._graph_axes[1:num_plots]):
            current_strategy = scheduler(break_points[i])
            batch_strategy = current_strategy.get('batch_strategy', None)
            if batch_strategy and batch_strategy.lower() == 'liar':
                batch_strategy = f'{batch_strategy} ({current_strategy["args"]["liar_value"]})'
            strategy_name = (f'{batch_strategy} + ' if batch_strategy else '') + current_strategy[
                'strategy'
            ]
            strategy_name += (
                f', {current_strategy["num_samples"]} samples'
                if 'num_samples' in current_strategy
                else ''
            )

            # ax.set_xlabel('Simulation Parameter #1')
            # ax.set_ylabel('Simulation Parameter #2')
            ax.set_title(f'{strategy_name}')

            colorbar = self._graph_fig.colorbar(None, ax=ax)
            colorbar.set_ticks(np.logspace(-2, 4, 7))
            # colorbar.set_label('Simulation Result')
            self._graph_colorbars[ax] = colorbar

        self.plot_ground_truth()

    def plot_ground_truth(self):
        """Plot the ground truth Rosenbrock function on the first subplot"""
        contourf = self._graph_axes[0].contourf(
            SURROGATE_MESHGRID[0],
            SURROGATE_MESHGRID[1],
            GROUND_TRUTH_Y,
            levels=np.logspace(-2, 4, 101),
            norm='log',
            extend='both',
        )
        self._graph_axes[0].contour(
            SURROGATE_MESHGRID[0],
            SURROGATE_MESHGRID[1],
            GROUND_TRUTH_Y,
            levels=np.logspace(-2, 4, 10),
            norm='log',
            extend='both',
            colors='k',
        )
        self._graph_axes[0].scatter(
            1,
            1,
            color='red',
            marker='*',
            s=200,
            zorder=10,
        )
        self._graph_axes[0].set_xlabel('Simulation Parameter #1')
        self._graph_axes[0].set_ylabel('Simulation Parameter #2')
        self._graph_axes[0].set_title('Ground Truth Rosenbrock Function')

        colorbar = self._graph_fig.colorbar(contourf, ax=self._graph_axes[0])
        colorbar.set_ticks(np.logspace(-2, 4, 7))
        # colorbar.set_label('Simulation Result')
        self._graph_colorbars[self._graph_axes[0]] = colorbar

        self._graph_fig.tight_layout()
        self._graph_fig.savefig(self.save_path, bbox_inches='tight')

    def __call__(
        self,
        sample_index: int,
        new_x: list[float] | list[list[float]],
        train_x,
        train_y,
        surrogate_y=None,
        final: bool = False,
    ):
        strategy_index = self.scheduler.get_strategy_index(sample_index)
        ax = self._graph_axes[strategy_index + 1]

        # Save attributes before clearing
        t = ax.get_title()
        xl = ax.get_xlabel()
        yl = ax.get_ylabel()
        # Clear the axis content only
        ax.cla()
        # Restore the attributes
        ax.set_title(t)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

        if NUM_DIMS == 2:
            new_x = np.asarray(new_x, dtype=float).reshape(-1, NUM_DIMS)

            if surrogate_y is not None:
                data = np.maximum(np.asarray(surrogate_y), 0.11)
            else:
                data = np.zeros((MESHGRID_SIZE, MESHGRID_SIZE))

            contourf = ax.contourf(
                SURROGATE_MESHGRID[0],
                SURROGATE_MESHGRID[1],
                data,
                levels=np.logspace(-2, 4, 101),
                norm='log',
                extend='both',
            )
            ax.scatter(
                1.0,
                1.0,
                s=300,
                facecolors='none',
                edgecolors='black',
                marker='o',
                label='True Minimum',
                zorder=10,
            )

            self._graph_colorbars[ax].update_normal(contourf)

            optimal_coords = None
            minpos = None
            if len(train_x) > 0:
                train_x = np.asarray(train_x)

                prev_strategies_points = self.scheduler.strategy_break_points[
                    max(0, strategy_index - 1)
                ]

                ax.scatter(
                    train_x[:prev_strategies_points, 0],
                    train_x[:prev_strategies_points, 1],
                    color='black',
                    marker='s',
                    label='Previous strategies',
                    s=30,
                    alpha=0.3,
                    zorder=10,
                )
                ax.scatter(
                    train_x[prev_strategies_points:, 0],
                    train_x[prev_strategies_points:, 1],
                    color='black',
                    marker='o',
                    label='Current strategy',
                    s=50,
                    zorder=10,
                )

                minpos = int(np.argmin(train_y))
                optimal_coords = np.asarray(train_x[minpos])

                ax.scatter(
                    optimal_coords[0],
                    optimal_coords[1],
                    color='red',
                    marker='*',
                    s=200,
                    label='Best Point Estimate',
                    zorder=10,
                )

            if final and optimal_coords is not None:
                ax.set_title('final surrogate')
                final_x = ', '.join(f'{coord:.2f}' for coord in optimal_coords)

                self._graph_fig.suptitle(
                    f'Best point estimate so far is x=({final_x}), '
                    f'y={train_y[minpos]:.3f}; '
                    'true minimum is at x=(1.00, 1.00), y=0.00',
                    x=0.5,
                    y=1.0,
                )
            else:
                ax.scatter(
                    new_x[:, 0],
                    new_x[:, 1],
                    color='red',
                    marker='o',
                    label='Recommended Points',
                    s=50,
                    zorder=10,
                )

            # Extract and deduplicate handles and labels from ALL subplots
            handles, labels = [], []
            for ax in self._graph_fig.axes:
                h, lab = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(lab)
            # Use a dictionary to keep only the first occurrence of each unique label
            by_label = dict(zip(labels, handles, strict=False))
            # Create the clean figure-level legend
            self._graph_fig.legend(
                by_label.values(),
                by_label.keys(),
                loc='outside lower center',
                ncol=3,
                bbox_to_anchor=(0.5, -0.05),
            )

            self._graph_fig.tight_layout()
            self._graph_fig.savefig(self.save_path, bbox_inches='tight')


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str, rosenbrock_destination: str):
        self.service_destination = service_destination
        self.rosenbrock_destination = rosenbrock_destination

        # This value gets populated from the return value of initializing the workflow
        self.workflow_id = ''

        # The full dataset object state only needs to exist for the purposes of generating the graph and determining a stop-workflow order
        # if we don't care about "step by step" data, we technically do NOT need to save these as stateful, as we can get the data at the end by calling "dial.get_workflow_data"
        self.dataset_x = []
        self.dataset_y: list[float] = []

        self.next_x = None
        self.surrogate_y = None

        self.scheduler = Scheduler(STRATEGY_SCHEDULE)
        self.plotter = Plotter(scheduler=self.scheduler)

    @property
    def current_strategy(self):
        return self.scheduler(len(self.dataset_y))

    # create a message to send to the server
    def assemble_message(self, operation: str, **kwargs: Any) -> IntersectClientCallback:
        print(f'assembling dial message: {operation}')
        if operation == 'initialize_workflow':
            payload = DialWorkflowCreationParamsClient(
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
        elif operation == 'update_workflow_with_data':
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == 'update_workflow_with_batch_data':
            payload = DialWorkflowDatasetUpdates(
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
            if self.current_strategy.get('batch_strategy', None) is None:
                return self.assemble_message('get_next_point')
            payload = DialInputMultipleOtherStrategy(
                workflow_id=self.workflow_id,
                batch_strategy=self.current_strategy.get('batch_strategy'),
                strategy=self.current_strategy['strategy'],
                strategy_args=self.current_strategy.get('args', None),
                points=self.current_strategy.get('num_samples', 1),
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
            payload = {
                'x': self.next_x[0],
                'y': self.next_x[1],
            }
        elif operation == 'rosenbrock_bulk':
            payload = [{'x': x[0], 'y': x[1]} for x in self.next_x]
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
            coord_str = ', '.join([f'{coord:.2f}' for coord in self.next_x])
            print(f'Running simulation at ({coord_str}): {payload:.3f}')
            self.dataset_x.append(self.next_x)
            self.dataset_y.append(payload)
            return self.assemble_message(
                'update_workflow_with_data', next_x=self.next_x, next_y=payload
            )
        if operation == 'Rosenbrock.rosenbrock_bulk':
            self.dataset_x += self.next_x
            self.dataset_y += payload
            coord_str = '\n '.join(
                [
                    '(' + ', '.join([f'{coord:.2f}' for coord in next_x]) + f') -> {next_y:.3f}'
                    for next_x, next_y in zip(self.next_x, payload, strict=False)
                ]
            )
            print(f'Running simulation at\n {coord_str}')
            return self.assemble_message(
                'update_workflow_with_batch_data', next_x_list=self.next_x, next_y_list=payload
            )
        if operation == 'dial.initialize_workflow':
            self.workflow_id: str = payload
            return self.assemble_message('get_next_points')
        if operation == 'dial.update_workflow_with_data':
            return self.assemble_message('get_surrogate_values')
        if operation == 'dial.update_workflow_with_batch_data':
            return self.assemble_message('get_surrogate_values')
        if operation == 'dial.get_surrogate_values':
            means = payload['values']
            self.surrogate_y = np.array(means).reshape((MESHGRID_SIZE,) * NUM_DIMS)
            if len(self.dataset_x) >= NUM_ITERATIONS:
                minpos = np.argmin(self.dataset_y)
                x_opt = self.dataset_x[minpos]
                y_opt = self.dataset_y[minpos]
                # self.graph(x_opt, True)
                self.plotter(
                    sample_index=len(self.dataset_x),
                    new_x=x_opt,
                    train_x=self.dataset_x,
                    train_y=self.dataset_y,
                    surrogate_y=self.surrogate_y,
                    final=True,
                )
                coord_str = ', '.join([f'{coord:.2f}' for coord in x_opt])
                print(
                    f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}',
                    end='\n',
                    flush=True,
                )
                msg = 'Client simulation completed successfully.'
                raise Exception(msg)  # noqa: TRY002 (INTERSECT interaction mechanism, do not need custom exception)
            return self.assemble_message('get_next_points')
        if operation == 'dial.get_next_point':
            # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
            self.next_x = payload['data']
            self.plotter(
                sample_index=len(self.dataset_x),
                new_x=self.next_x,
                train_x=self.dataset_x,
                train_y=self.dataset_y,
                surrogate_y=self.surrogate_y,
                final=False,
            )
            return self.assemble_rosenbrock_message('rosenbrock')
        if operation == 'dial.get_next_points':
            # if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
            self.next_x = payload['data']
            self.plotter(
                sample_index=len(self.dataset_x),
                new_x=self.next_x,
                train_x=self.dataset_x,
                train_y=self.dataset_y,
                surrogate_y=self.surrogate_y,
                final=False,
            )
            return self.assemble_rosenbrock_message('rosenbrock_bulk')

        err_msg = f'Unknown operation received: {operation}'
        raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)


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
