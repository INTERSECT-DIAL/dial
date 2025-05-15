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

from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
)

mpl.use('agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sinusoidal_growth(x: np.ndarray) -> np.ndarray:
    result = x + np.sin(6 * x)
    logger.debug(result)
    return result


class IntersectCallbackError(Exception):
    def __init__(self, operation, payload):
        message = f"Intersect callback_message error during operation \
            '{operation}'. Payload: {payload}"
        super().__init__(message)


class IntersectCallbackEnd(Exception):  # noqa: N818
    def __init__(self):
        message = 'Stopping Intersect Calls'
        super().__init__(message)


class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.bounds = np.array([[-2, 2]])
        self.num_dims = len(self.bounds)

        self.x_raw = np.array([[1], [2.0]])
        self.x_test = np.array([[-1], [0.5]])
        self.y_raw = sinusoidal_growth(self.x_raw)

        self.meshgrid_size = 100
        self.grid_points = [
            np.linspace(dim_bounds[0], dim_bounds[1], self.meshgrid_size)
            for dim_bounds in self.bounds
        ]
        self.meshgrids = np.meshgrid(*self.grid_points, indexing='ij')
        self.x_grid = np.hstack([mg.reshape(-1, 1) for mg in self.meshgrids])

        # Active learning variables
        self.dataset_x = self.x_raw.reshape(-1, 1).tolist()
        self.dataset_y = self.y_raw.reshape(-1).tolist()
        self.test_points = self.x_test.reshape(-1, 1).tolist()

        self.kernel = 'rbf'
        self.kernel_args = {'length_scale': 0.12, 'length_scale_bounds': 'fixed'}
        self.backend = 'sklearn'
        self.backend_args = None
        self.strategy = 'upper_confidence_bound'
        self.strategy_args = {'exploit': 0.4, 'explore': 1}
        self.niter = 0
        self.max_iter = 0
        self.at_grids = True
        self.variance_grid = None
        self.mean_grid = None
        self.variance_test = None
        self.mean_test = None
        self.x_next = None

        # Intersect variables
        self.workflow_id = None
        self.service_destination = service_destination

    def __call__(
        self, _source: str, operation: str, _has_error: bool, payload: INTERSECT_JSON_VALUE
    ) -> IntersectClientCallback:
        if _has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            raise IntersectCallbackError(operation, payload)

        if operation == 'dial.initialize_workflow':
            self.workflow_id = payload
            return self.callback_message('dial.get_surrogate_values')

        # ----------------- Active learning loop -----------------
        if operation == 'dial.get_surrogate_values':
            self.handle_surrogate_values(payload)

            if self.at_grids:
                print(f'Step {self.niter}')
                return self.callback_message('dial.get_surrogate_values', at_grids=False)
            return self.callback_message('dial.get_next_point')

        if operation == 'dial.get_next_point':
            self.handle_next_points(payload)
            self.graph()
            return self.callback_message('dial.update_workflow_with_data')

        if operation == 'dial.update_workflow_with_data':
            self.niter += 1
            return self.callback_message('dial.get_surrogate_values')

        raise IntersectCallbackError(operation, payload)

    def callback_message(self, operation: str, **kwargs) -> IntersectClientCallback:
        next_payload = None
        self.at_grids = kwargs.get('at_grids', True)

        if operation == 'dial.initialize_workflow':
            next_payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel=self.kernel,
                backend=self.backend,
                preprocess_standardize=True,
                y_is_good=True,
                seed=20,
            )

        elif operation == 'dial.get_surrogate_values':
            if self.at_grids:
                _points_to_predict = self.x_grid
            else:
                _points_to_predict = np.array(self.test_points).reshape(-1, self.num_dims)

            next_payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=_points_to_predict,
                kernel_args=self.kernel_args,
                extra_args={'length_per_dimension': True},
            )

        elif operation == 'dial.get_next_point':
            next_payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy=self.strategy,
                strategy_args=self.strategy_args,
                bounds=self.bounds.tolist(),
                kernel_args=self.kernel_args,
                extra_args={'length_per_dimension': True},
            )

        elif operation == 'dial.update_workflow_with_data':
            next_payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                next_x=self.dataset_x[-1],
                next_y=self.dataset_y[-1],
            )

        else:
            err_msg = f'Unknown operation received: {operation}'
            raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)

        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination, operation=operation, payload=next_payload
                )
            ]
        )

    def handle_surrogate_values(self, payload):
        if self.at_grids:
            self.variance_grid = np.array(payload[1]).reshape((self.meshgrid_size,) * self.num_dims)
            self.mean_grid = np.array(payload[0]).reshape((self.meshgrid_size,) * self.num_dims)
        else:
            self.variance_test = np.array(payload[1])
            self.mean_test = np.array(payload[0])
            print(f'Test Mean: {self.mean_test}, Variance: {self.variance_test}')

        # end of active learning loop after max_iter
        if self.niter > self.max_iter:
            raise IntersectCallbackEnd

    def handle_next_points(self, payload):
        self.x_next = payload
        coord_str = ', '.join([f'{coord:.2f}' for coord in self.x_next])
        print(f'Running simulation at ({coord_str}): ', end='', flush=True)

        y = sinusoidal_growth(*self.x_next)
        print(f'{y:.3f}')
        self.dataset_x.append(self.x_next)
        self.dataset_y.append(y)

        optpos = np.argmax(self.dataset_y)
        y_opt = self.dataset_y[optpos]
        optimal_coords = self.dataset_x[optpos]
        coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
        print(f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}\n')

    def graph(self):
        plt.clf()

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # First subplot: Mean and variance with training data
        axs[0].plot(self.x_grid, self.mean_grid, label='Mean Prediction')
        axs[0].fill_between(
            self.x_grid[:, 0],
            self.mean_grid + 2 * self.variance_grid,
            self.mean_grid - 2 * self.variance_grid,
            alpha=0.5,
            label='Confidence Interval',
        )
        axs[0].scatter(
            np.array(self.dataset_x)[:-1, 0],
            np.array(self.dataset_y)[:-1],
            color='black',
            marker='o',
            label='Training Data',
        )
        if self.x_next is not None:
            axs[0].axvline(x=self.x_next[0], color='red', linestyle='--')
        axs[0].set_ylabel('Response, y')
        axs[0].legend()
        axs[0].grid(True)

        # Second subplot: Acquisition function
        if self.strategy_args is not None:
            if self.mean_grid is not None and self.variance_grid is not None:
                exploit = self.strategy_args.get('exploit', 0.0)
                explore = self.strategy_args.get('explore', 1.0)
                acquisition_values = exploit * self.mean_grid + explore * np.sqrt(
                    self.variance_grid
                )
            else:
                acquisition_values = np.zeros_like(self.x_grid)

            axs[1].plot(self.x_grid, acquisition_values)
            if self.x_next is not None:
                axs[1].axvline(
                    x=self.x_next[0], color='red', linestyle='--', label='Next Point (x_next)'
                )
            axs[1].set_xlabel('Features, x')
            axs[1].set_ylabel('Acquisition Value')
            axs[1].legend()
            axs[1].grid(True)

        plt.tight_layout()
        plt.savefig('graph.png')
        plt.close(fig)


if __name__ == '__main__':
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
        initial_message_event_config=active_learning.callback_message('dial.initialize_workflow'),
        **from_config_file['intersect'],
    )

    client = IntersectClient(config=config, user_callback=active_learning)
    default_intersect_lifecycle_loop(client)
