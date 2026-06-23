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
    INTERSECT_RESPONSE_VALUE,
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
    Normal,
)

mpl.use('agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sigmoid_transition(
    x,
    low=-1.0,
    high=1.0,
    x0=0.0,
    width=0.01,
    slope=1.0,
):
    """
    Controlled sigmoid transition model.
    """
    x_arr = np.asarray(x, float)
    gate = 1.0 / (1.0 + np.exp(-(x_arr - x0) / width))
    value = low + (high - low) * gate
    if slope != 0.0:
        value = value - slope * np.maximum(x_arr - x0, 0.0) * gate
    return value


def truth_model(x, noise_level, rng):
    y = sigmoid_transition(x)
    y = y + noise_level * rng.normal(size=y.shape)
    y_err = noise_level * np.ones(y.shape)

    return y, y_err


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
        self.bounds = np.array([[-1.0, 1.0]])
        self.num_dims = len(self.bounds)
        self.seed = 5
        self.noise_level = 1e-4
        self.rng = np.random.default_rng(self.seed)

        self.x_raw = np.linspace(-1.0, 1.0, 5).reshape(-1, 1)
        self.x_test = np.array([[-0.051], [0.051]])
        self.meshgrid_size = 1000
        self.grid_points = [
            np.linspace(dim_bounds[0], dim_bounds[1], self.meshgrid_size)
            for dim_bounds in self.bounds
        ]
        self.meshgrids = np.meshgrid(*self.grid_points, indexing='ij')
        self.x_grid = np.hstack([mg.reshape(-1, 1) for mg in self.meshgrids])
        # Mirror the demo's RNG sequence before generating the noisy initial observations.
        truth_model(self.x_grid[:, 0], 0.0, self.rng)
        self.y_raw, _ = truth_model(self.x_raw[:, 0], self.noise_level, self.rng)

        self.dataset_x = self.x_raw.reshape(-1, 1).tolist()
        self.dataset_y = self.y_raw.reshape(-1).tolist()
        self.labels_y = ['y']
        self.statistics_y = Normal(loc='y', scale=self.noise_level)
        self.test_points = self.x_test.reshape(-1, 1).tolist()

        self.kernel = 'rbf'
        self.kernel_args = {'x_range': [0.0, 1.0], 'sigma_range': [2.5e-3, 0.5], 'gamma': 0.1}
        self.backend = 'sable'
        self.backend_args = {
            'n_features': 5000,
            'alpha': 0.05,
            'p': 1.25,
            'n_iter_irls': 100,
        }
        self.strategy = 'upper_confidence_bound'
        self.strategy_args = {'exploit': 0.0, 'explore': 1.0}
        self.niter = 0
        self.max_iter = 30
        self.at_grids = True
        self.stddev_grid = None
        self.mean_grid = None
        self.stddev_test = None
        self.mean_test = None
        self.x_next = None

        self.workflow_id = None
        self.service_destination = service_destination

    def __call__(
        self, _source: str, operation: str, _has_error: bool, payload: INTERSECT_RESPONSE_VALUE
    ) -> IntersectClientCallback:
        if _has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            raise IntersectCallbackError(operation, payload)

        if operation == 'dial.initialize_workflow':
            self.workflow_id = payload
            return self.callback_message('dial.get_surrogate_values')

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
                labels_y=self.labels_y,
                statistics_y=self.statistics_y,
                bounds=self.bounds.tolist(),
                kernel=self.kernel,
                kernel_args=self.kernel_args,
                backend=self.backend,
                backend_args=self.backend_args,
                preprocess_standardize=False,
                y_is_good=True,
                seed=self.seed,
                dim_x=self.num_dims,
            )

        elif operation == 'dial.get_surrogate_values':
            if self.at_grids:
                points_to_predict = self.x_grid
            else:
                points_to_predict = np.array(self.test_points).reshape(-1, self.num_dims)

            next_payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=points_to_predict,
            )

        elif operation == 'dial.get_next_point':
            next_payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy=self.strategy,
                strategy_args=self.strategy_args,
                bounds=self.bounds.tolist(),
                # To acquire data only on a grid, comment in the following
                # discrete_measurements=True,
                # discrete_measurement_grid_size=[50],
            )

        elif operation == 'dial.update_workflow_with_data':
            next_payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                next_x=self.dataset_x[-1],
                next_y=self.dataset_y[-1],
            )

        else:
            err_msg = f'Unknown operation received: {operation}'
            raise Exception(err_msg)  # noqa: TRY002

        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=operation,
                    payload=next_payload,
                )
            ]
        )

    def handle_surrogate_values(self, payload):
        means = payload['values']
        transformed_stddevs = payload['transformed_stddevs']
        if self.at_grids:
            self.stddev_grid = np.array(transformed_stddevs).reshape(
                (self.meshgrid_size,) * self.num_dims
            )
            self.mean_grid = np.array(means).reshape((self.meshgrid_size,) * self.num_dims)
        else:
            self.stddev_test = np.array(transformed_stddevs)
            self.mean_test = np.array(means)
            print(
                f'Values at testing points {self.x_test.reshape(-1)}: Mean: {self.mean_test}, Stddev: {self.stddev_test}'
            )

        if self.niter > self.max_iter:
            raise IntersectCallbackEnd

    def handle_next_points(self, payload):
        self.x_next = payload['data']
        coord_str = ', '.join([f'{coord:.2f}' for coord in self.x_next])
        print(f'Running simulation at ({coord_str}): ', end='', flush=True)

        y_next, _ = truth_model(np.asarray(self.x_next, dtype=float), self.noise_level, self.rng)
        y_scalar = float(np.asarray(y_next).reshape(-1)[0])
        print(f'{y_scalar:.3f}')

        self.dataset_x.append(self.x_next)
        self.dataset_y.append(y_scalar)

        # In this example we are running pure exploration, no optimization:
        # optpos = np.argmax(self.dataset_y)
        # y_opt = self.dataset_y[optpos]
        # optimal_coords = self.dataset_x[optpos]
        # coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
        # print(f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}\n')

    def graph(self):
        plt.clf()

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(self.x_grid[:, 0], self.mean_grid, label='Mean Prediction')
        axs[0].fill_between(
            self.x_grid[:, 0],
            self.mean_grid + 2 * self.stddev_grid,
            self.mean_grid - 2 * self.stddev_grid,
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

        acquisition_values = np.zeros_like(self.x_grid[:, 0])
        if self.mean_grid is not None and self.stddev_grid is not None:
            exploit = self.strategy_args.get('exploit', 0.0)
            explore = self.strategy_args.get('explore', 1.0)
            acquisition_values = exploit * self.mean_grid + explore * self.stddev_grid

        axs[1].plot(self.x_grid[:, 0], acquisition_values)
        if self.x_next is not None:
            axs[1].axvline(x=self.x_next[0], color='red', linestyle='--', label='Next Point')
        axs[1].set_xlabel('Features, x')
        axs[1].set_ylabel('Acquisition Value')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig('graph_sable.png')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated SABLE client')
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
        initial_message_event_config=active_learning.callback_message('dial.initialize_workflow'),
        **from_config_file['intersect'],
    )

    client = IntersectClient(config=config, user_callback=active_learning)
    default_intersect_lifecycle_loop(client)
