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

# Priors passed to andie_backend_general via extra_args.
# Structure: {param_name: {guess, std, limits}} — matches get_prior_distributions().
_PRIORS = {
    'TN': {'guess': 250.0, 'std': 80.0,  'limits': [90.0,  500.0]},
    'M0': {'guess': 18.0,  'std': 3.0,   'limits': [0.01,  35.0]},
    'J':  {'guess': 0.6,   'std': 0.5,   'limits': [0.01,  14.0]},
    'BK': {'guess': 75.0,  'std': 50.0,  'limits': [0.0,   150.0]},
}

_MODEL_NAME = 'ANDIE_second_order_I_vs_T'


def peak_val_at_T(T: np.ndarray) -> np.ndarray:
    result = 70.0 * 1.0/(1.0+np.exp((T-200)/7)) + 30.0
    logger.debug(result)
    return result


class IntersectCallbackError(Exception):
    def __init__(self, operation, payload):
        message = f"Intersect callback error during operation '{operation}'. Payload: {payload}"
        super().__init__(message)

class IntersectCallbackEnd(Exception):
    def __init__(self):
        message = "Stopping Intersect Calls"
        super().__init__(message)

class ActiveLearningOrchestrator:

    def __init__(self, service_destination: str):
        Temperature_Loops = 30

        T_start = 100.0  # Kelvin
        T_stop  = 500.0  # Kelvin
        T_step  = 0.5    # Kelvin

        T_grid = np.linspace(T_start, T_stop, int((T_stop-T_start)/T_step)+1).reshape(-1, 1)

        self.T_step   = T_step
        self.above_TN = 0
        self.last_idx = 0

        self.bounds   = np.array([[T_start, T_stop]])
        self.num_dims = len(self.bounds)

        self.plot_results = True

        self.x_raw  = np.array([[T_start]])
        self.x_test = np.array(T_grid)
        self.y_raw  = peak_val_at_T(self.x_raw)

        self.meshgrid_size = len(T_grid)

        self.dataset_x   = self.x_raw.reshape(-1, 1).tolist()
        self.dataset_y   = self.y_raw.reshape(-1).tolist()
        self.test_points = self.x_test.reshape(-1, 1).tolist()

        self.kernel       = None
        self.kernel_args  = None
        self.backend      = 'andie_general'
        self.backend_args = None
        self.strategy     = None
        self.strategy_args = None
        self.niter    = 0
        self.max_iter = Temperature_Loops
        self.x_next   = None

        # Static extra_args stored in the workflow so every backend call has access
        # to model_name and priors without re-sending them each request.
        self._static_extra_args = {
            'model_name': _MODEL_NAME,
            'priors': _PRIORS,
        }

        self.workflow_id = None
        self.service_destination = service_destination
        self.initialize_workflow_message = IntersectClientCallback(messages_to_send=[
            IntersectDirectMessageParams(
                destination=self.service_destination,
                operation='dial.initialize_workflow',
                payload=DialWorkflowCreationParamsClient(
                    dataset_x=self.dataset_x,
                    dataset_y=self.dataset_y,
                    bounds=self.bounds,
                    kernel=self.kernel,
                    backend=self.backend,
                    preprocess_standardize=False,
                    y_is_good=True,
                    seed=20,
                    extra_args=self._static_extra_args,
                ))
            ])

    def _dynamic_extra_args(self, include_T_step: bool = False) -> dict:
        """Build the per-request extra_args carrying mutable state.

        These are merged server-side with the stored workflow extra_args (which
        holds model_name and priors), so only the fields that change each
        iteration need to be sent here.
        """
        args = {
            'above_TN': self.above_TN,
            'last_idx': self.last_idx,
        }
        if include_T_step:
            args['T_step'] = self.T_step
        return args

    def callback(self, operation: str) -> IntersectClientCallback:
        print("send", operation)

        next_payload = None

        if operation == 'dial.get_surrogate_values':
            _points_to_predict = np.array(self.test_points).reshape(-1, self.num_dims)
            next_payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=_points_to_predict,
                extra_args=self._dynamic_extra_args(),
            )

        elif operation == 'dial.get_next_point':
            next_payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy=self.strategy,
                strategy_args=self.strategy_args,
                bounds=self.bounds.tolist(),
                extra_args=self._dynamic_extra_args(include_T_step=True),
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

        return IntersectClientCallback(messages_to_send=[
            IntersectDirectMessageParams(
                destination=self.service_destination,
                operation=operation,
                payload=next_payload)
        ])

    def __call__(self, _source: str, operation: str, _has_error: bool,
                 payload: INTERSECT_JSON_VALUE) -> IntersectClientCallback:
        print(operation)
        if _has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            raise IntersectCallbackError(operation, payload)

        if operation == 'dial.initialize_workflow':
            self.workflow_id = payload
            print("\n", "--"*20, "\n")
            self.niter += 1
            return self.callback('dial.get_next_point')

        elif operation == 'dial.get_next_point':
            self.handle_next_points(payload)
            return self.callback('dial.update_workflow_with_data')

        elif operation == 'dial.update_workflow_with_data':
            self.niter += 1
            return self.callback('dial.get_next_point')

        else:
            raise IntersectCallbackError(operation, payload)

    def handle_surrogate_values(self, payload):
        self.variance_grid = np.array(payload[1]).reshape(
            (self.meshgrid_size,) * self.num_dims)
        self.mean_grid = np.array(payload[0]).reshape(
            (self.meshgrid_size,) * self.num_dims)

        if self.niter > self.max_iter:
            raise IntersectCallbackEnd()

    def handle_next_points(self, payload):
        print(payload)
        self.x_next   = [payload[0]]
        self.above_TN = payload[1]
        self.last_idx = payload[2]

        print(f'Running simulation at ({self.x_next}): ', end='', flush=True)
        y = peak_val_at_T(*self.x_next)
        print(f'{y:.3f}')
        print(f'Adding ({self.x_next}, {y}) to dataset')

        self.dataset_x.append(self.x_next)
        self.dataset_y.append(y)

        optpos = np.argmax(self.dataset_y)
        y_opt  = self.dataset_y[optpos]
        optimal_coords = self.dataset_x[optpos]
        coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
        print(f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}\n')

        if self.plot_results:
            plt.figure()
            x_plot = np.linspace(0, 600, 300)
            plt.plot(x_plot, peak_val_at_T(x_plot))
            plt.plot(self.dataset_x, self.dataset_y, 'o')
            plt.savefig('andie_general.png')

        if self.niter >= self.max_iter:
            print("Maximum iteration reached")
            raise IntersectCallbackEnd()
        if self.last_idx == len(self.x_test) - 1:
            print("Last grid point reached")
            raise IntersectCallbackEnd()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated client for andie_general backend')
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
        initial_message_event_config=active_learning.initialize_workflow_message,
        **from_config_file['intersect'],
    )

    client = IntersectClient(config=config, user_callback=active_learning)
    default_intersect_lifecycle_loop(client)
