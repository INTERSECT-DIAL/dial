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

def peak_val_at_T(T: np.ndarray) -> np.ndarray:
    result = 40.0 * 1.0/(1.0+np.exp((T-50)/5)) + 35.0
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
        #Maximum Measurments in Temperature:
        Temperature_Loops = 30

        T_start = 5.0 #Kelvin
        T_stop = 300.0 #Kelvin
        T_step = .5 #Kelvin
        T_grid = np.linspace(T_start, T_stop, int((T_stop-T_start)/T_step)+1).reshape(-1,1)

        self.T_step = T_step
        above_TN = 0


        self.above_TN = above_TN
        self.last_idx = 0

        self.bounds = np.array([[T_start, T_stop]])
        self.num_dims = len(self.bounds)


        self.plot_results = True

        self.x_raw = np.array([[ T_start]])
        self.x_test = np.array(T_grid)
        self.y_raw = peak_val_at_T(self.x_raw)

        self.meshgrid_size = len(T_grid)
        # self.grid_points = [
        #     np.linspace(dim_bounds[0], dim_bounds[1], self.meshgrid_size)
        #     for dim_bounds in self.bounds
        # ]
        # self.meshgrids = np.meshgrid(*self.grid_points, indexing='ij')
        # self.x_grid = np.hstack([mg.reshape(-1, 1) for mg in self.meshgrids])

        # Active learning variables
        self.dataset_x = self.x_raw.reshape(-1, 1).tolist()
        print(self.dataset_x)
        self.dataset_y = self.y_raw.reshape(-1).tolist()
        self.test_points = self.x_test.reshape(-1, 1).tolist()

        self.kernel = None
        self.kernel_args = None
        self.backend = 'andie'
        self.backend_args = None
        self.strategy = None
        self.strategy_args = None
        self.niter = 0
        self.max_iter = Temperature_Loops
        self.at_grids = False
        self.variance_grid = None
        self.mean_grid = None
        self.variance_test = None
        self.mean_test = None
        self.x_next = None

        # Intersect variables
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
                        seed=20))
            ])

    def callback(self, operation: str, **kwargs) -> IntersectClientCallback:
        print("send", operation)

        next_payload = None
        # self.at_grids = kwargs.get('at_grids', True)

        if operation == 'dial.get_surrogate_values':

            _points_to_predict = np.array(self.test_points).reshape(-1, self.num_dims)

            next_payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=_points_to_predict,
                kernel_args=self.kernel_args,
                extra_args={'above_TN': self.above_TN,
                            'last_idx': self.last_idx},
            )

        elif operation == 'dial.get_next_point':
            next_payload = DialInputSingleOtherStrategy(
                            workflow_id=self.workflow_id,
                            strategy=self.strategy,
                            strategy_args=self.strategy_args,
                            bounds=self.bounds.tolist(),
                            kernel_args=self.kernel_args,
                            extra_args={'above_TN': self.above_TN,
                                        'last_idx': self.last_idx,
                                        'T_step': self.T_step},
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
            print("\n","--"*20,"\n")
            # return self.callback('dial.get_surrogate_values')
            self.niter += 1

            return self.callback('dial.get_next_point')
            
        # TODO: repplace this with update_workflow_with_data that can also train the model

        # ----------------- Active learning loop -----------------
        # if operation == 'dial.get_surrogate_values':
        #     self.handle_surrogate_values(payload)

        #     if self.at_grids:
        #         print(f"Step {self.niter}")
        #         return self.callback('dial.get_surrogate_values', at_grids=False)
        #     else:
        #         return self.callback('dial.get_next_point')

        elif operation == 'dial.get_next_point':
            self.handle_next_points(payload)
            return self.callback('dial.update_workflow_with_data')

        elif operation == 'dial.update_workflow_with_data':
            self.niter += 1
            return self.callback('dial.get_next_point')

        else:
            raise IntersectCallbackError(operation, payload)

    def handle_surrogate_values(self, payload):

        self.variance_grid = np.array(payload[1]).reshape(\
            (self.meshgrid_size,) * self.num_dims)
        self.mean_grid = np.array(payload[0]).reshape(\
            (self.meshgrid_size,) * self.num_dims)

        # end of active learning loop after max_iter
        if self.niter > self.max_iter:
            raise IntersectCallbackEnd()

    def handle_next_points(self, payload):

        print(payload)
        self.x_next = [payload[0]]
        self.above_TN = payload[1]
        self.last_idx = payload[2]

        print(f'Running simulation at ({self.x_next}): ', end='', flush=True)

        y = peak_val_at_T(*self.x_next)
        print(f'{y:.3f}')
        print(f'Adding ({self.x_next}, {y}) to dataset')
        print(f'Current dataset: {self.dataset_x}')
        print(f'Current dataset: {self.dataset_y}')

        self.dataset_x.append(self.x_next)
        self.dataset_y.append(y)

        print(f'new dataset: {self.dataset_x}')
        print(f'new dataset: {self.dataset_y}')

        optpos = np.argmax(self.dataset_y)
        y_opt = self.dataset_y[optpos]
        optimal_coords = self.dataset_x[optpos]
        coord_str = ', '.join([f'{coord:.2f}' for coord in optimal_coords])
        print(f'Optimal simulated datapoint at ({coord_str}), y={y_opt:.3f}\n')


        if self.plot_results:
            plt.figure()
            x_plot = np.linspace(0,310,300)
            plt.plot(x_plot,peak_val_at_T(x_plot))
            plt.plot(self.dataset_x, self.dataset_y, 'o')
            plt.savefig('andie.png')
        
        # end client when maximum iteration reached
        if self.niter >= self.max_iter:
            print("Maximum iteration reached")
            raise IntersectCallbackEnd()
        # end client when last point on the grid reached
        if self.last_idx == len(self.x_test) - 1:
            print("Last grid point reached")
            raise IntersectCallbackEnd()
        
    

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
        initial_message_event_config=active_learning.initialize_workflow_message,
        **from_config_file['intersect'],
    )

    client = IntersectClient(config=config, user_callback=active_learning)
    default_intersect_lifecycle_loop(client)
