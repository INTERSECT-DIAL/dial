from functools import cached_property

import numpy as np

from dial_dataclass import (
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingle,
)

from .service_specific_dataclasses import DialWorkflowCreationParamsService


# this is an extended version of ActiveLearningInputData.  This allows us to add on properties and methods to this class without impacting the client side
class ServersideInputBase:
    def __init__(self, data: DialWorkflowCreationParamsService):
        self.X_train = np.array(data.dataset_x)
        self.Y_raw = np.array(data.dataset_y)
        # it seems like there should be a smarter way to do this, but stuff involving loops doesn't work with static autocompleters:
        self.bounds = data.bounds
        self.y_is_good = data.y_is_good
        self.kernel = data.kernel
        self.backend: str = data.backend
        self.seed = data.seed
        self.numpy_rng = np.random.RandomState(None if data.seed == -1 else data.seed)
        self.preprocess_log = data.preprocess_log
        self.preprocess_standardize = data.preprocess_standardize
        # TODO should probably just allow for args to be passed through the constructor
        self.backend_args = None
        self.kernel_args = None
        self.extra_args = None

    @cached_property
    def stddev(self) -> float:
        return np.std(self.Y_train)

    @cached_property
    def Y_best(self) -> float:
        return self.Y_train.max() if self.y_is_good else self.Y_train.min()

    @cached_property
    def Y_train(self) -> np.ndarray:
        y = self.Y_raw
        if self.preprocess_log:
            y = np.log(y)
        if self.preprocess_standardize:
            y = (y - np.mean(y)) / np.std(y)
        return y

    # undoes the preprocessing.
    def inverse_transform(self, data: np.ndarray, is_stddev: bool = False):
        # not possible to un-log the standard deviations (-1 +- 1 in log space != .1 +- 10 in realspace)
        if self.preprocess_log and is_stddev:
            return np.repeat(-1, len(data))
        if self.preprocess_standardize:
            # the data that was used to calculate the standardization:
            prestandardized_y = np.log(self.Y_raw) if self.preprocess_log else self.Y_raw
            data = data * np.std(prestandardized_y)  # not the same as *= (which is in-place)
            if not is_stddev:
                data = data + np.mean(prestandardized_y)
        if self.preprocess_log:
            data = np.exp(data)
        return data


class ServersideInputSingle(ServersideInputBase):
    def __init__(self, workflow_state: DialWorkflowCreationParamsService, params: DialInputSingle):
        super().__init__(workflow_state)
        self.strategy = params.strategy
        self.strategy_args = params.strategy_args
        self.y_is_good = params.y_is_good
        self.bounds = params.bounds
        self.kernel_args = params.kernel_args
        self.backend_args = params.backend_args
        self.extra_args = params.extra_args
        self.numpy_rng = np.random.RandomState(None if params.seed == -1 else params.seed)

        self.optimization_points = params.optimization_points
        self.confidence_bound = (
            params.confidence_bound if params.strategy == 'confidence_bound' else 0.0
        )
        self.discrete_measurements = params.discrete_measurements
        self.discrete_measurement_grid_size = params.discrete_measurement_grid_size


class ServersideInputMultiple(ServersideInputBase):
    def __init__(
        self, workflow_state: DialWorkflowCreationParamsService, params: DialInputMultiple
    ):
        super().__init__(workflow_state)
        self.strategy = params.strategy
        self.points = params.points


class ServersideInputPrediction(ServersideInputBase):
    def __init__(
        self, workflow_state: DialWorkflowCreationParamsService, params: DialInputPredictions
    ):
        super().__init__(workflow_state)
        self.x_predict = np.array(params.points_to_predict)
        self.kernel_args = params.kernel_args
        self.backend_args = params.backend_args
        self.extra_args = params.extra_args
