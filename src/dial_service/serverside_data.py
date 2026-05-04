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
        self.X_raw = np.array(data.dataset_x)
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
        self.backend_args = data.backend_args
        self.kernel_args = data.kernel_args
        self.extra_args = data.extra_args
        self.dim_x = data.dim_x

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

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X into [0, 1]^D using self.bounds.
        X: array of shape (N, D)
        """
        X = np.asarray(X, dtype=float)

        if X.size == 0:
            D = len(self.bounds)
            return np.empty((0, D))

        bounds = np.asarray(self.bounds, float)  # (D, 2)
        lows = bounds[:, 0]
        highs = bounds[:, 1]
        span = np.where(highs - lows == 0, 1.0, highs - lows)

        return (X - lows) / span

    @cached_property
    def X_train(self) -> np.ndarray:
        """
        Return X scaled to [0, 1] per dimension based on self.bounds.

        dataset_x: list[list[float]], shape (N, D)
        bounds: list[[low, high], ...], shape (D, 2)
        """
        return self._scale_X(self.X_raw)

    # undoes the preprocessing.
    def inverse_transform(self, data: np.ndarray, is_stddev: bool = False):

        if len(self.Y_raw) == 0:
            return data

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
        self.numpy_rng = np.random.RandomState(None if params.seed == -1 else params.seed)

        self.optimization_points = params.optimization_points
        self.confidence_bound = (
            params.confidence_bound if params.strategy == 'confidence_bound' else 0.0
        )
        self.discrete_measurements = params.discrete_measurements
        self.discrete_measurement_grid_size = params.discrete_measurement_grid_size

    def set_x_predict(self, X_raw: np.ndarray) -> None:
        """
        Store raw prediction points and their scaled version.
        X_raw: shape (N, D) or (D,) for a single point.
        """
        raw_vals = np.asarray(X_raw, dtype=float).reshape(-1, self.dim_x)
        self.x_predict = self._scale_X(raw_vals)


class ServersideInputMultiple(ServersideInputBase):
    def __init__(
        self, workflow_state: DialWorkflowCreationParamsService, params: DialInputMultiple
    ):
        super().__init__(workflow_state)
        self.strategy = params.strategy
        self.points = params.points
        self.strategy = params.strategy
        self.strategy_args = params.strategy_args
        self.y_is_good = params.y_is_good
        self.bounds = params.bounds
        self.numpy_rng = np.random.RandomState(None if params.seed == -1 else params.seed)

        self.optimization_points = params.optimization_points
        self.confidence_bound = (
            params.confidence_bound if params.strategy == 'confidence_bound' else 0.0
        )
        self.discrete_measurements = params.discrete_measurements
        self.discrete_measurement_grid_size = params.discrete_measurement_grid_size

    def set_x_predict(self, X_raw: np.ndarray) -> None:
        """
        Store raw prediction points and their scaled version.
        X_raw: shape (N, D) or (D,) for a single point.
        """
        raw_vals = np.asarray(X_raw, dtype=float).reshape(-1, self.dim_x)
        self.x_predict = self._scale_X(raw_vals)

class ServersideInputPrediction(ServersideInputBase):
    def __init__(
        self, workflow_state: DialWorkflowCreationParamsService, params: DialInputPredictions
    ):
        super().__init__(workflow_state)
        self.x_predict_raw = np.asarray(params.points_to_predict, dtype=float)
        self.set_x_predict(self.x_predict_raw)

    def set_x_predict(self, X_raw: np.ndarray) -> None:
        """
        Store raw prediction points and their scaled version.
        X_raw: shape (N, D) or (D,) for a single point.
        """
        raw_vals = np.asarray(X_raw, dtype=float).reshape(-1, self.dim_x)
        self.x_predict = self._scale_X(raw_vals)
