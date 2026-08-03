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
        self.dim_x = data.dim_x
        self.dim_y = data.dim_y
        self.labels_x = data.labels_x
        self.labels_y = data.labels_y
        self.dataset_x = np.array(data.dataset_x)
        self.dataset_y = np.array(data.dataset_y).reshape((-1, self.dim_y))
        self.statistics_y = data.statistics_y
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

    @cached_property
    def X_train(self) -> np.ndarray:
        """
        Return X scaled to [0, 1] per dimension based on self.bounds.

        dataset_x: list[list[float]], shape (N, D)
        bounds: list[[low, high], ...], shape (D, 2)
        """
        return self.scale_X(self.dataset_x)

    def scale_X(self, X: np.ndarray) -> np.ndarray:
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
    def y_train_raw(self) -> np.ndarray:
        """
        Return the raw training target values extracted from the dataset.
        """
        y_label = self.statistics_y.loc
        if not isinstance(y_label, str):
            msg = 'statistics_y.loc must be a Label (str).'
            raise TypeError(msg)

        # Use the label from self.statistics_y.loc to find the data column with the mean y data
        # this may trigger a ValueError, if the label does not exist, but should be handled by dataclass validation
        pos_y = self.labels_y.index(y_label)
        return self.dataset_y[:, pos_y]

    @cached_property
    def yerr_train_raw(self) -> any:
        """
        Return the raw training error values extracted from the dataset.
        """
        yerr_label = self.statistics_y.scale
        if isinstance(yerr_label, float):
            _yerr_train_raw = yerr_label
        else:
            # yerr_label is str
            # this may trigger a ValueError, but should be handled by dataclass validation
            pos_yerr = self.labels_y.index(yerr_label)
            _yerr_train_raw = self.dataset_y[:, pos_yerr]

        if np.any(_yerr_train_raw < 0):
            idxs = np.where(_yerr_train_raw < 0)
            msg = f'yerr values in statistics_y.scale must be non-negative, found {_yerr_train_raw[idxs[0]]} at {idxs[0]}.'
            raise ValueError(msg)
        return _yerr_train_raw

    @cached_property
    def Y_train(self) -> np.ndarray:
        """
        Find output y and error values yerr in dataset_y, and apply transformation.
        Return transformed y value.
        """
        y, _ = self.transform_Y(self.y_train_raw, self.yerr_train_raw)

        # return only y, to conform to interface
        return y

    @cached_property
    def Yerr_train(self) -> any:
        """
        Find output y and error values in dataset y, and apply transformation.
        Return transformed yerr value.
        """
        # recompute transformation, at some overhead (probably not worth to optimize)
        _, yerr = self.transform_Y(self.y_train_raw, self.yerr_train_raw)

        # return only yerr, to conform to interface
        return yerr

    def _transform_Y_params(self) -> tuple[float, float]:
        """
        Return the appropriate mean and scaling of the raw y data for normalization
        """
        # find y_std from y_train_raw
        y_train = self.y_train_raw
        if len(y_train) > 0 and self.preprocess_standardize:
            if self.preprocess_log:
                y_train = np.log(y_train)
            y_std = np.std(y_train)
            y_mean = np.mean(y_train)
        else:
            y_std = 1.0
            y_mean = 0.0

        return y_mean, y_std

    def transform_Y(self, y: np.ndarray, yerr: any) -> tuple[np.ndarray, any]:
        """
        Transform y and yerr according to preprocess options
        """
        if self.preprocess_log:
            yerr = yerr / y
            y = np.log(y)

        if self.preprocess_standardize:
            y_mean, y_std = self._transform_Y_params()
            yerr = yerr / y_std
            y = (y - y_mean) / y_std

        return y, yerr

    def inverse_transform_Y(self, y: np.ndarray, yerr: any) -> tuple[np.ndarray, any]:
        """
        Inverse transforms of y and yerr, in reverse order
        """
        if self.preprocess_standardize:
            y_mean, y_std = self._transform_Y_params()
            y = y_mean + y_std * y
            yerr = y_std * yerr

        if self.preprocess_log:
            y = np.exp(y)
            yerr = y * yerr

        return y, yerr

    @cached_property
    def Y_best(self) -> float:
        return self.Y_train.max() if self.y_is_good else self.Y_train.min()

    def clear_cached_properties(self) -> None:
        # Track attribute names that have already been encountered.
        # Classes are inspected in Method Resolution Order (MRO) order,
        # starting with the most-derived class
        resolved_names = set()

        # Walk through the class hierarchy:
        # DerivedClass -> BaseClass -> ... -> object
        for cls in type(self).__mro__:
            # Inspect only attributes defined directly on the current class
            for name, attr in cls.__dict__.items():
                # Skip names already defined by a more-derived class
                if name in resolved_names:
                    continue

                resolved_names.add(name)

                # A cached_property descriptor is stored on the class,
                # while its computed value is stored in the instance __dict__
                if isinstance(attr, cached_property):
                    # Remove the cached value if it exists
                    self.__dict__.pop(name, None)


class ServersideInputSingle(ServersideInputBase):
    def __init__(
        self,
        workflow_state: DialWorkflowCreationParamsService,
        params: DialInputSingle,
    ):
        super().__init__(workflow_state)
        # set new inputs
        self.strategy = params.strategy
        self.strategy_args = params.strategy_args
        self.optimization_points = params.optimization_points
        self.confidence_bound = (
            params.confidence_bound if params.strategy == 'confidence_bound' else 0.0
        )
        # if params.strategy == 'confidence_bound':
        #    self.confidence_bound = params.confidence_bound
        # elif self.strategy_args is not None and 'confidence_bounds' in self.strategy_args:
        #    self.confidence_bound = params.strategy_args['confidence_bound']
        self.discrete_measurements = params.discrete_measurements
        self.discrete_measurement_grid_size = params.discrete_measurement_grid_size
        self.point_index = params.point_index

        # update values from workflow initialization, if provided
        if params.extra_args is not None:
            if self.extra_args is not None:
                self.extra_args.update(params.extra_args)
            else:
                self.extra_args = params.extra_args
        if params.y_is_good is not None:
            self.y_is_good = params.y_is_good
        if params.bounds is not None:
            self.bounds = params.bounds

        # always reinit rng, since initial rng is not updated in db!
        self.numpy_rng = np.random.RandomState(None if params.seed == -1 else params.seed)

    def set_x_predict(self, X_raw: np.ndarray) -> None:
        """
        Store raw prediction points and their scaled version.
        X_raw: shape (N, D) or (D,) for a single point.
        """
        raw_vals = np.asarray(X_raw, dtype=float).reshape(-1, self.dim_x)
        self.x_predict = self.scale_X(raw_vals)


class ServersideInputMultiple(ServersideInputBase):
    def __init__(
        self,
        workflow_state: DialWorkflowCreationParamsService,
        params: DialInputMultiple,
    ):
        # set new inputs
        super().__init__(workflow_state)
        self.strategy = params.strategy
        self.points = params.points
        self.strategy = params.strategy
        self.strategy_args = params.strategy_args
        self.optimization_points = params.optimization_points
        self.confidence_bound = (
            params.confidence_bound if params.strategy == 'confidence_bound' else 0.0
        )
        self.discrete_measurements = params.discrete_measurements
        self.discrete_measurement_grid_size = params.discrete_measurement_grid_size

        # update values from workflow initialization, if provided
        if params.extra_args is not None:
            if self.extra_args is not None:
                self.extra_args.update(params.extra_args)
            else:
                self.extra_args = params.extra_args
        if params.y_is_good is not None:
            self.y_is_good = params.y_is_good
        if params.bounds is not None:
            self.bounds = params.bounds

        # always reinit rng, since initial rng is not updated in db!
        self.numpy_rng = np.random.RandomState(None if params.seed == -1 else params.seed)

    def set_x_predict(self, X_raw: np.ndarray) -> None:
        """
        Store raw prediction points and their scaled version.
        X_raw: shape (N, D) or (D,) for a single point.
        """
        raw_vals = np.asarray(X_raw, dtype=float).reshape(-1, self.dim_x)
        self.x_predict = self.scale_X(raw_vals)


class ServersideInputPrediction(ServersideInputBase):
    def __init__(
        self,
        workflow_state: DialWorkflowCreationParamsService,
        params: DialInputPredictions,
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
        self.x_predict = self.scale_X(raw_vals)
