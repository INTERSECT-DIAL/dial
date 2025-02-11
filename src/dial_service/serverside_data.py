from functools import cached_property

import numpy as np

from dial_dataclass import (
    DialInputBase,
    DialInputMultiple,
    DialInputPredictions,
    DialInputSingle,
)


# this is an extended version of ActiveLearningInputData.  This allows us to add on properties and methods to this class without impacting the client side
class ServersideInputBase:
    def __init__(self, data: DialInputBase):
        self.X_train = np.array(data.dataset_x)
        self.Y_raw = np.array(data.dataset_y)
        # it seems like there should be a smarter way to do this, but stuff involving loops doesn't work with static autocompleters:
        self.bounds = data.bounds
        self.y_is_good = data.y_is_good
        self.kernel = data.kernel
        self.length_per_dimension = data.length_per_dimension
        self.backend = data.backend
        self.seed = data.seed
        self.preprocess_log = data.preprocess_log
        self.preprocess_standardize = data.preprocess_standardize

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
    def __init__(self, data: DialInputSingle):
        super().__init__(data)
        self.strategy = data.strategy
        self.optimization_points = data.optimization_points
        self.confidence_bound = data.confidence_bound
        self.discrete_measurements = data.discrete_measurements
        self.discrete_measurement_grid_size = data.discrete_measurement_grid_size


class ServersideInputMultiple(ServersideInputBase):
    def __init__(self, data: DialInputMultiple):
        super().__init__(data)
        self.strategy = data.strategy
        self.points = data.points


class ServersideInputPrediction(ServersideInputBase):
    def __init__(self, data: DialInputPredictions):
        super().__init__(data)
        self.x_predict = np.array(data.points_to_predict)
