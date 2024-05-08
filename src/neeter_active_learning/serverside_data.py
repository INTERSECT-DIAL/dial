import numpy as np
from functools import cached_property
from .data_class import BoalasInputBase, BoalasInputSingle, BoalasInputMultiple

#this is an extended version of ActiveLearningInputData.  This allows us to add on properties and methods to this class without impacting the client side
class ServersideInputBase:
    def __init__(self, data: BoalasInputBase):
        self.X_train = np.array(data.dataset_x)
        self.Y_raw = np.array(data.dataset_y)
        #it seems like there should be a smarter way to do this, but stuff involving loops doesn't work with static autocompleters:
        self.bounds = data.bounds
        self.y_is_good = data.y_is_good
        self.kernel = data.kernel
        self.length_per_dimension = data.length_per_dimension
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
            y = (y - np.mean(y))/np.std(y)
        return y

class ServersideInputSingle(ServersideInputBase):
    def __init__(self, data: BoalasInputSingle):
        super().__init__(data)
        self.strategy = data.strategy
        self.confidence_bound = data.confidence_bound

class ServersideInputMultiple(ServersideInputBase):
    def __init__(self, data: BoalasInputMultiple):
        super().__init__(data)
        self.strategy = data.strategy
        self.points = data.points
