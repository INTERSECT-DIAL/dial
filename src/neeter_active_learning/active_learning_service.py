import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Callable, get_type_hints
from functools import cached_property

from intersect_sdk import (
    IntersectBaseCapabilityImplementation,
    intersect_message,
    intersect_status
)

from .data_class import ActiveLearningInputData

logger = logging.getLogger(__name__)

#this is an extended version of ActiveLearningInputData.  This allows us to add on properties and methods to this class without impacting the client side
class ServersideInputData:
    def __init__(self, data: ActiveLearningInputData):
        self.X_train = np.array(data.dataset_x)
        self.Y_raw = np.array(data.dataset_y)
        #it seems like there should be a smarter way to do this, but stuff involving doesn't work with static autocompleters:
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

class ActiveLearningServiceCapabilityImplementation(IntersectBaseCapabilityImplementation):

    '''
    Internal guts for GP usage:
    '''
    #trains a model based on the user's data
    def _train_model(self, data: ServersideInputData) -> GaussianProcessRegressor:
        model = GaussianProcessRegressor(kernel=self._kernel(data), n_restarts_optimizer=250)
        model.fit(data.X_train, data.Y_train)
        return model

    _KERNELS = {"rbf": RBF, "matern": Matern}
    #parses the user's requested kernel
    def _kernel(self, data: ServersideInputData):
        kernel_name = data.kernel.lower()
        if kernel_name not in self._KERNELS:
            raise ValueError(f'Unknown kernel {kernel_name}')
        length_scale = [1.0]*len(data.X_train[0]) if data.length_per_dimension else 1.0
        return self._KERNELS[kernel_name](length_scale=length_scale)
    
    #looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
    def _create_n_dim_grid(self, data: ServersideInputData, points_per_dim):
        meshgrid = np.meshgrid(*(np.linspace(low, high, points_per_dim) for low, high in data.bounds), indexing='ij')
        return np.column_stack([arr.flatten() for arr in meshgrid])
        
    def _minimize(self, data: ServersideInputData, f: Callable[[np.ndarray], float]) -> np.ndarray:
        guess = min(self._create_n_dim_grid(data, 11), key=f)
        return minimize(f, guess, bounds=data.bounds, method="L-BFGS-B").x

    '''
    Endpoints users can hit:
    '''
    @intersect_message()
    #returns the point with the highest standard deviation
    def next_point_by_uncertainty(self, client_data: ActiveLearningInputData) -> list[float]:
        data = ServersideInputData(client_data)
        model = self._train_model(data)
        neg_stddev = lambda x: -1*model.predict(x.reshape(1, -1), return_std=True)[1][0]
        return self._minimize(data, neg_stddev).tolist()

    @intersect_message()
    #returns the point with the highest Expected Improvement
    def next_point_by_EI(self, client_data: ActiveLearningInputData) -> list[float]:
        data = ServersideInputData(client_data)

        print(f"Running EI on {len(data.X_train)} datapoints")

        model = self._train_model(data)
        def EI(x): #if y is good, this is actually -1*EI, since we have a minimizer
            mean, sigma = model.predict(x.reshape(1, -1), return_std=True)
            mean, sigma = mean[0], data.stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            if data.y_is_good:
                z = (mean - data.Y_best)/sigma
            else:
                z = (data.Y_best - mean)/sigma
            return -sigma*(z * norm.cdf(z) + norm.pdf(z))
        return self._minimize(data, EI).tolist()
    
    @intersect_message()
    #trains a model and then returns a grid of the predicted means (for graphing/display purposes)
    def mean_grid(self, client_data: ActiveLearningInputData) -> list[list[float]]:
        data = ServersideInputData(client_data)

        model = self._train_model(data)
        xx, yy = np.meshgrid(np.linspace(data.bounds[0][0], data.bounds[0][1], 101), np.linspace(data.bounds[1][0], data.bounds[1][1], 101))
        X_data = np.c_[xx.ravel(), yy.ravel()]
        mean_data = model.predict(X_data).reshape(xx.shape).tolist()
        return mean_data

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
