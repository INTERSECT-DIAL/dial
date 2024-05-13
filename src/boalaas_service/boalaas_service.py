import logging
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Union

from intersect_sdk import (
    IntersectBaseCapabilityImplementation,
    intersect_message,
    intersect_status
)

from boalaas_dataclass import BOALaaSInputSingle, BOALaaSInputMultiple, BOALaaSInputPredictions
from .serverside_data import *

logger = logging.getLogger(__name__)

class BOALaaSCapabilityImplementation(IntersectBaseCapabilityImplementation):

    '''
    Internal guts for GP usage:
    '''
    #trains a model based on the user's data
    def _train_model(self, data: ServersideInputBase) -> GaussianProcessRegressor:
        model = GaussianProcessRegressor(kernel=self._kernel(data), n_restarts_optimizer=250)
        model.fit(data.X_train, data.Y_train)
        return model

    _KERNELS = {"rbf": RBF, "matern": Matern}
    #parses the user's requested kernel
    def _kernel(self, data: ServersideInputBase):
        kernel_name = data.kernel.lower()
        if kernel_name not in self._KERNELS:
            raise ValueError(f'Unknown kernel {kernel_name}')
        length_scale = [1.0]*len(data.X_train[0]) if data.length_per_dimension else 1.0
        return self._KERNELS[kernel_name](length_scale=length_scale)
    
    #looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
    def _create_n_dim_grid(self, data: ServersideInputBase, points_per_dim: Union[int,list[int]]):
        if isinstance(points_per_dim, int):
            points_per_dim = [points_per_dim]*len(data.bounds)
        meshgrid = np.meshgrid(*(np.linspace(low, high, pts)
                                 for (low, high), pts in zip(data.bounds, points_per_dim)), indexing='ij')
        return np.column_stack([arr.flatten() for arr in meshgrid])

    '''
    Endpoints users can hit:
    '''
    @intersect_message()
    # trains a model and then recommends a point to measure based on user's requested strategy:
    def next_point(self, client_data: BOALaaSInputSingle) -> list[float]:
        data = ServersideInputSingle(client_data)
        model = self._train_model(data)
        # generate the function that gives -1 * the "value" of measuring a point with the given mean & stddev
        negative_value = None
        match data.strategy:
            case "uncertainty":
                def negative_value(_mean: float, stddev: float):
                    return -stddev
            case "expected_improvement":
                def negative_value(mean: float, stddev: float):
                    z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
                    return -stddev * (z*norm.cdf(z) + norm.pdf(z))
            case "confidence_bound":
                #calculate the z score that corresponds to the confidence bound:
                Z_VALUE = norm.ppf(.5 + data.confidence_bound/2) #need this because this is the "inverse CDF" which is one-tailed, but we want two-tailed
                def negative_value(mean: float, stddev: float):
                    #y_is_good: maximize mean + z*stddev, so minimize -mean - z*stdev
                    #y_is_bad:  minimize mean - z*stddev
                    return -Z_VALUE*stddev + mean*(-1 if data.y_is_good else 1)
        # create a function that can be minimized over the bounds:
        def to_minimize(x):
            mean, sigma = model.predict(x.reshape(1, -1), return_std=True)
            mean, sigma = mean[0], data.stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            return negative_value(mean, sigma)
        guess = min(self._create_n_dim_grid(data, 11), key=to_minimize)
        return minimize(to_minimize, guess, bounds=data.bounds, method="L-BFGS-B").x
    
    @intersect_message
    def next_points(self, client_data: BOALaaSInputMultiple) -> list[list[float]]:
        data = ServersideInputMultiple(client_data)
        #model = self._train_model(data) #this will be needed when we add qEI/constant liars
        output_points = []
        match data.strategy:
            case "random":
                for _ in range(data.points):
                    output_points.append([random.uniform(low, high) for low, high in data.bounds])

            case "hypercube":
                coordinates = []
                for low, high in data.bounds:
                    #for each dimension, generate a list of spaced coordinates and shuffle it:
                    step = (high - low)/data.points
                    coordinates.append([random.uniform(low + i*step, low + (i+1)*step) for i in range(data.points)])
                    random.shuffle(coordinates[-1])
                #add the points:
                for point in zip(*coordinates):
                    output_points.append(list(point))
            
        return output_points

    @intersect_message
    #trains a model then returns 2 lists: means and standard deviations
    def predictions(self, client_data: BOALaaSInputPredictions) -> list[list[float]]:
        data = ServersideInputPrediction(client_data)
        model = self._train_model(data)
        x_predict = self._create_n_dim_grid(data, data.points_per_dimension)
        means, stddevs = model.predict(x_predict, return_std=True)
        stddevs *= data.stddev #turn sigma into stddev of prediction
        #TODO: Undo preprocessing
        return [means.tolist(), stddevs.tolist()]

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
