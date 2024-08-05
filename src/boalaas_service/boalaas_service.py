import logging
import random
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import gpax
gpax.utils.enable_x64()
import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
print('gpax was imported succesfully')
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

    _BACKENDS = ["sklearn", "gpax"]
    #trains a model based on the user's data
    def _train_model(self, data: ServersideInputBase): # -> GaussianProcessRegressor or -> gpax.ExactGP:
        backend_name = data.backend.lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(f'Unknown backend {backend_name}')
        if backend_name == "gpax":
            model = gpax.ExactGP(input_dim=2, kernel=self._kernel(data))
            key1, key2 = gpax.utils.get_keys()
            model.fit(key1, data.X_train, data.Y_train)
            print('Completed GP model training')
            return model
        else:
            model = GaussianProcessRegressor(kernel=self._kernel(data), n_restarts_optimizer=250)
            model.fit(data.X_train, data.Y_train)
            return model

    _KERNELS_SKLEARN = {"rbf": RBF, "matern": Matern}
    _KERNELS_GPAX = {"rbf": "RBF", "matern": "matern"}

    #parses the user's requested kernel
    def _kernel(self, data: ServersideInputBase):
        kernel_name = data.kernel.lower()
        backend_name = data.backend.lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(f'Unknown backend {backend_name}')
        
        if backend_name == "gpax":
            if kernel_name not in self._KERNELS_GPAX:
                raise ValueError(f'Unknown kernel {kernel_name}')
            return self._KERNELS_GPAX[kernel_name]
        
        else:
            if kernel_name not in self._KERNELS_SKLEARN:
                raise ValueError(f'Unknown kernel {kernel_name}')
            length_scale = [1.0]*len(data.X_train[0]) if data.length_per_dimension else 1.0
            return self._KERNELS_SKLEARN[kernel_name](length_scale=length_scale)
    
    #looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
    def _create_n_dim_grid(self, data: ServersideInputBase, points_per_dim: Union[int,list[int]]):
        if isinstance(points_per_dim, int):
            points_per_dim = [points_per_dim]*len(data.bounds)
        meshgrid = np.meshgrid(*(np.linspace(low, high, pts)
                                 for (low, high), pts in zip(data.bounds, points_per_dim)), indexing='ij')
        return np.column_stack([arr.flatten() for arr in meshgrid])

    def _random_in_bounds(self, data: ServersideInputBase):
        return [random.uniform(low, high) for low, high in data.bounds]
    
    def _hypercube(self, data: ServersideInputBase, num_points) -> list[list[float]]:
        coordinates = []
        for low, high in data.bounds:
            #for each dimension, generate a list of spaced coordinates and shuffle it:
            step = (high - low)/num_points
            coordinates.append([random.uniform(low + i*step, low + (i+1)*step) for i in range(num_points)])
            random.shuffle(coordinates[-1])
        #add the points:
        return [list(point) for point in zip(*coordinates)]

    '''
    Endpoints users can hit:
    '''
    @intersect_message()
    # trains a model and then recommends a point to measure based on user's requested strategy:
    def get_next_point(self, client_data: BOALaaSInputSingle) -> list[float]:
        data = ServersideInputSingle(client_data)
        #if it's random point, we don't need to train a model or anything:
        if data.strategy=="random":
            return self._random_in_bounds(data)
        
        model = self._train_model(data)
        # generate the function that gives -1 * the "value" of measuring a point with the given mean & stddev
        negative_value = None
        match data.strategy:
            case "uncertainty":
                def negative_value(_mean: float, stddev: float):
                    return -stddev
            case "expected_improvement":
                def negative_value(mean: float, stddev: float):
                    if stddev == 0:
                        return 0
                    else:
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
            key1, key2 = gpax.utils.get_keys()
            mean, f_samples = model.predict_in_batches(key2, x.reshape(1, -1), batch_size=1, noiseless=True)
            sigma = f_samples.std(axis=(0,1))
            mean, sigma = mean[0], data.stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            return negative_value(mean, sigma)


            # print('To minimize?')
            # backend_name = data.backend.lower()
            # if backend_name not in self._BACKENDS:
            #     raise ValueError(f'Unknown backend {backend_name}')
            
            # if backend_name == "gpax":
            #     print('Test 6')
            #     key1, key2 = gpax.utils.get_keys()
            #     print('Test 6.5')
            #     print(key2)
            #     print(x)
            #     print(model)
            #     mean, f_samples = model.predict_in_batches(key2, x.reshape(1, -1), batch_size=1, noiseless=True)
            #     print('Test 7')
            #     sigma = f_samples.std(axis=(0,1))
            #     print('Test 8')
            #     mean, sigma = mean[0], data.stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            #     return negative_value(mean, sigma)

            # else:
            #     mean, sigma = model.predict(x.reshape(1, -1), return_std=True)
            #     mean, sigma = mean[0], data.stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            #     return negative_value(mean, sigma)
            
        guess = min(np.array(self._hypercube(data, data.optimization_points)), key=to_minimize)
        return minimize(to_minimize, guess, bounds=data.bounds, method="L-BFGS-B").x.tolist()
    
    @intersect_message
    def get_next_points(self, client_data: BOALaaSInputMultiple) -> list[list[float]]:
        data = ServersideInputMultiple(client_data)
        #model = self._train_model(data) #this will be needed when we add qEI/constant liars
        output_points = None
        match data.strategy:
            case "random":
                output_points = [self._random_in_bounds(data) for _ in range(data.points)]
            case "hypercube":
                output_points = self._hypercube(data, data.points)
        return output_points

    '''Trains a model then returns 3 lists based on user-supplied points:
        -Index 0: Predicted values.  These are inverse transformed (undoing the preprocessing to put them on the same scale as dataset_y)
        -Index 1: Inverse-transformed uncertainties.  If inverse-transforming is not possible (due to log-preprocessing), this will be all -1
        -Index 2: Uncertainties without inverse transformation
    '''
    @intersect_message
    def get_surrogate_values(self, client_data: BOALaaSInputPredictions) -> list[list[float]]:
        data = ServersideInputPrediction(client_data)

        backend_name = data.backend.lower()
        if backend_name not in self._BACKENDS:
            raise ValueError(f'Unknown backend {backend_name}')
        
        if backend_name == "gpax":
            model = self._train_model(data)
            key1, key2 = gpax.utils.get_keys()
            means, f_samples = model.predict_in_batches(key2, data.x_predict, batch_size=100, noiseless=True)
            stddevs = f_samples.std(axis=(0,1))
            stddevs *= data.stddev #turn sigma into stddev of prediction
            #undo preprocessing:
            means = data.inverse_transform(means)
            transformed_stddevs = data.inverse_transform(stddevs)
            return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]
        
        else:
            model = self._train_model(data)
            means, stddevs = model.predict(data.x_predict, return_std=True)
            stddevs *= data.stddev #turn sigma into stddev of prediction
            #undo preprocessing:
            means = data.inverse_transform(means)
            transformed_stddevs = data.inverse_transform(stddevs)
            return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
