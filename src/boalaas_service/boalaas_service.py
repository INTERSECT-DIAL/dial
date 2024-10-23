import logging
import random
from typing import ClassVar

import gpax
import numpy as np
from boalaas_dataclass import (
    BOALaaSInputMultiple,
    BOALaaSInputPredictions,
    BOALaaSInputSingle,
)
from intersect_sdk import IntersectBaseCapabilityImplementation, intersect_message, intersect_status
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

from .serverside_data import (
    ServersideInputBase,
    ServersideInputMultiple,
    ServersideInputPrediction,
    ServersideInputSingle,
)

gpax.utils.enable_x64()

logger = logging.getLogger(__name__)

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".10"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


class BOALaaSCapabilityImplementation(IntersectBaseCapabilityImplementation):
    """Internal guts for GP usage."""

    intersect_sdk_capability_name = 'BOALaaS'

    _BACKENDS: ClassVar = ['sklearn', 'gpax']
    _KERNELS_SKLEARN: ClassVar = {'rbf': RBF, 'matern': Matern}
    _KERNELS_GPAX: ClassVar = {'rbf': 'RBF', 'matern': 'matern'}

    # trains a model based on the user's data
    def _train_model(
        self, data: ServersideInputBase
    ):  # -> GaussianProcessRegressor or -> gpax.ExactGP:
        if data.seed != -1:
            random.seed(data.seed)
            np.random.seed(data.seed)
        backend_name = data.backend.lower()
        if backend_name not in self._BACKENDS:
            msg = f'Unknown backend {backend_name}'
            raise ValueError(msg)

        if backend_name == 'gpax':
            if data.seed == -1:
                rng_key_train, rng_key_predict = gpax.utils.get_keys()
            else:
                rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
            # Initialize and train a variational inference GP model
            gp_model = gpax.viGP(len(data.bounds), self._kernel(data), guide='delta')
            gp_model.fit(
                rng_key_train,
                data.X_train,
                data.Y_train,
                num_steps=250,
                step_size=0.05,
                print_summary=False,
                progress_bar=False,
            )
            return gp_model

        model = GaussianProcessRegressor(kernel=self._kernel(data), n_restarts_optimizer=250)
        model.fit(data.X_train, data.Y_train)
        return model

    # parses the user's requested kernel
    def _kernel(self, data: ServersideInputBase):
        kernel_name = data.kernel.lower()
        backend_name = data.backend.lower()
        if backend_name not in self._BACKENDS:
            msg = f'Unknown backend {backend_name}'
            raise ValueError(msg)

        if backend_name == 'gpax':
            if kernel_name not in self._KERNELS_GPAX:
                msg = f'Unknown kernel {kernel_name}'
                raise ValueError(msg)
            return self._KERNELS_GPAX[kernel_name]

        if kernel_name not in self._KERNELS_SKLEARN:
            msg = f'Unknown kernel {kernel_name}'
            raise ValueError(msg)
        length_scale = [1.0] * len(data.X_train[0]) if data.length_per_dimension else 1.0
        return self._KERNELS_SKLEARN[kernel_name](length_scale=length_scale)

    # looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
    def _create_n_dim_grid(self, data: ServersideInputBase, points_per_dim: int | list[int]):
        if isinstance(points_per_dim, int):
            points_per_dim = [points_per_dim] * len(data.bounds)
        meshgrid = np.meshgrid(
            *(
                np.linspace(low, high, pts)
                for (low, high), pts in zip(data.bounds, points_per_dim, strict=False)
            ),
            indexing='ij',
        )
        return np.column_stack([arr.flatten() for arr in meshgrid])

    def _random_in_bounds(self, data: ServersideInputBase):
        return [random.uniform(low, high) for low, high in data.bounds]  # noqa: S311  (TODO - probably OK to use cryptographically insecure random numbers, but check this first)

    def _hypercube(self, data: ServersideInputBase, num_points) -> list[list[float]]:
        coordinates = []
        for low, high in data.bounds:
            # for each dimension, generate a list of spaced coordinates and shuffle it:
            step = (high - low) / num_points
            coordinates.append(
                [random.uniform(low + i * step, low + (i + 1) * step) for i in range(num_points)]  # noqa: S311  (TODO - probably OK to use cryptographically insecure random numbers, but check this first)
            )
            random.shuffle(coordinates[-1])
        # add the points:
        return [list(point) for point in zip(*coordinates, strict=False)]

    """
    Endpoints users can hit:
    """

    @intersect_message()
    # trains a model and then recommends a point to measure based on user's requested strategy:
    def get_next_point(self, client_data: BOALaaSInputSingle) -> list[float]:
        data = ServersideInputSingle(client_data)
        if data.seed != -1:
            random.seed(data.seed)
            np.random.seed(data.seed)
        
        # If it's random point, we don't need to train a model or anything else
        if data.strategy=="random":
            return self._random_in_bounds(data)

        model = self._train_model(data)

        # Generate the function that gives -1 * the "value" of measuring a point with the given mean & stddev
        negative_value = None
        match data.strategy:
            case 'uncertainty':

                def negative_value(_mean: float, stddev: float):
                    return -stddev
            case 'expected_improvement':

                def negative_value(mean: float, stddev: float):
                    if stddev == 0:
                        return 0

                    z = (mean - data.Y_best) / stddev * (1 if data.y_is_good else -1)
                    return -stddev * (z * norm.cdf(z) + norm.pdf(z))
            case 'confidence_bound':
                # calculate the z score that corresponds to the confidence bound:
                Z_VALUE = norm.ppf(
                    0.5 + data.confidence_bound / 2
                )  # need this because this is the "inverse CDF" which is one-tailed, but we want two-tailed

                def negative_value(mean: float, stddev: float):
                    #y_is_good: maximize mean + z*stddev, so minimize -mean - z*stdev
                    #y_is_bad:  minimize mean - z*stddev
                    return -Z_VALUE*stddev + mean*(-1 if data.y_is_good else 1)
                
        # Create a function that can be minimized over the bounds:
        def to_minimize(x):
            backend_name = data.backend.lower()
            if backend_name not in self._BACKENDS:
                raise ValueError(f'Unknown backend {backend_name}')
            if backend_name == "gpax":
                if data.seed == -1:
                    rng_key_train, rng_key_predict = gpax.utils.get_keys()
                else:
                    rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
                mean, y_var = model.predict(
                    rng_key_predict, x.reshape(1, -1)
                )  # output is y_pred, y_var
                mean, sigma = (
                    mean[0],
                    data.stddev * y_var[0],
                )  # it returns arrays, so fix that.  Also turn sigma into stddev of prediction
                return negative_value(mean, sigma)

        if data.discrete_measurements:
            self.measurement_grid = []
            row_values = np.linspace(data.bounds[0][0], data.bounds[0][1], data.discrete_measurement_grid_size[0])
            col_values = np.linspace(data.bounds[1][0], data.bounds[1][1], data.discrete_measurement_grid_size[1])
            for row in range(0,data.discrete_measurement_grid_size[0]):
                for col in range(0, data.discrete_measurement_grid_size[1]):
                    self.measurement_grid.append([row_values[row], col_values[col]])

            backend_name = data.backend.lower()
            if backend_name == "gpax":
                if data.seed == -1:
                    rng_key_train, rng_key_predict = gpax.utils.get_keys()
                else:
                    rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
                y_pred, y_var = model.predict(rng_key_predict, self.measurement_grid)
                stddevs = data.stddev*y_var #turn sigma into stddev of prediction
        
            else:
                means, stddevs = model.predict(self.measurement_grid, return_std=True)
                stddevs *= data.stddev #turn sigma into stddev of prediction

            response_surface = negative_value(means, stddevs)

            index = np.int64(np.argmin(response_surface))
            selected_point = self.measurement_grid[index]

            return selected_point
            
        else:
            guess = min(np.array(self._hypercube(data, data.optimization_points)), key=to_minimize)
            selected_point = minimize(to_minimize, guess, bounds=data.bounds, method="L-BFGS-B").x.tolist()
            print(selected_point)
            return selected_point
    
    @intersect_message
    def get_next_points(self, client_data: BOALaaSInputMultiple) -> list[list[float]]:
        data = ServersideInputMultiple(client_data)
        if data.seed != -1:
            random.seed(data.seed)
            np.random.seed(data.seed)
        # model = self._train_model(data) #this will be needed when we add qEI/constant liars
        output_points = None
        match data.strategy:
            case 'random':
                output_points = [self._random_in_bounds(data) for _ in range(data.points)]
            case 'hypercube':
                output_points = self._hypercube(data, data.points)
        return output_points

    """Trains a model then returns 3 lists based on user-supplied points:
        -Index 0: Predicted values.  These are inverse transformed (undoing the preprocessing to put them on the same scale as dataset_y)
        -Index 1: Inverse-transformed uncertainties.  If inverse-transforming is not possible (due to log-preprocessing), this will be all -1
        -Index 2: Uncertainties without inverse transformation
    """

    @intersect_message
    def get_surrogate_values(self, client_data: BOALaaSInputPredictions) -> list[list[float]]:
        data = ServersideInputPrediction(client_data)

        if data.backend == 'gpax':
            model = self._train_model(data)
            if data.seed == -1:
                rng_key_train, rng_key_predict = gpax.utils.get_keys()
            else:
                rng_key_train, rng_key_predict = gpax.utils.get_keys(seed=data.seed)
            y_pred, y_var = model.predict(rng_key_predict, data.x_predict)
            stddevs = data.stddev * y_var  # turn sigma into stddev of prediction
            # undo preprocessing:
            means = data.inverse_transform(y_pred)
            transformed_stddevs = data.inverse_transform(stddevs)
            return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]

        model = self._train_model(data)
        means, stddevs = model.predict(data.x_predict, return_std=True)
        stddevs *= data.stddev  # turn sigma into stddev of prediction
        # undo preprocessing:
        means = data.inverse_transform(means)
        transformed_stddevs = data.inverse_transform(stddevs)
        return [means.tolist(), transformed_stddevs.tolist(), stddevs.tolist()]

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'
