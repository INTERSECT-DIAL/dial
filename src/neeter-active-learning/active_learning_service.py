import logging
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from scipy.optimize import minimize, basinhopping
from scipy.stats import norm
from typing import Callable

from intersect_sdk import (
    HierarchyConfig,
    IntersectService,
    IntersectServiceConfig,
    default_intersect_lifecycle_loop,
    intersect_message,
    intersect_status,
)

from data_class import ActiveLearningInputData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearningServiceCapabilityImplementation:

    '''
    Internal guts for GP usage:
    '''
    #trains a model based on the user's data
    def _train_model(self, data: ActiveLearningInputData) -> GaussianProcessRegressor:
        X_train, Y_train = np.array(data.dataset_x), np.array(data.dataset_y)
        model = GaussianProcessRegressor(kernel=self._kernel(data), n_restarts_optimizer=100)
        model.fit(X_train, Y_train)
        return model

    _KERNELS = {"rbf": RBF, "matern": Matern}
    #parses the user's requested kernel
    def _kernel(self, data: ActiveLearningInputData):
        kernel_name = data.kernel.lower()
        if kernel_name not in self._KERNELS:
            raise ValueError(f'Unknown kernel {kernel_name}')
        length_scale = [1.0]*len(data.dataset_x[0]) if data.length_per_dimension else 1.0
        return self._KERNELS[kernel_name](length_scale=length_scale)
    
    #looks at the data bounds to create a regular mesh with (points_per_dim)^N points (where N is the number of dimensions)
    def _create_n_dim_grid(self, data: ActiveLearningInputData, points_per_dim):
        meshgrid = np.meshgrid(*(np.linspace(low, high, points_per_dim) for low, high in data.bounds), indexing='ij')
        return np.column_stack([arr.flatten() for arr in meshgrid])
        
    def _minimize(self, data: ActiveLearningInputData, f: Callable[[np.ndarray], float]) -> np.ndarray:
        guess = min(self._create_n_dim_grid(data, 11), key=f)
        return minimize(f, guess, bounds=data.bounds, method="L-BFGS-B").x

    '''
    Endpoints users can hit:
    '''
    @intersect_message()
    #returns the point with the highest standard deviation
    def next_point_by_uncertainty(self, data: ActiveLearningInputData) -> list[float]:
        model = self._train_model(data)
        neg_stddev = lambda x: -1*model.predict(x.reshape(1, -1), return_std=True)[1][0]
        return self._minimize(data, neg_stddev).tolist()

    @intersect_message()
    #returns the point with the highest Expected Improvement
    def next_point_by_EI(self, data: ActiveLearningInputData) -> list[float]:
        print(f"Running EI on {len(data.dataset_x)} datapoints")

        model = self._train_model(data)
        Y_best = max(data.dataset_y) if data.y_is_good else min(data.dataset_y)
        Y_stddev = np.std(data.dataset_y)
        def EI(x): #if y is good, this is actually -1*EI, since we have a minimizer
            mean, sigma = model.predict(x.reshape(1, -1), return_std=True)
            mean, sigma = mean[0], Y_stddev*sigma[0] #it returns arrays, so fix that.  Also turn sigma into stddev of prediction
            if data.y_is_good:
                z = (mean - Y_best)/sigma
            else:
                z = (Y_best - mean)/sigma
            return -sigma*(z * norm.cdf(z) + norm.pdf(z))
        return self._minimize(data, EI).tolist()
    
    @intersect_message()
    #trains a model and then returns a grid of the predicted means (for graphing/display purposes)
    def mean_grid(self, data: ActiveLearningInputData) -> list[list[float]]:
        model = self._train_model(data)
        xx, yy = np.meshgrid(np.linspace(data.bounds[0][0], data.bounds[0][1], 101), np.linspace(data.bounds[1][0], data.bounds[1][1], 101))
        X_data = np.c_[xx.ravel(), yy.ravel()]
        mean_data = model.predict(X_data).reshape(xx.shape).tolist()
        return mean_data

    @intersect_status()
    def status(self) -> str:
        """Basic status function which returns a hard-coded string."""
        return 'Up'


if __name__ == '__main__':
    """
    step one: create configuration class, which handles validation - see the IntersectServiceConfig class documentation for more info

    In most cases, everything under from_config_file should come from a configuration file, command line arguments, or environment variables.
    """
    from_config_file = {
        'data_stores': {
            'minio': [
                {
                    'host': '',
                    'username': '',
                    'password': '',
                    'port': 0,
                },
            ],
        },
        'brokers': [
            {
                'host': '',
                'username': '',
                'password': '',
                'port': 0,
                'protocol': '',
            },
        ],
    }
    config = IntersectServiceConfig(
        hierarchy=HierarchyConfig(
            organization='hello-organization',
            facility='hello-facility',
            system='hello-system',
            subsystem='hello-subsystem',
            service='hello-service',
        ),
        schema_version='0.0.1',
        **from_config_file,
    )

    """
    step two - create your own capability implementation class.

    You have complete control over how you construct this class, as long as it has decorated functions with
    @intersect_message and @intersect_status, and that these functions are appropriately type-annotated.
    """
    capability = ActiveLearningServiceCapabilityImplementation()

    """
    step three - create service from both the configuration and your own capability
    """
    service = IntersectService(capability, config)

    """
    step four - start lifecycle loop. The only necessary parameter is your service.
    with certain applications (i.e. REST APIs) you'll want to integrate the service in the existing lifecycle,
    instead of using this one.
    In that case, just be sure to call service.startup() and service.shutdown() at appropriate stages.
    """
    logger.info('Starting hello_service, use Ctrl+C to exit.')
    default_intersect_lifecycle_loop(
        service,
    )

    """
    Note that the service will run forever until you explicitly kill the application (i.e. Ctrl+C)
    """
