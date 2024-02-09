import logging
from typing import Any
from time import sleep
import matplotlib.pyplot as plt
import numpy as np

from intersect_sdk import (
    IntersectClient,
    IntersectClientConfig,
    IntersectClientMessageParams,
    INTERSECT_JSON_VALUE,
    default_intersect_lifecycle_loop,
)
from data_class import ActiveLearningInputData

logging.basicConfig(level=logging.INFO)

def read_float(prompt: str):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid floating point input")

class ActiveLearningOrchestrator:
    def __init__(self, dataset_x: list[list[float]], dataset_y: list[float], bounds: list[list[float]]):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.bounds = bounds

    def assemble_message(self, operation:str = "next_point_by_EI") -> IntersectClientMessageParams:
        return IntersectClientMessageParams(
            destination='hello-organization.hello-facility.hello-system.hello-subsystem.hello-service',
            operation=operation,
            payload=ActiveLearningInputData(
                dataset_x=self.dataset_x, dataset_y=self.dataset_y,
                kernel="matern", length_per_dimension=self.SEPARATE_LENGTH_SCALES,
                bounds=self.bounds, y_is_good=self.Y_IS_GOOD
            ),
        )
    
    def graph(self):
        plt.clf()
        xx, yy = np.meshgrid(np.linspace(self.bounds[0][0], self.bounds[0][1], 101), np.linspace(self.bounds[1][0], self.bounds[1][1], 101))
        plt.contourf(xx, yy, self.mean_grid, levels=np.linspace(self.Y_LOW, self.Y_HIGH, 101), extend="both")
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(self.Y_LOW, self.Y_HIGH, 7))
        cbar.set_label(self.AXIS_LABEL)
        plt.xlabel(self.XLABEL)
        plt.ylabel(self.YLABEL)
        #add black dots for data points and a red marker for the recommendation:
        X_train = np.array(self.dataset_x)
        plt.scatter(X_train[:,0], X_train[:,1], color="black", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="red", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="none", edgecolors="red", marker="o", s=300)
        plt.savefig("graph.png")

    #the callback function:
    def __call__(
        self, source: str, operation: str, _has_error: bool, payload: INTERSECT_JSON_VALUE
    ) -> IntersectClientMessageParams:
        if len(self.dataset_x)==23:
            raise Exception
        if operation=="mean_grid":
            self.mean_grid = payload
            return self.assemble_message()
        self.x_EI = payload
        self.graph()
        self.add_data()
        return self.assemble_message("mean_grid")
    
    def add_data(self): #get another data point (x vector and y value), either from the user or by running a simulation
        raise NotImplementedError

class ActiveLearningExperiment(ActiveLearningOrchestrator):
    Y_LOW = 0
    Y_HIGH = 12
    Y_IS_GOOD = True #We're tring to maximize yield
    SEPARATE_LENGTH_SCALES = True
    AXIS_LABEL = "C2H4 Yield (%)"
    XLABEL = "Pulse Duration (ms)"
    YLABEL = "Peak Temperature (K)"

    def __init__(self):
        return super().__init__(
            [[20.0, 1200.0], [20.0, 1400.0], [20.0, 1600.0], [20.0, 1800.0], [20.0, 2000.0], [55.0, 1200.0], [55.0, 1400.0], [55.0, 1600.0], [55.0, 1800.0], [55.0, 2000.0], [70.0, 1600.0], [70.0, 1800.0], [70.0, 2000.0], [110.0, 1400.0], [110.0, 1600.0], [110.0, 1800.0], [110.0, 2000.0], [220.0, 1600.0], [220.0, 1800.0], [220.0, 2000.0], [330.0, 1200.0], [330.0, 1800.0], [330.0, 2000.0], [165.0, 1800.0], [165.0, 2000.0]],
            [0.4023, 0.8534, 2.9733, 4.7806, 5.5944, 0.191, 1.1508, 3.3015, 6.51, 9.8264, 5.28, 6.35, 9.68, 2.5194, 7.1148, 9.8176, 9.7903, 5.55, 7.04, 11.52, 1.3018, 7.15, 10.42, 6.77, 11.51],
            [[20,330], [1200,2000]],
        )

    def add_data(self):
        print(f'\nEI recommends running at: {self.x_EI[0]:.1f} ms, {self.x_EI[1]:.1f} K')
        print("\nEnter Experimental Data:")
        x0 = read_float("Duration (ms): ")
        x1 = read_float("Temperature (K): ")
        y  = read_float("Yield (%): ")
        self.dataset_x.append([x0, x1])
        self.dataset_y.append(y)

class ActiveLearningSimulation(ActiveLearningOrchestrator):
    Y_LOW = -1
    Y_HIGH = 5
    Y_IS_GOOD = False #we're trying to minimize error
    SEPARATE_LENGTH_SCALES = False
    AXIS_LABEL = "log(Simulation Result)"
    XLABEL = "Simulation Parameter #1"
    YLABEL = "Simulation Parameter #2"

    def log_rosenbrock(self, x0, x1): #Represents simulation error (vs experimental data) as a function of 2 simulation parameters
        r = (1-x0)**2 + 100*(x1-x0**2)**2
        return np.log10(max(.01, r)) #prevent errors from taking log of 0

    def __init__(self):
        x_data = [[0, 0], [-.1, 1.5], [.2, -1.4], [-1.7, -.3]]
        return super().__init__(x_data, [self.log_rosenbrock(*x) for x in x_data], [[-2, 2], [-2, 2]])

    def add_data(self):
        x0, x1 = self.x_EI
        print(f'Running simulation at ({x0:.2f},{x1:.2f}): ', end="", flush=True)
        sleep(1.5)
        y = self.log_rosenbrock(x0, x1)
        print(f'{10**y:.2f}')
        self.dataset_x.append([x0, x1])
        self.dataset_y.append(y)


if __name__ == '__main__':
    """
    step one: create configuration class, which handles user input validation - see the IntersectClientConfig class documentation for more info

    In most cases, everything under from_config_file should come from a configuration file, command line arguments, or environment variables.
    """
    from_config_file = {
        'data_stores': {
            'minio': [
                {
                    'host': '10.64.193.144',
                    'username': 'treefrog',
                    'password': 'XSD#n6!&Ro4&fjrK',
                    'port': 30020,
                },
            ],
        },
        'brokers': [
            {
                'host': '10.64.193.144',
               'username': 'postman',
                'password': 'ZTQ0YjljYTk0MTBj',
                'port': 30011,
                'protocol': 'mqtt3.1.1',
            },
        ],
    }
    config = IntersectClientConfig(
        **from_config_file,
    )

    """
    step two: construct the initial messages you want to send. In this case we will only send a single starting message.

    - The destination should match info.title in the service's schema. Another way to look at this is to see the service's
      HierarchyConfig, and then use
    - The operation should match one of the channels listed in the schema. In this case, 'say_hello_to_name' is the only
      operation exposed in the service.
    - The payload should represent what you want to send to the service's operation. You can determine the valid format
      from the service's schema. In this case, we're sending it a simple string. As long as the payload is a string,
      you'll get a message back.
    """

    #active_learning = ActiveLearningExperiment()
    active_learning = ActiveLearningSimulation()
    client = IntersectClient(
        config=config,
        initial_messages=[active_learning.assemble_message("mean_grid")],
        user_callback=active_learning,
    )

    """
    step four - start lifecycle loop. The only necessary parameter is your client.
    with certain applications (i.e. REST APIs) you'll want to integrate the client in the existing lifecycle,
    instead of using this one.
    In that case, just be sure to call client.startup() and client.shutdown() at appropriate stages.
    """
    default_intersect_lifecycle_loop(
        client,
    )

    """
    When running the loop, you should have 'Hello, hello_client!' printed to your console.
    Note that the client will automatically exit.
    """
