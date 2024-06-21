import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import numpy as np

from intersect_sdk import (
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectClientMessageParams,
    INTERSECT_JSON_VALUE,
    default_intersect_lifecycle_loop,
)

from boalaas_dataclass import BOALaaSInputSingle, BOALaaSInputPredictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
Here, we attempt to find the minimum of the Rosenbrock function, which occurs at (1,1).  We are pretending that this is the result of running some computational simulation.
We start with 10 data points, which form a Latin Hypercube over the bounded space.  We then call the service, which calculates the point with the maximial EI (Expected Improvement).  We then evaluate the rosenbrock at this point, giving us our 11th data point.
This process repeats until we sample 15 more points (representing a limited budget of computational time/runs).

Overall, this represents an automated workflow: The experiment/simulation/whatever that produces results is connected to INTERSECT.  Points can be sampled automatically without human intervention.

Note that the service must be started first, then the client.
'''

def rosenbrock(x0, x1): #Represents simulation error (vs experimental data) as a function of 2 simulation parameters
    return (1-x0)**2 + 100*(x1-x0**2)**2

class ActiveLearningOrchestrator:
    def __init__(self):
        self.dataset_x = [[0.9317758694133622, -0.23597335497782845], [-0.7569874398003542, -0.76891211613756], [-0.38457336507729645, -1.1327391183311766], [-0.9293590899359039, 0.25039725076881014], [1.984696498789749, -1.7147926093003538], [1.2001856430453541, 1.572387611848939], [0.5080666898409634, -1.566722183270571], [-1.871124738716507, 1.9022651997285078], [-1.572941300813352, 1.0014173171150125], [0.033053333077524005, 0.44682040004191537]]
        self.dataset_y = [rosenbrock(*x) for x in self.dataset_x]
        self.bounds = [[-2,2], [-2,2]]
        #generate a 101x101 grid for predictions and graphing:
        self.xx, self.yy = np.meshgrid(np.linspace(self.bounds[0][0], self.bounds[0][1], 101), np.linspace(self.bounds[1][0], self.bounds[1][1], 101), indexing="ij")
        self.points_to_predict = np.hstack([self.xx.reshape(-1, 1), self.yy.reshape(-1, 1)])

    #create a message to send to the server
    def assemble_message(self, operation:str) -> IntersectClientCallback:
        payload = None
        if operation=="get_next_point":
            payload = BOALaaSInputSingle(
                strategy="expected_improvement",
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel="rbf",
                length_per_dimension=True, #allow the matern to use separate length scales for the two parameters
                y_is_good=False            #we wish to minimize y (the error)
            )
        else:
            payload = BOALaaSInputPredictions(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                points_to_predict=self.points_to_predict,
                kernel="rbf",
                length_per_dimension=True,
                y_is_good=False
            )
        return IntersectClientCallback(
            messages_to_send=[
                IntersectClientMessageParams(
                    destination='neeter-active-learning-organization.neeter-active-learning-facility.neeter-active-learning-system.neeter-active-learning-subsystem.neeter-active-learning-service',
                    operation=operation,
                    payload=payload,
                )
            ]
        )
    
    #The callback function.  This is called whenever the server responds to our message.
    #This could instead be implemented by defining a callback method (and passing it later), but here we chose to directly make the object callable.
    def __call__(
        self, source: str, operation: str, _has_error: bool, payload: INTERSECT_JSON_VALUE
    ) -> IntersectClientCallback:
        if operation=="get_surrogate_values": #if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
            self.mean_grid = np.array(payload[0]).reshape((101,101))
            return self.assemble_message("get_next_point") #returning a message automatically sends it to the server
        #if we receive an EI recommendation, record it, show the user the current graph, and run the "simulation":
        self.x_EI = payload
        self.graph()
        if len(self.dataset_x)==25:
            raise Exception
        self.add_data()
        return self.assemble_message("get_surrogate_values")
    
    def graph(self):
        plt.clf()
        data = np.maximum(np.array(self.mean_grid), .11) #the predicted means can be <0 which causes white patches in the graph; this fixes that
        plt.contourf(self.xx, self.yy, data, levels=np.logspace(-2, 4, 101), norm="log", extend="both")
        cbar = plt.colorbar()
        cbar.set_ticks(np.logspace(-2, 4, 7))
        cbar.set_label("Simulation Result")
        plt.xlabel("Simulation Parameter #1")
        plt.ylabel("Simulation Parameter #2")
        #add black dots for data points and a red marker for the recommendation:
        X_train = np.array(self.dataset_x)
        plt.scatter(X_train[:,0], X_train[:,1], color="black", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="red", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="none", edgecolors="red", marker="o", s=300)
        plt.savefig("graph.png")
    
    #calculates the rosenbrock at a certain spot and adds it to our dataset
    def add_data(self):
        x0, x1 = self.x_EI
        print(f'Running simulation at ({x0:.2f},{x1:.2f}): ', end="", flush=True)
        y = rosenbrock(x0, x1)
        print(f'{y:.3f}')
        self.dataset_x.append([x0, x1])
        self.dataset_y.append(y)

if __name__ == '__main__':
    #Create configuration class, which handles user input validation - see the IntersectClientConfig class documentation for more info
    #In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='Automated client')
    parser.add_argument(
        '--config',
        type=Path,
        default=os.environ.get('NEETER_CONFIG_FILE', Path(__file__).parents[1] / 'local-conf.json'),
    )
    args = parser.parse_args()
    try:
        with open(args.config, 'rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)
    
    #Create our orchestrator
    active_learning = ActiveLearningOrchestrator()
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message("get_surrogate_values"), #the initial message to send
        **from_config_file,
    )
    
    #use the orchestator to create the client
    client = IntersectClient(
        config=config,
        user_callback=active_learning, #the callback (here we use a callable object, as discussed above)
    )

    #This will run the send message -> wait for response -> callback -> repeat cycle until we have 25 points (and then raise an Exception)
    default_intersect_lifecycle_loop(
        client,
    )