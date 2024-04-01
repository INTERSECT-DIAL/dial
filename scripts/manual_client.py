import logging
import matplotlib.pyplot as plt
import numpy as np

from intersect_sdk import (
    IntersectClient,
    IntersectClientConfig,
    IntersectClientMessageParams,
    INTERSECT_JSON_VALUE,
    default_intersect_lifecycle_loop,
)

from neeter_active_learning.data_class import ActiveLearningInputData

logging.basicConfig(level=logging.INFO)

'''
This is a replication of the workflow in the Maryland nature paper.  In this workflow, we have two tuneable "dials" on a chemical process: Duration (20-300 ms) and Temperature (1200-2000 K).  We seek to maximize the yield (amount of desired chemical produced).
We start with 25 data points, then call the service, which calculates the point with the maximial EI (Expected Improvement).  This point is recommended to the user, who would then run a experiment, giving the 26th data point.
This process repeats until killed.

Overall, this represents a manual workflow: The user has the client open in a terminal, and runs the experiment/simulation/whatever is producing results, which is not connected to INTERSECT.

Note that the service must be started first, then the client.
'''

def read_float(prompt: str): #read a floating point value from the user, keep asking until they give a valid one
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid floating point input")

class ActiveLearningOrchestrator:
    def __init__(self):
        self.dataset_x = [[20., 1200.], [20., 1400.], [20., 1600.], [20., 1800.], [20., 2000.], [55., 1200.], [55., 1400.], [55., 1600.], [55., 1800.], [55., 2000.], [70., 1600.], [70., 1800.], [70., 2000.], [110., 1400.], [110., 1600.], [110., 1800.], [110., 2000.], [220., 1600.], [220., 1800.], [220., 2000.], [330., 1200.], [330., 1800.], [330., 2000.], [165., 1800.], [165., 2000.]]
        self.dataset_y = [0.4023, 0.8534, 2.9733, 4.7806, 5.5944, 0.191, 1.1508, 3.3015, 6.51, 9.8264, 5.28, 6.35, 9.68, 2.5194, 7.1148, 9.8176, 9.7903, 5.55, 7.04, 11.52, 1.3018, 7.15, 10.42, 6.77, 11.51]
        self.bounds = [[20,330], [1200,2000]]

    #create a message to send to the server
    def assemble_message(self, operation:str) -> IntersectClientMessageParams:
        return IntersectClientMessageParams(
            destination='hello-organization.hello-facility.hello-system.hello-subsystem.hello-service',
            operation=operation,
            payload=ActiveLearningInputData(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=self.bounds,
                kernel="matern",
                length_per_dimension=True, #allow the matern to use separate length scales for temp and duration
                y_is_good=True             #we wish to maximize y (the yield)
            ),
        )
    
    #The callback function.  This is called whenever the server responds to our message.
    #This could instead be implemented by defining a callback method (and passing it later), but here we chose to directly make the object callable.
    def __call__(
        self, source: str, operation: str, _has_error: bool, payload: INTERSECT_JSON_VALUE
    ) -> IntersectClientMessageParams:
        if operation=="mean_grid": #if we receive a grid of predictions, record it for graphing, then ask for the next recommended point
            self.mean_grid = payload
            return self.assemble_message("next_point_by_EI") #returning a message automatically sends it to the server
        #if we receive an EI recommendation, record it, show the user the current graph, and ask the user for the results of their experiment:
        self.x_EI = payload
        self.graph()
        self.add_data()
        return self.assemble_message("mean_grid")

    #makes a color graph of the predicted yields, with markers for the training data and EI-recommended point:
    def graph(self):
        plt.clf()
        xx, yy = np.meshgrid(np.linspace(self.bounds[0][0], self.bounds[0][1], 101), np.linspace(self.bounds[1][0], self.bounds[1][1], 101))
        plt.contourf(xx, yy, self.mean_grid, levels=np.linspace(0, 12, 101), extend="both")
        cbar = plt.colorbar()
        cbar.set_ticks(np.linspace(0, 12, 7))
        cbar.set_label("C2H4 Yield (%)")
        plt.xlabel("Pulse Duration (ms)")
        plt.ylabel("Peak Temperature (K)")
        #add black dots for data points and a red marker for the recommendation:
        X_train = np.array(self.dataset_x)
        plt.scatter(X_train[:,0], X_train[:,1], color="black", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="red", marker="o")
        plt.scatter([self.x_EI[0]], [self.x_EI[1]], color="none", edgecolors="red", marker="o", s=300)
        plt.savefig("graph.png")

    #asks the user for a data point (an experimental result) and adds it to our dataset
    def add_data(self):
        print(f'\nEI recommends running at: {self.x_EI[0]:.1f} ms, {self.x_EI[1]:.1f} K')
        print("\nEnter Experimental Data:")
        x0 = read_float("Duration (ms): ")
        x1 = read_float("Temperature (K): ")
        y  = read_float("Yield (%): ")
        self.dataset_x.append([x0, x1])
        self.dataset_y.append(y)


if __name__ == '__main__':
    #Create configuration class, which handles user input validation - see the IntersectClientConfig class documentation for more info
    #In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
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
    config = IntersectClientConfig(
        **from_config_file,
    )

    #Create our orchestrator
    active_learning = ActiveLearningOrchestrator()
    #use the orchestator to create the client
    client = IntersectClient(
        config=config,
        initial_messages=[active_learning.assemble_message("mean_grid")], #the initial message to send
        user_callback=active_learning, #the callback (here we use a callable object, as discussed above)
    )

    #This will run the send message -> wait for response -> callback -> repeat cycle until the user raises an Exception with Ctrl+C
    default_intersect_lifecycle_loop(
        client,
    )
