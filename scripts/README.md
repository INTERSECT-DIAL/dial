# Information About the Automated Client:
In the `automated_client.py` script, we search for the minimum of the Rosenbrock function, which occurs at (1,1).  We are pretending that this is the result of running some computational simulation.
We start with 10 data points, which form a Latin Hypercube over the bounded space.  We then call the service, which calculates the point with the maximial EI (Expected Improvement).  We then evaluate the rosenbrock at this point, giving us our 11th data point.

This process repeats until we sample 15 more points (representing a limited budget of computational time/runs).

Overall, this represents an automated workflow: The experiment/simulation/whatever that produces results is connected to INTERSECT.  Points can be sampled automatically without human intervention.

# Generalizing to Higher Dimensions
The `automated_client` written here optimizes the 2D Rosenbrock function by default. To optimize functions of higher dimensions, the following changes must be made manually:
1. Update the `rosenbrock` function to accept additional variable(s). For example, the `rosenbrock` function in 3D is `100*(x1-x0**2)**2 + (1-x0)**2 + 100*(x2-x1**2)**2 + (1-x1)**2` with an optimum at (1,1,1).
2. Append a list of form `[var_lower_bound, var_upper_bound]` to `self.bounds` corresponding to each additional variable.
3. Uncomment and use the LHS code in the class initialization (instead of the explicitly written `self.dataset_x`) to generate initial samples in the desired dimensions that are scaled according to corresponding variable bounds.
4. (Optional) Update the `graph(self)` function to represent the surrogate and tested points in more than 2 dimensions. If the dimension is not equal to 2, the graph will save an image with the message "Number of dimensions is not equal to two - Bayesian Optimization plot is not available. Add plotting to the graph(self) function in automated_client.py to generate a custom plot."

# Notes on Runnning `automated_client.py`
The service must be started first in a dedicated terminal by running the `launch_service.py` script. This should print the following: 
`INFO:intersect-sdk:Service is starting up`
`INFO:intersect-sdk:Service startup complete`
Then, the client can be run in a separate terminal by running the `automated_client.py` script.