"""NOTE: This file should not be imported in application code except dynamically via the get_backend_module function in __init__.py ."""

import numpy as np
import scipy as sp
from scipy.optimize import root, OptimizeResult

from bumps.names import Curve, FitProblem, fit
from bumps.formatnum import format_uncertainty
from bumps.bounds import BoundedNormal


from . import AbstractBackend


class AndieBackend(
    AbstractBackend[OptimizeResult, None, tuple[np.ndarray, np.ndarray]]
):

### set correct model type in first argument

    ### ---          These are hyperparameters             --- ###
    ### --- Define the priors on the Isothermal parameters --- ###

    #Transition Temperature
    TN_guess = 60.0
    TN_std = 20
    TN_limits = (0.01, 200.0)

    #Background
    BK_guess = 70.0
    BK_std = 20.0
    BK_limits = (0.0, 150.0)

    #Second Order Scale
    M0_guess = 18.0
    M0_std = 3.0
    M0_limits = (0.01, 35.0)

    #Total Anglular Momentum
    J_guess = 0.6
    J_std = 0.5
    J_limits = (0.01, 14.0)


    #Define the prior distributions
    TN_dist =  BoundedNormal(TN_guess, TN_std, limits=TN_limits)
    M0_dist =  BoundedNormal(M0_guess, M0_std, limits=M0_limits)
    J_dist =  BoundedNormal(J_guess, J_std, limits=J_limits)
    BK_dist =  BoundedNormal(BK_guess, BK_std, limits=BK_limits)

    ### ---            End of hyperparameters             --- ###

    ### --- The following functions can be put into a model.py file --- ###
    ### --- Brillouin_J(), Wiess(), Mag_solve(), and second_order_I_vs_T() --- ###

    # Define the Brillouin Function
    @staticmethod
    def _Brillouin_J(x, J):
        # x is a matrix
        # J is a row-vector

        f1 = (2*J+1)/(2*J) #factor 1
        f2 = 1/(2*J) #factor 2
        B_J = f1/(np.tanh(f1*x)) - f2/(np.tanh(f2*x))
        return B_J

    # Define the Wiess equation (re-arranged so that it always = 0)
    @staticmethod
    def _Wiess(t, m, J):
        # t is a collum vector
        # J is a row vector
        # m is a matrix

        f1 = 3.0*J/(J +1.0) # factor 1
        f2 = 1.0/t #factor 2

        m = np.ones_like(t)*m
        x = f1*m*f2


        W = m - AndieBackend._Brillouin_J(x, J)
        return W.reshape(-1)

    #Define a Magnetization Solver for givin t and J:
    @staticmethod
    def _Mag_solve(t, J):
        # t is a collum vector
        # J is a row vector

        def fun(x):
            x= x.reshape((t.shape[0], J.shape[1]))
            return AndieBackend._Wiess(t, x, J)

        X0 = np.ones((t.shape[0], J.shape[1]), dtype=np.float64) # matrix of every t,J combination

        solution = root(fun, X0, method ='hybr')

        m = np.array(solution.x).reshape((t.shape[0], J.shape[1]))
        return m

    #Define the Intensity as function of Temperature for Second order phase transitions
    @staticmethod
    def _second_order_I_vs_T(Temperatures, TN, M0, J, BK):
        # TN, M0, J, BK are row vectors
        TN = np.array(TN).reshape(1,-1)
        M0 = np.array(M0).reshape(1,-1)
        J = np.array(J).reshape(1,-1)
        BK = np.array(BK).reshape(1,-1)
        # Temperatures is a collumn vector
        Temperatures = Temperatures.reshape(-1,1)

        Inverse_TN = 1/TN

        t = Temperatures @ Inverse_TN #Create a matrix of every (temperature, TN) combination

        #Find where all the temperatures below the TN.
        t_low_truth = t <1.0
        t_low_truth = t_low_truth.reshape(t.shape)
        t_low = t*t_low_truth

        #Because "Mag_solve" is expensive, only pass the non-zero rows of the t_low matrix.
        t_low_cut = t_low[t_low.sum(axis=1) !=0]

        #Calculate the reduced magnetization for below Tn\
        m_low = AndieBackend._Mag_solve(t_low_cut, J) #returns a matrix where each collumn of m is associated with a collum of J

        #The mag is zero above the TN, construct that matrix
        m_high = np.zeros((t.shape[0]-t_low_cut.shape[0], t.shape[1]))
        #Combine the m matrixies for below and above the TN
        m = np.concatenate((m_low,m_high), axis=0)

        #"t_low_cut" could have a raggid bottom edge, so only keep the m values below the TN
        m = m*t_low_truth

        #Calcualte the Magnetization (not reduced)
        M = m * M0 #row by row mulitplication of m by M0

        ##Calculate the intensities
        #Square of Magnetization
        I = M**2

        #Add the background
        I = I + BK

        return I

    @staticmethod
    def _Second_order_model(X_grid, Y_grid):
        M = Curve(AndieBackend._second_order_I_vs_T, X_grid, Y_grid,
            dy = np.sqrt(Y_grid/100),
            TN = AndieBackend.TN_guess,
            M0 = AndieBackend.M0_guess,
            J = AndieBackend.J_guess,
            BK = AndieBackend.BK_guess)

        M.TN.bounds = AndieBackend.TN_dist
        M.M0.bounds = AndieBackend.M0_dist
        M.J.bounds  = AndieBackend.J_dist
        M.BK.bounds = AndieBackend.BK_dist

        if len(X_grid) < 5 :
            problem = FitProblem(M, partial = True)
        else:
            problem = FitProblem(M, partial = False)

        return problem

    @staticmethod
    def train_model(data):
        print("train", "-"*20)

        # X_grid = [temperature for temperature in list(np.array(data.X_train)[0,:])]
        X_grid = np.array(data.X_train)
        Y_grid = np.array(data.Y_train)
        # X_grid = data.measuredTemperatures
        # Y_grid = data.measuredTemperatureDependentIntensities
        problem = AndieBackend._Second_order_model(X_grid, Y_grid)

        method = 'dream'

        print("initial chisq", problem.chisq_str())
        result = fit(problem, method=method, xtol=1e-6, ftol=1e-8,
                    samples = 100000,
                    burn = 100,
                    pop = 10,
                    init = 'eps',
                    thin = 1,
                    alpha = 0.1,
                    outliers = 'none',
                    trim = False,
                    steps = 0)
        print("final chisq", problem.chisq_str())
        for k, v, dv in zip(problem.labels(), result.x, result.dx):
            print(k, ":", format_uncertainty(v, dv))

        return result

    @staticmethod
    def predict(posterior_results, data):
        print("predict", "-"*20)

        #Extract the Posterior of the parameters
        draw = posterior_results.state.draw()

        #Posterior results are in alphabetical order
        Posterior_TN = draw.points[:,3]
        Posterior_M0 = draw.points[:,2]
        Posterior_J = draw.points[:,1]
        Posterior_BK = draw.points[:,0]

        # #Calcualte the mean of the parameters
        # TN_mean = np.mean(Posterior_TN).reshape(1,-1)
        # M0_mean = np.mean(Posterior_M0).reshape(1,-1)
        # J_mean = np.mean(Posterior_J).reshape(1,-1)
        # BK_mean = np.mean(Posterior_BK).reshape(1,-1)

        # #Calculate the curve from the mean of the parameters
        # Thermal_Mean_Posterior_parameter_curve = AndieBackend._second_order_I_vs_T(data.X_predict,
        #                                                             TN_mean, M0_mean, J_mean, BK_mean)



        #Calcualte all the predictive curves by interating through the posterior samples
        print('Predicting')
        # Thermal_CI_curves = np.empty((data.temperatureGrid.shape[0],0))
        Thermal_CI_curves = np.empty((np.array(data.x_predict).shape[0],0))
        thin_posterior_TN_dist = Posterior_TN[::50]
        thin_posterior_M0_dist = Posterior_M0[::50]
        thin_posterior_J_dist = Posterior_J[::50]
        thin_posterior_BK_dist = Posterior_BK[::50]
        for i, _ in enumerate(thin_posterior_TN_dist):
            I_interem = AndieBackend._second_order_I_vs_T(np.array(data.x_predict),
                                            TN=thin_posterior_TN_dist[i].reshape(1,-1),
                                            M0=thin_posterior_M0_dist[i].reshape(1,-1),
                                            J=thin_posterior_J_dist[i].reshape(1,-1),
                                            BK=thin_posterior_BK_dist[i].reshape(1,-1))
            Thermal_CI_curves = np.concatenate((Thermal_CI_curves, I_interem), axis=1)

        # q = np.array([0.95, 0.05])
        # I_ci = np.quantile(Thermal_CI_curves, q, axis=1)#.reshape((-1,2))
        # Thermal_lower_curve = I_ci[1,:].reshape(-1,1)
        # Thermal_upper_curve = I_ci[0,:].reshape(-1,1)

        Thermal_mean_of_posterior_curves = np.mean(Thermal_CI_curves, axis=1).reshape(-1,1)
        Thermal_variance = np.var(Thermal_CI_curves, axis=1).reshape(-1,1)
        # dictionary = {'Thermal Mean Posterior Parameter Curve': Thermal_Mean_Posterior_parameter_curve,
        #             'Thermal Mean of Posterior Curves': Thermal_mean_of_posterior_curves,
        #             'Thermal Variance': Thermal_variance,
        #             'Thermal CI Curves': Thermal_CI_curves,
        #             'Thermal lower Curve': Thermal_lower_curve,
        #             'Thermal upper Curve': Thermal_upper_curve}
        return Thermal_mean_of_posterior_curves.flatten(), Thermal_variance.flatten()

    @staticmethod
    def sample(module, model, data):
        print("sample", "-"*20)
        off_flag = False

        # Thermal_next_sample = np.max(data.measuredTemperatureIdx) + 1
        # Thermal_next_sample = np.max(data.X_train[-1][1]) + 1
        Thermal_next_sample = int(data.extra_args['last_idx']) + 1
        

        above_TN = data.extra_args['above_TN']
        # above_TN = data.above_TN
        T_start = data.bounds[0][0]
        T_stop = data.bounds[0][1]

        T_step = data.extra_args['T_step']
        T2 = np.linspace(T_start, T_stop, int((T_stop-T_start)/T_step)+1).reshape(-1,1)

        TN_best = model.state.best()[0][3]
        TN_std = model.dx[3]
        TN_upper = TN_best + TN_std

        data.x_predict = T2
        Thermal_mean_of_posterior_curves, Thermal_variance = module.predict(model, data)

        # def find_next_temperature(T2, T_step, above_TN, Thermal_next_sample, Thermal_posterior_results, Thermal_predictions):

        if Thermal_next_sample >= T2.shape[0]:
            print('You have reached your destination.')
            off_flag = True


        if (off_flag == False):
            next_sample_acquired = False
            while next_sample_acquired == False:
                print("**"*20, Thermal_next_sample)
                uncertainty = Thermal_variance[Thermal_next_sample]
                Bravery_factor = 8.5
                if T2[Thermal_next_sample] > TN_upper:
                    #If the next temp is bigger than the upper confidence bound TN, take a big step
                    #Or if the isothermal inference did not find a peak, take a big step
                    above_TN += 1
                    print('We are in the end game now' )
                    step = np.round(5*above_TN**2.5) #Degrees to increase the temperature.
                    if Thermal_next_sample + int(step/T_step) < T2.shape[0]:
                        #If the index after the step would still be within the search space, choose that temp.
                        Thermal_next_sample = Thermal_next_sample + int(step/T_step)

                    else:
                        print('You have reached your destination.')
                        off_flag = True

                    next_sample_acquired = True
                #Optional further condition to take more data points within confidence interval of the predicted TN
            #       elif T2[Thermal_next_sample] >= TN_lower:
            #         #If the next temp is bigger than the lower confidence bound of TN, take a small step
            #         x_step = 0.5 #Degrees to increase the temperature
            #         Thermal_next_sample = Thermal_next_sample + int(x_step/T_step)
            #         next_sample_acquired = True
            #         print('Theres nothing for it, you got to take the next step' )
                elif uncertainty < Bravery_factor*Thermal_mean_of_posterior_curves[Thermal_next_sample]/100:  #With the instrument error scaling factor
                    Thermal_next_sample = Thermal_next_sample+1
                else:
                    print('Acquired Next Temp on Main path')
                    next_sample_acquired = True


        if off_flag:
            Thermal_next_sample = T2.shape[0] - 1



        ## these updates may not be necessary (they are on the server side)
        ## the updates on the client side are from the outputs of this function
        ## the client updates are handled in handle_next_points() in the clinet file
        data.extra_args['above_TN'] = above_TN
        data.extra_args['last_idx'] = Thermal_next_sample

        return [T2[Thermal_next_sample], above_TN, Thermal_next_sample]



