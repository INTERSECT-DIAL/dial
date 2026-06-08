import numpy as np
from scipy.optimize import root



PARAMETRIC_MODELS = {
    'ANDIE_second_order_I_vs_T': 'ANDIE_second_order_I_vs_T'
}


def ANDIE_second_order_I_vs_T(inputs, model_params):

    ### Example usage:
    ### Temperature = np.linspace(100,500,11)
    ### ANDIE_second_order_I_vs_T(Temperature, {'TN':250, 'M0':75, 'J':18, "BK":0.6})

    ### --- The following functions can be put into a model.py file --- ###
    ### --- Brillouin_J(), Wiess(), Mag_solve(), and second_order_I_vs_T() --- ###

    # Define the Brillouin Function
    def Brillouin_J(x, J):
        # x is a matrix
        # J is a row-vector

        f1 = (2*J+1)/(2*J) #factor 1
        f2 = 1/(2*J) #factor 2
        B_J = f1/(np.tanh(f1*x)) - f2/(np.tanh(f2*x))
        return B_J

    # Define the Wiess equation (re-arranged so that it always = 0)
    def Wiess(t, m, J):
        # t is a collum vector
        # J is a row vector
        # m is a matrix

        f1 = 3.0*J/(J +1.0) # factor 1
        f2 = 1.0/t #factor 2

        m = np.ones_like(t)*m
        x = f1*m*f2


        W = m - Brillouin_J(x, J)
        return W.reshape(-1)

    #Define a Magnetization Solver for givin t and J:
    def Mag_solve(t, J):
        # t is a collum vector
        # J is a row vector

        def fun(x):
            x= x.reshape((t.shape[0], J.shape[1]))
            return Wiess(t, x, J)

        X0 = np.ones((t.shape[0], J.shape[1]), dtype=np.float64) # matrix of every t,J combination

        solution = root(fun, X0, method ='hybr')

        m = np.array(solution.x).reshape((t.shape[0], J.shape[1]))
        return m

    #Define the Intensity as function of Temperature for Second order phase transitions
    def second_order_I_vs_T(Temperatures, TN=None, M0=None, J=None, BK=None):
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
        m_low = Mag_solve(t_low_cut, J) #returns a matrix where each collumn of m is associated with a collum of J

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

    return second_order_I_vs_T(inputs, **model_params)

