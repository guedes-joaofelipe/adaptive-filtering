#################################################################################
#                        Example: System Identification                         #
#################################################################################
#                                                                               #
#  In this example we have a typical system identification scenario. We want    #
# to estimate the filter coefficients of an unknown system given by Wo. In      #
# order to accomplish this task we use an adaptive filter with the same         #
# number of coefficients, N, as the unkown system. The procedure is:            #
# 1)  Excitate both filters (the unknown and the adaptive) with the signal      #
#   x. In this case, x is chosen according to the 4-QAM constellation.          #
#   The variance of x is normalized to 1.                                       #
# 2)  Generate the desired signal, d = Wo' x + n, which is the output of the    #
#   unknown system considering some disturbance (noise) in the model. The       #
#   noise power is given by sigma_n2.                                           #
# 3)  Choose an adaptive filtering algorithm to govern the rules of coefficient #
#   updating.                                                                   #
#                                                                               #
#     Adaptive Algorithm used here: Affine Projection                           #
#                                                                               #
#################################################################################

import sys, os
import numpy as np 
import matplotlib.pyplot as plt
sources_path = './Sources/'
if sources_path not in sys.path:
    sys.path.append(sources_path)

from adaptive_filtering.lms import AffineProjection
from adaptive_filtering.utils import rolling_window, generate_learning_plots


def main(output_filepath = None):

    ## Definitions
    j = complex(0,1)
    n_ensembles = 100   # number of realizations within the ensemble
    K = 500             # number of iterations (signal length)
    H = np.array([0.32+0.21j,-0.3+0.7j,0.5-0.8j,0.2+0.5j])    
    w_o = H             # Unknown system
    sigma_n2 = .04      # noise power
    N = 4               # Number of coefficients of the adaptive filter
    mu = .1            # Convergence factor (step) (0 < mu < 1)
    gamma = 1e-12       # small positive number to avoid singularity
    L = 1               # data reuse factor (L = 0 -> NLMS, L = 1 -> BNLMS, ...)

    ## Computing 

    # coefficient vector for each iteration and realization, w[0] = [1, 1, ..., 1]
    W = np.ones([n_ensembles, K+1, N], dtype=complex) 
    MSE = np.zeros([n_ensembles, K]) # MSE vector for each realization
    MSE_min = np.zeros([n_ensembles, K]) # Minimum MSE for each realization 

    for ensemble in np.arange(n_ensembles):    
        d = np.zeros([K], dtype=complex) # Desired signal
        
        # Creating the input signal (normalized)        
        x = (np.sign(np.random.randn(K)) + j*np.sign(np.random.randn(K)))/np.sqrt(2) # Complex signal
        n = np.sqrt(sigma_n2/2)*(np.random.normal(size=K)+j*np.random.normal(size=K)) # Complex noise
        sigma_x2 = np.var(x) # signal power = 1

        # Creating a tapped version of x with a N-sized window 
        prefixed_x = np.append(np.zeros([N-1]), x)
        X_tapped = rolling_window(prefixed_x, N)

        for k in np.arange(K):        
            d[k] = np.dot(np.conj(w_o), X_tapped[k])+n[k]       
        
        init_coef = W[ensemble][0]
        filter_order = N-1    
        
        lms = AffineProjection(step=mu, filter_order=filter_order, init_coef=init_coef, gamma=gamma, memory_length=L)
        lms.fit(d, x)     

        W[ensemble] = lms.coef_vector
        MSE[ensemble] = MSE[ensemble] + np.absolute(lms.error_vector)**2
        MSE_min[ensemble] = MSE_min[ensemble] + np.absolute(n)**2

    W_av = np.sum(W, axis=0)/n_ensembles
    MSE_av = sum(MSE, 2)/n_ensembles
    MSEmin_av = np.sum(MSE_min, axis=0)/n_ensembles

    # Generating plots    
    generate_learning_plots(K, N, MSE_av, MSEmin_av, W_av, w_o, output_filepath=output_filepath, algorithm='LMS')

if __name__ == "__main__":
    main(output_filepath='./Outputs/')