import os
import numpy as np 
import matplotlib.pyplot as plt

def rolling_window(x, window):
    """ Creates a N-sized rolling window for vector x
    
    Arguments:
        x {np.array} -- input vector
        window {int} -- window size
    
    Returns:
        [np.array] -- array of x window's
    """
    shape = x.shape[:-1] + (x.shape[-1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def generate_learning_plots(K, N, MSE_av, MSEmin_av, W_av, w_o, output_filepath=None, algorithm='LMS'):
    """ Generates Learning Curve and Evolution of Coef plots
    
    Arguments:
        K {int} -- number of iterations
        N {int} -- number of coefficients
        MSE_av {np.array} -- average MSE for all ensembles
        MSEmin_av {np.array} -- average minimum MSE for all ensembles
        W_av {np.array} -- average coefficients for all ensembles
        w_o {np.array} -- optimal values of coefficients
    
    Keyword Arguments:
        output_filepath {str} -- folder to save plots (default: {None})
        algorithm {str} -- algorithm's name (default: {'LMS'})
    """
    # Generating Learning Curve plots 
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(16,8), sharex=True)
    ax[0].plot(np.arange(K), 10*np.log10(MSE_av))
    ax[0].set_title('Learning Curve for MSE')
    ax[0].set_ylabel('MSE [dB]')
    ax[0].grid(True)

    ax[1].plot(np.arange(K), 10*np.log10(MSEmin_av))
    ax[1].set_title('Learning Curve for MSEmin')
    ax[1].set_ylabel('MSEmin [dB]')
    ax[1].set_xlabel('Number of iterations, k') 
    ax[1].grid(True)

    if output_filepath is not None:
        # Creating plots output folder if they don't exist
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        fig.savefig(output_filepath + algorithm + '_learning_curve.jpg', bbox_inches = 'tight')

    # Generating Evolution of Coefficients plots
    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(16,8), sharex=True)
    for n in np.arange(N):
        ax[0].plot(np.arange(K+1), np.real(W_av[:,n]), label='coeff {} {}'.format(str(n), str(complex(w_o[n]))))
    ax[0].set_title('Evolution of the coefficients (real part)')
    ax[0].set_ylabel('Coefficient')
    ax[0].grid(True)
    ax[0].legend()

    for n in np.arange(N):
        ax[1].plot(np.arange(K+1), np.imag(W_av[:,n]), label='coeff {} {}'.format(str(n), str(complex(w_o[n]))))    
    ax[1].set_title('Evolution of the coefficients (imaginary part)')
    ax[1].set_ylabel('Coefficient')
    ax[1].set_xlabel('Number of iterations, k') 
    ax[1].grid(True)
    ax[1].legend()
    
    if output_filepath is not None:        
        fig.savefig(output_filepath + algorithm + '_coef_evolution.jpg', bbox_inches = 'tight')

    plt.show()
