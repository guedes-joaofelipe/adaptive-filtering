

__authors__ = ['Joao Felipe Guedes da Silva <guedes.joaofelipe@poli.ufrj.br>']

import numpy as np 
from .utils import rolling_window
from scipy.fftpack import dct
from scipy.linalg import dft

class LMS:
    """ 
    Implements the Complex LMS algorithm for COMPLEX valued data.
        (Algorithm 3.2 - book: Adaptive Filtering: Algorithms and Practical
                                                       Implementation, Diniz)

    Base class for other LMS-based classes

    ...

    Attributes
    ----------    
    . step: (float)
        Convergence (relaxation) factor.
    . filter_order : (int)
        Order of the FIR filter.
    . init_coef : (row np.array)
        Initial filter coefficients.  (optional)
    . d: (row np.array)
        Desired signal. 
    . x: (row np.array)
        Signal fed into the adaptive filter. 
    
    Methods
    -------
    fit(d, x)
        Fits the coefficients according to desired and input signals
    
    predict(x)
        After fitted, predicts new outputs according to new input signal    
    """
    def __init__(self, step, filter_order, init_coef = None):        
        self.step = step
        self.filter_order = filter_order
        self.init_coef = np.array(init_coef)
    
        # Initialization Procedure
        self.n_coef = self.filter_order + 1
        self.d = None
        self.x = None
        self.n_iterations = None
        self.error_vector = None
        self.output_vector = None
        self.coef_vector = None
        
    def __str__(self):
        """ String formatter for the class"""
        return "LMS(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        """ Fits the LMS coefficients according to desired and input signals
        
        Arguments:
            d {np.array} -- desired signal
            x {np.array} -- input signal
        
        Returns:
            {np.array, np.array, np.array} -- output_vector, error_vector, coef_vector
        """
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            self.coef_vector[k+1] = self.coef_vector[k]+self.step*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
    def predict(self, x):
        """ Makes predictions for a new signal after weights are fit
        
        Arguments:
            x {row np.array} -- new signal
        
        Returns:
            float -- resulting output""" 

        # taking the last n_coef iterations of x and making w^t.x
        return np.dot(self.coef_vector[-1], x[:-self.n_coef])


class NLMS(LMS):
    def __init__(self, step, filter_order, gamma, init_coef = None):                
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)        
        self.gamma = gamma
        
    def __str__(self):
        return "NLMS(step={}, gamma={}, filter_order={})".format(self.step, self.gamma, self.filter_order)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]            
            self.coef_vector[k+1] = self.coef_vector[k]+(self.step/(self.gamma+np.dot(np.conj(regressor.T), regressor)))*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector


class SignError(LMS):
    def __init__(self, step, filter_order, init_coef = None):                
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        
    def __str__(self):
        return "SignError(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(regressor), self.coef_vector[k])
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            self.coef_vector[k+1] = self.coef_vector[k]+2*self.step*np.sign(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    

class SignData(LMS):
    def __init__(self, step, filter_order, init_coef = None):                        
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        
    def __str__(self):
        return "SignData(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(regressor), self.coef_vector[k])
            self.error_vector[k] = self.d[k]-self.output_vector[k]
            self.coef_vector[k+1] = self.coef_vector[k]+2*self.step*self.error_vector[k]*np.sign(regressor)
                        
        return self.output_vector, self.error_vector, self.coef_vector


class DualSign(LMS):
    def __init__(self, step, filter_order, rho, gamma, init_coef = None):                        
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        self.rho = rho 
        self.gamma = gamma
        self.dual_sign_error = None

    def __str__(self):
        return "DualSign(step={}, filter_order={}, rho={}, gamma={})".format(self.step, self.filter_order, self.rho, self.gamma)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        self.dual_sign_error = 0

        # Initial State Weight Vector if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = X_tapped[k]

            self.output_vector[k] = np.dot(np.conj(regressor), self.coef_vector[k])
            self.error_vector[k] = self.d[k]-self.output_vector[k]

            self.dual_sign_error = np.sign(self.error_vector[k]) 
            self.dual_sign_error = self.gamma*self.dual_sign_error if (np.absolute(self.error_vector[k]) > self.rho) else self.dual_sign_error

            self.coef_vector[k+1] = self.coef_vector[k]+2*self.step*self.dual_sign_error*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector    


class LMSNewton(LMS):
    def __init__(self, step, filter_order, alpha = .01, init_inv_rx_hat = None, init_coef = None):                        
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        self.alpha = alpha
        self.init_inv_rx_hat = init_inv_rx_hat
        self.dual_sign_error = None
        self.inv_rx_hat = None

    def __str__(self):
        return "LMSNewton(step={}, filter_order={}, alpha={})".format(self.step, self.filter_order, self.alpha)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        
        # Initial Vectors if passed as argument        
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]
        self.inv_rx_hat = self.init_inv_rx_hat if self.init_inv_rx_hat is not None else np.eye(self.n_coef)

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor = np.array(X_tapped[k])
            self.output_vector[k] = np.dot(np.conj(self.coef_vector[k]), regressor)            
            self.error_vector[k] = self.d[k]-self.output_vector[k]

            aux_prod_a = np.dot(self.inv_rx_hat, np.array([regressor]).T)
            aux_prod_b = np.dot(np.conj(regressor).T, self.inv_rx_hat)
            aux_num = np.dot(aux_prod_a, np.array([aux_prod_b]))
            aux_den = (1-self.alpha)/self.alpha + np.dot(np.dot(np.conj(regressor), self.inv_rx_hat), regressor)

            self.inv_rx_hat = 1/(1-self.alpha)*(self.inv_rx_hat - aux_num/aux_den)                        
            self.coef_vector[k+1] = self.coef_vector[k]+self.step*self.error_vector[k]*np.dot(self.inv_rx_hat, regressor)
                        
        return self.output_vector, self.error_vector, self.coef_vector


class TransformDomain(LMS):
    """
        Written and Directed by Michael Bay
    """
    def __init__(self, step, filter_order, init_power, matrix='DCT', alpha = .01, gamma = 1e-8, init_coef=None):        
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        self.alpha = alpha
        self.gamma = gamma
        self.init_power = init_power      
        self.T = None  
        self.matrix = matrix.lower()

    def __str__(self):
        return "TransformDomainDCT(step={}, filter_order={}, alpha={}, gamma={}, init_power={})".format(self.step, self.filter_order, self.alpha, self.gamma, self.init_power)
    
    def get_transform_matrix(self, size):
        """ Calculates the transform matrix based on self.matrix parameter, 
            which can be {'dct', 'dft'}."""

        if self.matrix == 'dct':
            '''Calculate the square DCT transform matrix. Results are 
                equivalent to Matlab dctmtx(n) with 64 bit precision.'''
            transform_matrix = np.array(range(size),np.float64).repeat(size).reshape(size,size)
            transform_matrixT = np.pi * (transform_matrix.transpose()+0.5) / size
            transform_matrixT = (1.0/np.sqrt( size / 2.0)) * np.cos(transform_matrix * transform_matrixT)
            transform_matrixT[0] = transform_matrixT[0] * (np.sqrt(2.0)/2.0)
        elif self.matrix == 'dft':
            transform_matrixT = dft(size, scale='sqrtn')

        return transform_matrixT

    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        self.coef_vector_dct = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        self.T = self.get_transform_matrix(self.n_coef)
        T_hermitian = np.conj(np.transpose(self.T))        
        self.power_vector = self.init_power*np.ones(self.n_coef)        
        self.coef_vector_dct[0] = np.dot(self.T, self.init_coef)

        # Improve source code regularity
        prefixed_x = np.append(np.zeros([self.n_coef-1]), self.x)        
        X_tapped = rolling_window(prefixed_x, self.n_coef)

        for k in np.arange(self.n_iterations):                    
            regressor_dct = np.dot(self.T, X_tapped[k])     
            
            # Summing two column vectors
            self.power_vector = self.alpha*np.multiply(regressor_dct, np.conj(regressor_dct)) + (1-self.alpha)*self.power_vector                        
            self.output_vector[k] = np.dot(np.conj(self.coef_vector_dct[k]), regressor_dct)
            self.error_vector[k] = self.d[k]-self.output_vector[k]

            aux_numerator = np.dot(np.conj(self.error_vector[k]), regressor_dct)
            aux_denominator = self.gamma+self.power_vector            
            self.coef_vector_dct[k+1] = self.coef_vector_dct[k]+self.step*np.divide(aux_numerator, aux_denominator)        
            self.coef_vector[k+1] = np.dot(T_hermitian, self.coef_vector_dct[k+1])
                     
        return self.output_vector, self.error_vector, self.coef_vector, self.coef_vector_dct


class AffineProjection(LMS):
    """
        A final projection
    """
    def __init__(self, step, filter_order, gamma = .001, memory_length=1, init_coef=None):        
        LMS.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                        
        self.gamma = gamma
        self.memory_length = memory_length
        
    def __str__(self):
        return "AffineProjection(step={}, filter_order={}, L={}, gamma={})".format(self.step, self.filter_order, self.memory_length, self.gamma)

    def fit(self, d, x):
        # Pre allocations
        self.d = d
        self.x = x
        self.n_iterations = len(self.d)

        # Initialization
        d_ap = np.zeros(self.memory_length+1, dtype=complex)
        output_vector_ap = np.zeros(self.memory_length+1, dtype=complex)
        error_vector_ap = np.zeros(self.memory_length+1, dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)

        # Input initial conditions (assumed relaxed)
        input_vector = np.zeros(self.n_coef, dtype=complex)
        input_ap = np.zeros([self.memory_length+1, self.n_coef], dtype=complex)                

        for k in np.arange(self.n_iterations):
            input_vector    = np.roll(input_vector,1)
            input_vector[0] = self.x[k]           
                        
            input_ap    = np.roll(input_ap, 1, axis = 0)
            input_ap[0] = input_vector
            
            d_ap    = np.roll(d_ap,1)
            d_ap[0] = self.d[k]
            
            output_vector_ap = np.matmul(np.conj(self.coef_vector[k]),np.transpose(input_ap))
            error_vector_ap  = d_ap - output_vector_ap
            
            self.coef_vector[k+1] = self.coef_vector[k] + self.step*np.matmul(np.conj(error_vector_ap),np.matmul(np.transpose(np.linalg.inv(np.matmul(np.conj(input_ap),np.transpose(input_ap))+ self.gamma*np.eye(self.memory_length+1, dtype = complex))),input_ap))
            self.output_vector[k] = output_vector_ap[0]
            self.error_vector[k]  = error_vector_ap[0]

        return self.output_vector, self.error_vector, self.coef_vector