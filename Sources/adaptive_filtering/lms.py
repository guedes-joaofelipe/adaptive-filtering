

__authors__ = ['Joao Felipe Guedes da Silva <guedes.joaofelipe@poli.ufrj.br>']

import numpy as np 
from .utils import rolling_window

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

            # TODO: complex signals don't converge in this step (error vector gets too big elements)
            self.coef_vector[k+1] = self.coef_vector[k]+self.step*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
    def predict(self, x):
        """ Makes predictions for a new signal after weights are fit
        
        Arguments:
            x {row np.array} -- new signal
        
        Returns:
            float -- resulting output"""                
        return np.dot(self.coef_vector[-1], x)


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
