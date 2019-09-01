"""

    !!!!!!!!!!!!!!!!!!!!!!!!
    Classes to be tested yet
    !!!!!!!!!!!!!!!!!!!!!!!!

"""


__authors__ = ['Joao Felipe Guedes da Silva <guedes.joaofelipe@poli.ufrj.br>']

import numpy as np 
from .utils import rolling_window

class Blind:
    
    def __init__(self, step, filter_order, init_coef = None):        
                
        self.filter_order = filter_order
        self.step = step
        self.init_coef = init_coef

        # Initialization Procedure
        self.n_coef = self.filter_order + 1        
        self.x = None
        self.n_iterations = None
        self.error_vector = None
        self.output_vector = None
        self.coef_vector = None
        
    def __str__(self):
        """ String formatter for the class"""
        return "Blind(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        # To be implemented in the child classes
        raise NotImplementedError
            
    def predict(self, x):
        # To be implemented in the child classes
        raise NotImplementedError
        

class Sato(Blind):
    def __init__(self, step, filter_order, init_coef = None):                
        Blind.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        
        self.desired_level = None # defines the level which 
                                # abs(outputVector(it,1))^2 should approach

    def __str__(self):
        return "Sato(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, x):
        # Pre allocations        
        self.x = np.array(x)        
        self.desired_level = np.mean(np.abs(x)**2)/np.mean(np.abs(x))
        self.n_iterations = len(self.x)

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
            self.error_vector[k] = self.output_vector[k]-np.sign(self.output_vector[l])*self.desired_level
            self.coef_vector[k+1] = self.coef_vector[k]-self.step*np.conj(self.error_vector[k])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector

class Godard(Blind):
    def __init__(self, step, filter_order, p_exponent, q_exponent, init_coef = None):                
        Blind.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        
        self.p_exponent = p_exponent # Godard-error's exponent
        self.q_exponent = q_exponent # Exponent used to define the desired "output level" (desiredLevel)
        self.desired_level = None # defines the level which  abs(outputVector(it,1))^2 should approach

    def __str__(self):
        return "Sato(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, x):
        # Pre allocations        
        self.x = np.array(x)        
        self.desired_level = np.mean(np.abs(x)**(2*self.q_exponent))/np.mean(np.abs(x)**self.q_exponent)
        self.n_iterations = len(self.x)

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
            self.error_vector[k] = np.abs(self.output_vector[k])**self.q_exponent-self.desired_level
            self.coef_vector[k+1] = self.coef_vector[k]- \
                self.step*self.p_exponent*self.q_exponent*(self.error_vector[k]**(self.p_exponent-1))* \
                    (np.abs(self.output_vector[k])**(self.q_exponent-2))*np.conj(self.output_vector[k])*regressor/2
                        
        return self.output_vector, self.error_vector, self.coef_vector

    
class CMA(Blind):
    def __init__(self, step, filter_order, init_coef = None):                
        Blind.__init__(self, step=step, filter_order=filter_order, init_coef=init_coef)                
        
    def __str__(self):
        return "CMA(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, x):
        # Pre allocations        
        self.x = np.array(x)        
        self.desired_level = np.mean(np.abs(x)**4)/np.mean(np.abs(x)**2)
        self.n_iterations = len(self.x)

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
            self.error_vector[k] = np.abs(self.output_vector[k])**2-self.desired_level
            self.coef_vector[k+1] = self.coef_vector[k]- \
                self.step*2self.error_vector[k]*np.conj(self.output_vector[k])*regressor/2
                        
        return self.output_vector, self.error_vector, self.coef_vector
