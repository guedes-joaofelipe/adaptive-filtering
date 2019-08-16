import numpy as np 

class LMS:
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
        return "LMS(step={}, filter_order={})".format(self.step, self.filter_order)
        
    def fit(self, d, x):
        # Pre allocations
        self.d = np.array(d)        
        self.x = np.array(x)        
        self.n_iterations = len(self.d)
        self.output_vector = np.zeros([self.n_iterations], dtype=complex)
        self.error_vector = np.zeros([self.n_iterations], dtype=complex)
        self.coef_vector = np.zeros([self.n_iterations+1, self.n_coef], dtype=complex)

        # Initial State Weight Vector if passed as argument
        #self.coef_vector = np.array([self.init_coef]) if self.init_coef is not None else np.array([np.zeros([self.n_coef])])
        self.coef_vector[0] = np.array([self.init_coef]) if self.init_coef is not None else self.coef_vector[0]

        # Improve source code regularity
        prefixed_input = np.append(np.zeros([self.n_coef-1]), self.x)
        
        for i in np.arange(self.n_iterations):        
            regressor = prefixed_input[i+self.n_coef-1:i-1:-1] if i != 0 else prefixed_input[i+self.n_coef-1::-1]
            
            self.output_vector[i] = np.dot(self.coef_vector[i], regressor)            
            self.error_vector[i] = self.d[i]-self.output_vector[i]
            self.coef_vector[i+1] = self.coef_vector[i]+self.step*np.conj(self.error_vector[i])*regressor
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
    def predict(self, x):
        # Makes predictions for a new signal after weights are fit
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
        
        # Initial State Weight Vector if passed as argument
        self.coef_vector = np.array([self.init_coef]) if self.init_coef is not None else np.array([np.zeros([self.n_coef])])

        # Improve source code regularity
        prefixed_input = np.append(np.zeros([self.n_coef]), self.x)
        
        for i in np.arange(self.n_iterations):        
            regressor = prefixed_input[i+self.n_coef:i:-1]            
            y = np.dot(self.coef_vector[i], regressor)            
            if i == 0:
                self.output_vector = np.array([y])
                self.error_vector = np.array([self.d[i]-self.output_vector[i]])
            else:
                self.output_vector = np.append(self.output_vector, y)
                self.error_vector = np.append(self.error_vector, self.d[i]-self.output_vector[i])
            
            self.error_vector[i] = self.d[i]-self.output_vector[i]            
            self.coef_vector = np.append(self.coef_vector, [self.step/(self.gamma+np.dot(regressor.T, regressor))*np.conj(self.error_vector[i])*regressor], axis=0)
                        
        return self.output_vector, self.error_vector, self.coef_vector
    
