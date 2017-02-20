################################################################################
#  SSGP: Sparse Spectrum Gaussian Process
#  Github: https://github.com/MaxInGaussian/SSGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import abc
import numpy as np

class Trainer(object):
    
    ssgp = None
    penalty = None
    maxIters, maxBestIters, erorTol = np.Infinity, 50, 1e-2
    iter, min_err, bestIter = 0, 1e10, 0
    best_params = None
    
    def __init__(self, ssgp, penalty=None, options=None):
        self.ssgp = ssgp
        self.penalty = penalty
        if(options is not None):
            self.maxIters, self.maxBestIters, self.erorTol = options
    
    def cost(self):
        if(self.penalty is None):
            return self.ssgp.get_nlml()
        else:
            return self.ssgp.get_pnlml(self.penalty)
    
    def numericGrad(self):
        epsilon = 1e-8
        params_ori = self.ssgp.get_params()
        cost_ori = self.cost()
        grad = np.zeros(params_ori.shape)
        for i in range(grad.size):
            params_ori[i] += epsilon
            self.ssgp.set_params(params_ori.copy())
            cost_plus = self.cost()
            params_ori[i] -= epsilon
            grad[i] = (cost_plus-cost_ori)/(epsilon)
        self.ssgp.set_params(params_ori.copy())
        return grad

    def train(self):
        self.best_params = self.ssgp.get_params()
        self.iter, self.min_err, self.bestIter = 0, 1e10, 0
        N = self.ssgp.n
        while(True):
            grad = None
            if(self.penalty is None):
                grad = self.ssgp.get_nlml_grad()
            else:
                grad = self.ssgp.get_pnlml_grad(self.penalty)
            params = self.ssgp.get_params()
            self.ssgp.set_params(self.apply_update_rule(params, grad))
            self.iter += 1
            err = self.cost()
            self.err_diff = abs(err-self.min_err)
            if(err < self.min_err):
                if(self.min_err - err < self.erorTol):
                    self.bestIter += 1
                else:
                    self.bestIter = 0
                self.min_err = err
                self.best_params = self.ssgp.get_params()
            else:
                self.bestIter += 1
            print(self.iter, ":", self.min_err)
            if(self.stop_condition()):
                self.ssgp.set_params(self.best_params)
                break
    
    def stop_condition(self):
        if(self.iter==self.maxIters or self.bestIter == self.maxBestIters):
            return True
        return False
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def apply_update_rule(self, params, grad):
        pass
            
            