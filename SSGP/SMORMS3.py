################################################################################
#  SSGP: Sparse Spectrum Gaussian Process
#  Github: https://github.com/MaxInGaussian/SSGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from .Trainer import Trainer

class SMORMS3(Trainer):
    
    mem, g, g2 = None, None, None
    sumGrad2, learningRate, fixedRate = -1, -1, 1e-1
    
    def __init__(self, ssgp, penalty=None, options=None, fixedRate=1e-1):
        self.mem = None
        self.g = None
        self.g2 = None
        self.fixedRate = fixedRate
        super(SMORMS3, self).__init__(ssgp, penalty, options)
    
    def apply_update_rule(self, params, grad):
        if(self.mem is None):
            self.mem = np.ones(params.shape)
            self.g = np.zeros(params.shape)
            self.g2 = np.zeros(params.shape)
        r = 1/(self.mem+1)
        self.g = (1-r)*self.g+r*grad
        self.g2 = (1-r)*self.g2+r*grad**2
        rate = self.g*self.g/(self.g2+1e-16)
        self.mem *= (1 - rate)
        self.learningRate = self.fixedRate/(max(self.bestIter, 7))
        self.mem += 1
        alpha = np.minimum(self.learningRate, rate)/(np.sqrt(self.g2)+1e-16)
        return params-grad*alpha