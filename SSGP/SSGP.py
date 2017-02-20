################################################################################
#  SSGP: Sparse Spectrum Gaussian Process
#  Github: https://github.com/MaxInGaussian/SSGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import math
import random
import numpy as np
import scipy.linalg as la
from .SMORMS3 import SMORMS3

class SSGP(object):
    
    """ Sparse Spectrum Gaussian Process """
    
    hashed_name = ""
    m, n, d = -1, -1, -1
    freq_noisy = True
    y_noise, sigma, lengthscales, S = None, None, None, None
    X_train, y_train = None, None
    X_valid, y_valid = None, None
    X_scaler, y_scaler = None, None
    
    def __init__(self, m=-1, freq_noisy=True):
        self.m = m
        self.freq_noisy = freq_noisy
        self.hashed_name = random.choice("ABCDEF")+str(hash(self)&0xffff)
    
    def transform(self, X=None, y=None):
        _X, _y = None, None
        if(X is not None):
            _X = 3.*(X-self.X_scaler[0])/self.X_scaler[1]
        if(y is not None):
            _y = (y-self.y_scaler[0])/self.y_scaler[1]
        return _X, _y
    
    def inverse_transform(self, X=None, y=None):
        _X, _y = None, None
        if(X is not None):
            _X = X/3.*self.X_scaler[1]+self.X_scaler[0]
        if(y is not None):
            _y = y*self.y_scaler[1]+self.y_scaler[0]
        return _X, _y
    
    def init_params(self, rand_num=100):
        if(self.freq_noisy):
            log_y_noise = np.random.randn(self.m)*1e-1
        else:
            log_y_noise = np.random.randn(1)*1e-1
        log_sigma = np.random.randn(1)*1e-1
        ranges = np.max(self.X_train, 0)-np.min(self.X_train, 0)
        log_lengthscales = np.log(ranges/2.)
        best_nlml = np.Infinity
        best_rand_params = np.zeros(self.d+1+self.m*(1+self.d))
        kern_params = np.concatenate((log_y_noise, log_sigma, log_lengthscales))
        for _ in range(rand_num):
            spectrum_params = np.random.randn(self.m*self.d)
            rand_params = np.concatenate((kern_params, spectrum_params))
            self.set_params(rand_params)
            nlml = self.get_nlml()
            if(nlml < best_nlml):
                best_nlml = nlml
                best_rand_params = rand_params
        self.set_params(best_rand_params)
    
    def get_params(self):
        sn = 1
        if(self.freq_noisy):
            sn = self.m
        params = np.zeros(self.d+1+self.m*self.d+sn)
        params[:sn] = np.log(self.y_noise)/2.
        params[sn] = np.log(self.sigma)/2.
        log_lengthscales = np.log(self.lengthscales)
        params[sn+1:sn+self.d+1] = log_lengthscales
        spectrum = self.S*np.tile(self.lengthscales[None, :], (self.m, 1))
        params[sn+self.d+1:] = np.reshape(spectrum, (self.m*self.d,))
        return params
    
    def set_params(self, params):
        sn = 1
        if(self.freq_noisy):
            sn = self.m
        self.y_noise = np.exp(2*params[:sn])
        self.sigma = np.exp(2*params[sn])
        self.lengthscales = np.exp(params[sn+1:sn+self.d+1])
        self.S = np.reshape(params[sn+self.d+1:], (self.m, self.d))
        self.S /= np.tile(self.lengthscales[None, :], (self.m, 1))
        self.Phi = self.X_train.dot(self.S.T)
        cosX = np.cos(self.Phi)
        sinX = np.sin(self.Phi)
        self.Phi = np.concatenate((cosX, sinX), axis=1)
        A = self.sigma/self.m*self.Phi.T.dot(self.Phi)
        if(self.freq_noisy):
            noise_diag = np.diag(np.concatenate((self.y_noise, self.y_noise)))
        else:
            noise_diag = np.double(self.y_noise)*np.eye(2*self.m)
        self.R = la.cho_factor(A+noise_diag)[0]
        self.PhiRi = la.solve_triangular(self.R, self.Phi.T, trans=1).T
        self.RtiPhit = self.PhiRi.T
        self.Rtiphity = self.RtiPhit.dot(self.y_train)
        self.alpha = la.solve_triangular(self.R, self.Rtiphity)
        self.alpha *= self.sigma/self.m
    
    def get_nlml(self):
        sn = self.m
        if(self.freq_noisy):
            sn = 1
        L1 = np.sum(self.y_train**2)-self.sigma/self.m*np.sum(self.Rtiphity**2.)
        L2 = np.sum(np.log(np.diag(self.R)))
        L3 = self.n/2*np.log(np.mean(self.y_noise))
        L3 -= np.sum(np.log(self.y_noise))*sn
        L4 = self.n/2*np.log(2*np.pi)
        nlml = 0.5/np.mean(self.y_noise)*L1+L2+L3+L4
        return nlml
        
    def get_pnlml(self, penalty=['bridge', 0.8, 0.01]):
        nlml = self.get_nlml()
        pnlml = nlml/self.n
        for l in self.lengthscales:
            if(penalty[0] == 'ridge'):
                lamb = penalty[1]/(self.d)
                pnlml += lamb*(np.abs(1./l)**2.)
            if(penalty[0] == 'lasso'):
                lamb = penalty[1]/(self.d)
                pnlml += lamb*(np.abs(1./l))
            if(penalty[0] == 'bridge'):
                lamb = penalty[1]/(self.d)
                gamma = penalty[2]
                pnlml += lamb*(np.abs(1./l)**gamma)
        for i in range(self.m*self.d):
            s = self.S[i/self.d, i%self.d]
            if(penalty[0] == 'ridge'):
                lamb = penalty[1]/(self.m*self.d)
                pnlml += lamb*(np.abs(s)**2.)
            if(penalty[0] == 'lasso'):
                lamb = penalty[1]/(self.m*self.d)
                pnlml += lamb*(np.abs(s))
            if(penalty[0] == 'bridge'):
                lamb = penalty[1]/(self.m*self.d)
                gamma = penalty[2]
                pnlml += lamb*(np.abs(s)**gamma)
        return pnlml
    
    def get_nlml_grad(self):
        sn = 1
        if(self.freq_noisy):
            sn = self.m
        grad = np.zeros(self.d+1+self.m*self.d+sn)
        a1 = self.y_train/np.mean(self.y_noise)
        const = self.sigma/self.m
        noise_diag = const/np.mean(self.y_noise)*np.eye(2*self.m)
        a1 -= self.PhiRi.dot(noise_diag.dot(self.Rtiphity))
        a2 = self.PhiRi.dot(np.sqrt(noise_diag))
        A = np.concatenate((a1, a2), axis=1)
        diagfact = -1./np.mean(self.y_noise)+np.sum(A**2, axis=1)
        AtPhi = A.T.dot(self.Phi)
        B = A.dot(AtPhi[:, 0:self.m])*self.Phi[:, self.m:]
        B -= A.dot(AtPhi[:, self.m:])*self.Phi[:, 0:self.m]
        grad[:sn] = -1*np.sum(diagfact)*self.y_noise
        grad[sn] = self.n*self.m/np.mean(self.y_noise)
        grad[sn] -= np.sum(np.sum(AtPhi**2))
        grad[sn] *= (self.sigma/self.m)
        for i in range(self.d):
            grad[sn+1+i] = self.X_train[:, i].dot(B).dot(self.S[:, i])*-const
            grad[self.d+1+sn+i*self.m:self.d+1+sn+(1+i)*self.m] =\
                self.X_train[:, i].dot(B)*const/self.lengthscales[i]
        return grad
    
    def get_pnlml_grad(self, penalty=['bridge', 0.8, 0.01]):
        sn = 1
        if(self.freq_noisy):
            sn = self.m
        nlml_grad = self.get_nlml_grad()
        pnlml_grad = nlml_grad/self.n
        for i in range(sn+1, sn+self.d+1):
            l = self.lengthscales[i-sn-1]
            if(penalty[0] == 'ridge'):
                lamb = penalty[1]/(self.d)
                pnlml_grad[i] += lamb*(-2/l**3.)
            if(penalty[0] == 'lasso'):
                lamb = penalty[1]/(self.d)
                pnlml_grad[i] += lamb*(-l/np.abs(l)**3.)
            if(penalty[0] == 'bridge'):
                lamb = penalty[1]/(self.d)
                gamma = penalty[2]
                pnlml_grad[i] += -lamb*l*gamma*(np.abs(1./l)**(gamma+2))
        for i in range(sn+self.d+1, len(nlml_grad)):
            s = self.S[(i-(sn+self.d+1))/self.d, (i-(sn+self.d+1))%self.d]
            if(penalty[0] == 'ridge'):
                lamb = penalty[1]/(self.m*self.d)
                pnlml_grad[i] += lamb*2*s
            if(penalty[0] == 'lasso'):
                lamb = penalty[1]/(self.m*self.d)
                pnlml_grad[i] += lamb*s/np.abs(s)
            if(penalty[0] == 'bridge'):
                lamb = penalty[1]/(self.m*self.d)
                gamma = penalty[2]
                pnlml_grad[i] += lamb*s*gamma*(np.abs(s)**(gamma-2))
        return pnlml_grad
    
    def fit(self, X_train, y_train, trainer=None):
        if(trainer is None):
            trainer = SMORMS3(self)
        self.X_scaler = (np.min(X_train, axis=0), np.max(X_train, axis=0))
        self.y_scaler = (np.mean(y_train, axis=0), np.std(y_train, axis=0))
        self.X_train, self.y_train = self.transform(X_train, y_train)
        self.n, self.d = self.X_train.shape
        self.init_params()
        trainer.train()
        
    def predict(self, X_test, y_test=None):
        X, _ = self.transform(X_test)
        PhiS = X.dot(self.S.T)
        cosX = np.cos(PhiS)
        sinX = np.sin(PhiS)
        PhiS = np.concatenate((cosX, sinX), axis=1)
        mu = PhiS.dot(self.alpha)
        _, mu = self.inverse_transform(None, mu)
        PhiSRi = la.solve_triangular(self.R, PhiS.T, trans=1).T
        std = np.mean(self.y_noise)*(1+self.sigma/self.m*np.sum(PhiSRi**2, 1))
        std = self.y_scaler[1]*np.sqrt(std)
        if(y_test is None):
            return mu, std
        y_mu = np.mean(y_test.ravel())
        self.mse = np.mean(np.square(y_test-mu))
        self.nmse = np.mean(np.square(y_test-mu))/np.mean(np.square(y_test-y_mu))
        self.mnlp = np.mean(np.square((y_test-mu)/std)+2*np.log(std))
        self.mnlp = 0.5*(np.log(2*np.pi)+self.mnlp)
        return self.mse, self.nmse, self.mnlp
    
    def save(self, path):
        prior_settings = (self.m, self.freq_noisy, self.hashed_name)
        raw_data = (self.n, self.d, self.X_scaler, self.y_scaler)
        hyper_params = (self.y_noise, self.sigma, self.lengthscales, self.S)
        computed_matrices = (self.R, self.alpha)
        performances = (self.nmse, self.mnlp)
        save_pack = [prior_settings, raw_data, hyper_params,
                     computed_matrices, performances]
        import pickle
        with open(path, "wb") as save_f:
            pickle.dump(save_pack, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_pack = pickle.load(load_f)
        self.m, self.freq_noisy, self.hashed_name = load_pack[0]
        self.n, self.d, self.X_scaler, self.y_scaler = load_pack[1]
        self.y_noise, self.sigma, self.lengthscales, self.S = load_pack[2]
        self.R, self.alpha = load_pack[3]
        self.nmse, self.mnlp = load_pack[4]
    
    
    