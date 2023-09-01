#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:38:21 2022

@author: andremarchildon
"""

import numpy as np

class GpHparaGrad:
    
    def calc_KernGrad_hp(self, hp_optz_info, hp_vals, Rtensor, etaK = None):
        '''
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from file GpHparaOptz
            Info on the numerical optimization of the hyperparameters.
        hp_vals : HparaOptzVal from GpHpara
            Class that contains all of the hyperparameters
        Rtensor : 3D numpy array of floats of shape [dim, n_data, n_data]
            Distance between two sets of nodes along all dim coordinates.

        Returns
        -------
        KernGrad_hp : 3D numpy array of floats
            Derivative of the regularized kernel matrix with respect to all of the hyperparameters.
        '''
        
        assert hp_optz_info.has_varK      is False, 'If has_varK is True, use, calc_Kcov_grad_hp()'
        assert hp_optz_info.has_var_fval  is False, 'If has_var_fval is True, use, calc_Kcov_grad_hp()'
        assert hp_optz_info.has_var_fgrad is False, 'If has_var_fgrad is True, use, calc_Kcov_grad_hp()'

        KernGrad_hp = np.zeros((hp_optz_info.n_hp, self.n_data, self.n_data))

        if hp_optz_info.has_theta:
            KernGrad_hp[hp_optz_info.idx_theta,:,:] \
                = self.calc_Kern_grad_theta(Rtensor, hp_vals.theta, hp_vals.kernel, self.bvec_use_grad)
                
            if self.wellcond_mtd == 'precon':
                if etaK is None: etaK = self._etaK
                
                pvec, pvec_inv, grad_precon \
                    = self.calc_Kern_precon(self.n_eval, self.n_grad, hp_vals.theta, 
                                            calc_grad = True, b_return_vec = True)
                gamma_vec  = pvec[self.n_eval:]
                grad_pvec2 = (2 * etaK) * gamma_vec[:,None] * grad_precon[self.n_eval:,:]
                
                for i in range(self.dim):
                    KernGrad_hp[hp_optz_info.idx_theta[i], self.n_eval:, self.n_eval:] += np.diag(grad_pvec2[:,i])
                
        if hp_optz_info.has_kernel:
            KernGrad_hp[hp_optz_info.idx_kernel,:,:] \
                = self.calc_Kern_grad_alpha(Rtensor, hp_vals.theta, hp_vals.kernel, self.bvec_use_grad)

        return KernGrad_hp

    def calc_Kcov_grad_hp(self, hp_optz_info, hp_vals, Kern, Rtensor):
        '''
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from file GpHparaOptz
            Info on the numerical optimization of the hyperparameters.
        hp_vals : HparaOptzVal from GpHpara
            Class that contains all of the hyperparameters
        Rtensor : 3D numpy array of floats of shape [dim, n_data, n_data]
            Distance between two sets of nodes along all dim coordinates.

        Returns
        -------
        Kcov_grad_hp : 3D numpy array of floats
            Derivative of the covariance matrix with respect to all of the hyperparameters.
        '''
        
        assert hp_vals.varK is not None, 'The method calc_Kcov_grad_hp() only needs to be called if varK is a hyperparameter'
        
        Kcov_grad_hp = np.zeros((hp_optz_info.n_hp, self.n_data, self.n_data))

        if hp_optz_info.has_theta:
            Kcov_grad_hp[hp_optz_info.idx_theta,:,:] \
                = self.calc_Kcov_grad_theta(hp_vals, Rtensor)

        if hp_optz_info.has_kernel:
            Kcov_grad_hp[hp_optz_info.idx_kernel,:,:] = self.calc_Kcov_grad_alpha(hp_vals, Rtensor)
            
        if hp_optz_info.has_varK:
            Kcov_grad_hp[hp_optz_info.idx_varK,:,:] \
                = self.calc_Kcov_grad_varK(hp_vals, Kern)

        if hp_optz_info.has_var_fval:
            Kcov_grad_hp[hp_optz_info.idx_var_fval,:,:] \
                = self.calc_Kcov_grad_var_fval(hp_vals)

        if hp_optz_info.has_var_fgrad:
            Kcov_grad_hp[hp_optz_info.idx_var_fgrad,:,:] \
                = self.calc_Kcov_grad_var_fgrad(hp_vals)

        return Kcov_grad_hp

    def calc_Kcov_grad_theta(self, hp_vals, Rtensor):
        
        varK         = self.hp_varK if hp_vals.varK is None else hp_vals.varK
        Kcov_grad_th = varK * self.calc_Kern_grad_theta(Rtensor, hp_vals.theta, hp_vals.kernel, self.bvec_use_grad) 
        
        if self.wellcond_mtd == 'precon':
            eta = self._etaK
            for i in range(self.dim):
                diag_Kcov_grad_th_i = np.diag(Kcov_grad_th[i])
                Kcov_grad_th[i]    += np.diag(diag_Kcov_grad_th_i) * eta
            
        return Kcov_grad_th

    def calc_Kcov_grad_alpha(self, hp_vals, Rtensor):
        
        varK            = self.hp_varK if hp_vals.varK is None else hp_vals.varK
        Kcov_grad_alpha = varK * self.calc_Kern_grad_alpha(Rtensor, hp_vals.theta, hp_vals.kernel, self.bvec_use_grad)
        
        if self.wellcond_mtd == 'precon':
            eta = self._etaK
            n_alpha = Kcov_grad_alpha.shape[0]
            
            for i in range(n_alpha):
                diag_Kcov_grad_alpha_i = np.diag(Kcov_grad_alpha[i])
                Kcov_grad_alpha[i]    += np.diag(diag_Kcov_grad_alpha_i) * eta
        
        return Kcov_grad_alpha

    def calc_Kcov_grad_varK(self, hp_vals, Kern):
        
        eta = self._etaK
        
        if self.wellcond_mtd == 'precon':
            Kcov_der_varK = Kern + eta * np.diag(np.diag(Kern))
        else:
            Kcov_der_varK = Kern + eta * np.eye(Kern.shape[0])

        return Kcov_der_varK

    def calc_Kcov_grad_var_fval(self, hp_vals):
        
        if self.wellcond_mtd == 'precon':
            scalar = 1 + self._etaK
        else:
            scalar = 1
        
        return scalar * np.diag(np.hstack((np.ones(self.n_eval), np.zeros(self.n_grad * self.dim))))
        
    def calc_Kcov_grad_var_fgrad(self, hp_vals):
        
        if self.wellcond_mtd == 'precon':
            scalar = 1 + self._etaK
        else:
            scalar = 1
            
        return scalar * np.diag(np.hstack((np.zeros(self.n_eval), np.ones(self.n_grad * self.dim))))
