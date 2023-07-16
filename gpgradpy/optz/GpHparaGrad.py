#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:38:21 2022

@author: andremarchildon
"""

import numpy as np

class GpHparaGrad:
    
    def calc_KernGrad_hp(self, hp_optz_info, hp_vals, Rtensor):
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
                = self.calc_Kern_grad_theta(Rtensor, hp_vals.theta, hp_vals.kernel)
                
            if self.wellcond_mtd == 'precon':
                eta = self._etaK
                
                pvec, pvec_inv, grad_precon \
                    = self.calc_Kern_precon(self.n_eval, self.n_grad, hp_vals.theta, 
                                            calc_grad = True, b_return_vec = True)
                gamma_vec  = pvec[self.n_eval:]
                grad_pvec2 = (2 * eta) * gamma_vec[:,None] * grad_precon[self.n_eval:,:]
                
                for i in range(self.dim):
                    KernGrad_hp[hp_optz_info.idx_theta[i], self.n_eval:, self.n_eval:] += np.diag(grad_pvec2[:,i])
                
        if hp_optz_info.has_kernel:
            KernGrad_hp[hp_optz_info.idx_kernel,:,:] \
                = self.calc_Kern_grad_alpha(Rtensor, hp_vals.theta, hp_vals.kernel)

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
        
        n_eval = Rtensor.shape[1]
        n_data = int(n_eval * (self.dim + 1)) if self.use_grad else n_eval
        
        Kcov_grad_hp = np.zeros((hp_optz_info.n_hp, n_data, n_data))

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
                = self.calc_Kcov_grad_var_fval(n_data, hp_vals)

        if hp_optz_info.has_var_fgrad:
            Kcov_grad_hp[hp_optz_info.idx_var_fgrad,:,:] \
                = self.calc_Kcov_grad_var_fgrad(n_data, hp_vals)

        return Kcov_grad_hp

    def calc_Kcov_grad_theta(self, hp_vals, Rtensor):
        
        varK         = self.hp_varK if hp_vals.varK is None else hp_vals.varK
        Kcov_grad_th = varK * self.calc_Kern_grad_theta(Rtensor, hp_vals.theta, hp_vals.kernel) 
        
        if self.wellcond_mtd == 'precon':
            n_eval = Rtensor.shape[1]
            eta    = self._etaK
            
            pvec, pvec_inv, grad_precon = self.calc_Kern_precon(hp_vals.theta, n_eval, calc_grad = True, b_return_vec = True)
            gamma_vec  = pvec[n_eval:]
            gamma_grad = grad_precon[n_eval:,:]
            
            if self.use_grad and self.known_eps_fgrad:
                std_fgrad     = self.get_init_eval_data()[-1]
                vec_var_fgrad = std_fgrad.reshape(std_fgrad.size, order='f')**2
            else:
                vec_var_fgrad = np.full(n_eval * self.dim, hp_vals.var_fgrad)
                
            bvec = vec_var_fgrad > (pvec[n_eval:]**2 * varK * eta)
            
            for i in range(self.dim):
                P_2dPdth       = (2 * eta * varK) * gamma_vec * gamma_grad[:,i]
                P_2dPdth[bvec] = 0
                
                Kcov_grad_th[i, n_eval:,n_eval:] += np.diag(P_2dPdth)
            
        return Kcov_grad_th

    def calc_Kcov_grad_alpha(self, hp_vals, Rtensor):
        
        dKda = self.calc_Kern_grad_alpha(Rtensor, hp_vals.theta, hp_vals.kernel)
        
        return dKda * hp_vals.varK

    def calc_Kcov_grad_varK(self, hp_vals, Kern):
        
        ''' Preliminaries '''
        
        eta    = self._etaK
        varK   = hp_vals.varK
        n_data = Kern.shape[0]
        n_eval = int(n_data / (self.dim + 1)) if self.use_grad else n_data
        theta  = self.hp_theta if hp_vals.theta is None else hp_vals.theta 
        
        if self.known_eps_fval or (self.use_grad and self.known_eps_fgrad):
            std_fval_scl, _, std_fgrad_scl = self.get_scl_eval_data(theta)[1:]
        
        if hp_vals.var_fval is None:
            var_fval = std_fval_scl**2
        else:
            var_fval = hp_vals.var_fval * np.ones(n_eval)   
            
        ''' Calculate derivative '''

        Kern_der_varK = np.copy(Kern)
        
        # Add contribution from noise on the function evaluation
        bvec_fval = (varK * eta) > var_fval
        Kern_der_varK[:n_eval, :n_eval] += np.diag(bvec_fval * eta) 
        
        # Add contribution from noise on the gradient evaluation
        if self.use_grad:
            if self.wellcond_mtd == 'precon':
                pvec      = self.calc_Kern_precon(theta, n_eval, b_return_vec = True)[0]
                gamma_vec = pvec[n_eval:]
                
                if hp_vals.var_fgrad is None:
                    var_fgrad = np.reshape(std_fgrad_scl**2, n_eval * self.dim, order='f')
                else:
                    var_fgrad = hp_vals.var_fgrad
                    
                bvec_fgrad = (varK * eta * gamma_vec**2) > var_fgrad
                grad_vec   = bvec_fgrad * (eta * gamma_vec**2)
                Kern_der_varK[n_eval:, n_eval:] += np.diag(grad_vec)
                    
            else:
                if hp_vals.var_fgrad is None:
                    b_vec = (varK * eta) > np.reshape(std_fgrad_scl**2, n_eval * self.dim, order='f')
                    Kern_der_varK[n_eval:, n_eval:] += np.diag(b_vec * eta)
                else:
                    if (varK * eta) > hp_vals.var_fgrad:
                        Kern_der_varK[n_eval:, n_eval:] += np.diag(np.full(self.dim * n_eval, eta)) # eta * np.eye(self.dim * n_eval)
            
        return Kern_der_varK

    def calc_Kcov_grad_var_fval(self, n_data, hp_vals):
        
        assert hp_vals.varK     is not None, 'The parameter varK cannot be varK'
        assert hp_vals.var_fval is not None, 'The parameter var_fgrad cannot be var_fval'
        
        eta     = self._etaK
        b_check = hp_vals.var_fval > (hp_vals.varK * eta)
        
        if b_check:
            if self.use_grad:
                n_eval = int(n_data / (self.dim + 1)) 
                kernel_der_var_fval                   = np.zeros((n_data, n_data))
                kernel_der_var_fval[:n_eval, :n_eval] = np.eye(n_eval)
            else:
                kernel_der_var_fval = np.eye(n_data)
        else:
            return np.zeros((n_data, n_data))

        return kernel_der_var_fval

    def calc_Kcov_grad_var_fgrad(self, n_data, hp_vals):
        
        assert hp_vals.varK      is not None, 'The parameter varK cannot be varK'
        assert hp_vals.var_fgrad is not None, 'The parameter var_fgrad cannot be var_fgrad'
        assert self.use_grad, 'Method should only be called if gradient information is provided'
        
        eta    = self._etaK
        n_eval = int(n_data // (self.dim + 1)) 
        
        if self.wellcond_mtd == 'precon':
            pvec      = self.calc_Kern_precon(hp_vals.theta, n_eval, b_return_vec = True)[0]
            gamma_vec = pvec[n_eval:]
            bvec      = hp_vals.var_fgrad > (gamma_vec**2 * hp_vals.varK * eta)
            Kcov_grad_var_fgrad = np.diag(np.hstack((np.zeros(n_eval), bvec)))
        else:
            b_check = hp_vals.var_fgrad > (hp_vals.varK * eta)
            
            if b_check:
                Kcov_grad_var_fgrad = np.diag(np.hstack((np.zeros(n_eval), np.ones(n_eval * self.dim))))
            else:
                Kcov_grad_var_fgrad = np.zeros((n_data, n_data))

        return Kcov_grad_var_fgrad
    