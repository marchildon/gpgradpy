#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:42:27 2021

@author: andremarchildon
"""

import time
import numpy as np

from scipy.linalg import cho_factor

from . import KernelSqExp
from . import KernelRatQuad
from . import KernelMatern5f2


class Kernel(KernelSqExp, KernelRatQuad, KernelMatern5f2):

    def __init__(self, kernel_type = 'SqExp'):

        # Add inputs to the class
        self.kernel_type = kernel_type

        # Set methods
        if self.kernel_type == 'SqExp':
            
            # Hyper-parameter for the kernel
            self.hp_kernel_default        = self.sq_exp_hp_kernel_default
            self.range_hp_kernel          = self.sq_exp_range_hp_kernel

            # Kernel calculation
            self.calc_KernBase            = self.sq_exp_calc_KernBase
            self.calc_KernGrad            = self.sq_exp_calc_KernGrad
            
            # Correlation matrix 
            self.theta2gamma              = self.sq_exp_theta2gamma
            self.gamma2theta              = self.sq_exp_gamma2theta
            self.calc_Kern_precon         = self.sq_exp_Kern_precon
            
            # For optimizing the hyper-parameter theta for the kernel matrix
            self.calc_KernBase_grad_th    = self.sq_exp_calc_KernBase_grad_th
            self.calc_KernGrad_grad_th    = self.sq_exp_calc_KernGrad_grad_th
            
            # For optimizing the kernel hyper-parameters
            self.calc_KernBase_grad_alpha = self.sq_exp_calc_KernBase_grad_alpha
            self.calc_KernGrad_grad_alpha = self.sq_exp_calc_KernGrad_grad_alpha
            
            # Derivative of the kernel matrix wrt to its first argument x
            self.calc_KernBase_hess_x     = self.sq_exp_calc_KernBase_hess_x
            self.calc_KernGrad_grad_x     = self.sq_exp_calc_KernGrad_grad_x
            
        elif self.kernel_type == 'Ma5f2':
            
            # Hyper-parameter for the kernel
            self.hp_kernel_default        = self.matern_5f2_hp_kernel_default
            self.range_hp_kernel          = self.matern_5f2_range_hp_kernel

            # Kernel calculation
            self.calc_KernBase            = self.matern_5f2_calc_KernBase
            self.calc_KernGrad            = self.matern_5f2_calc_KernGrad
            
            # Correlation matrix 
            self.theta2gamma              = self.matern_5f2_theta2gamma
            self.gamma2theta              = self.matern_5f2_gamma2theta
            self.calc_Kern_precon         = self.matern_5f2_Kern_precon
            
            # For optimizing the hyper-parameter theta for the kernel matrix
            self.calc_KernBase_grad_th    = self.matern_5f2_calc_KernBase_grad_th
            self.calc_KernGrad_grad_th    = self.matern_5f2_calc_KernGrad_grad_th
            
            # For optimizing the kernel hyper-parameters
            self.calc_KernBase_grad_alpha = self.matern_5f2_calc_KernBase_grad_alpha
            self.calc_KernGrad_grad_alpha = self.matern_5f2_calc_KernGrad_grad_alpha
            
            # Derivative of the kernel matrix wrt to its first argument x
            self.calc_KernBase_hess_x     = self.matern_5f2_calc_KernBase_hess_x
            self.calc_KernGrad_grad_x     = self.matern_5f2_calc_KernGrad_grad_x
            
        elif self.kernel_type == 'RatQu':
            
            # Hyper-parameter for the kernel
            self.hp_kernel_default        = self.rat_quad_hp_kernel_default
            self.range_hp_kernel          = self.rat_quad_range_hp_kernel

            # Kernel calculation
            self.calc_KernBase            = self.rat_quad_calc_KernBase
            self.calc_KernGrad            = self.rat_quad_calc_KernGrad
            
            # Correlation matrix 
            self.theta2gamma              = self.rat_quad_theta2gamma
            self.gamma2theta              = self.rat_quad_gamma2theta
            self.calc_Kern_precon         = self.rat_quad_Kern_precon
            
            # For optimizing the hyper-parameter theta for the kernel matrix
            self.calc_KernBase_grad_th    = self.rat_quad_calc_KernBase_grad_th
            self.calc_KernGrad_grad_th    = self.rat_quad_calc_KernGrad_grad_th

            # For optimizing the hyper-parameter alpha of the kernel matrix
            self.calc_KernBase_grad_alpha = self.rat_quad_calc_KernBase_grad_alpha
            self.calc_KernGrad_grad_alpha = self.rat_quad_calc_KernGrad_grad_alpha
            
            # Derivative of the kernel matrix wrt to its first argument x
            self.calc_KernBase_hess_x     = self.rat_quad_calc_KernBase_hess_x
            self.calc_KernGrad_grad_x     = self.rat_quad_calc_KernGrad_grad_x
            
        else:
            raise Exception('Kernel type is not available')
            
        self.kernel_has_hp = False if (self.hp_kernel_default is None) else True 
        self.hp_kernel     = self.hp_kernel_default

        # Set the method depending on whether gradients are used or not
        if self.use_grad:
            self.calc_Kern              = self.calc_KernGrad
            self.calc_Kern_hess_x       = self.calc_KernGrad_grad_x
            
            self.calc_Kern_grad_theta   = self.calc_KernGrad_grad_th
            self.calc_Kern_grad_alpha   = self.calc_KernGrad_grad_alpha
        else:   
            self.calc_Kern              = self.calc_KernBase
            self.calc_Kern_hess_x       = self.calc_KernBase_hess_x
            
            self.calc_Kern_grad_theta   = self.calc_KernBase_grad_th
            self.calc_Kern_grad_alpha   = self.calc_KernBase_grad_alpha

    def calc_Kern_w_chofac(self, Rtensor, hp_vals, 
                           noise_vec = None,
                           calc_chofac   = True, calc_cond = False):
        ''' 
        See input info for calc_all_K_w_chofac()
        '''
    
        assert self.b_has_noisy_data is False, 'This function should not be called if there is noisy data'
        
        # Set varK to 1 such that the Kcov and Kern are the same with the addtion of the nugget
        Kern, Kcor, Kern_w_eta, Kern_chofac, cond_K \
            = self.calc_all_K_w_chofac(Rtensor, hp_vals, noise_vec,
                                       calc_chofac, calc_cond, varK = 1)
            
        return Kern, Kcor, Kern_w_eta, Kern_chofac, cond_K

    def calc_all_K_w_chofac(self, Rtensor, hp_vals,  
                            noise_vec   = None,
                            calc_chofac = True, calc_cond       = False,
                            varK        = None, b_normlz_w_varK = False):
        '''
        Parameters
        ----------
        Rtensor : 3D numpy tensor
            Relative distance between all data points in each coordinate 
            direction. Calculated with calc_Rtensor() from CommonFun
        hp_vals : HparaOptzVal from GpHpara
            Class that contains all of the hyperparameters
        bvec_use_grad : 1D numpy array of bool of length n_eval, optional
            Indicates which gradients to use to construct the kernel matrix. 
            The default is None.
        noise_vec : 1D numpy array, optional
            Noise for each of the function evalution values and gradients 
            (if needed). The default is None.
        calc_chofac : bool, optional
            Set to True to calculate the Cholesky decompositon of the 
            regularized kernel matrix. The default is True.
        calc_cond : bool, optional
            Set to true to calculate the condition number of the . The default is False.
        varK : float, optional
            Value for the hyperparameter varK. The default is None.
        b_normlz_w_varK : bool, optional
            If set to True, the kernel matrix is not multiplied by varK. Used
            when evaluating the surrogate in GpEvalModel. The default is False.

        Returns
        -------
        Kern : 2D numpy array
            Kernel matrix.
        Kcor : 2D numpy array
            Correlation matrix.
        Kcov : 2D numpy array
            Covariance matrix.
        Kcov_chofac : scipy.linalg.cho_factor
            Cholesky factorization of Kcov.
        condK : float
            Condition number of either the regularized Kcov matrix, or in the 
            preconditioned case, the condition number of the regularized Kcor.
        '''
        
        ''' Check inputs ''' 
        
        theta     = hp_vals.theta 
        hp_kernel = hp_vals.kernel
        
        dim, n1, n2 = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes' 
        
        if varK is None:
            assert hp_vals.varK  is not None, f'varK is not provided and hp_vals.varK is None, hp_vals = {hp_vals}'
            varK = hp_vals.varK 
            
        assert np.sum(np.isnan(theta)) == 0, f'There are nan values theta = {theta}'
        
        if noise_vec is None:
            nugget    = self._etaK
            noise_vec = self.make_vec_var_noise(hp_vals, nugget, varK)
        else:
            assert noise_vec.size == self.n_data, f'Size of noise_vec is {noise_vec.size} but it should be {self.n_data}'
            assert np.min(noise_vec) >= (0.99 * self._etaK), \
                f'min(noise_vec) = {np.min(noise_vec):.2e}, which is smaller than etaK = {self._etaK:.2e}'
        
        if self.wellcond_mtd == 'precon':
            
            assert self.use_grad is True, 'self.wellcond_mtd should be None if use_grad is False'
            
            P1, P1inv   = self.calc_Kern_precon(self.n_eval, self.n_grad, theta)[:2]
            Kern        = self.calc_Kern(Rtensor, theta, hp_kernel, self.bvec_use_grad, self.bvec_use_grad)
            Kern_w_eta  = Kern + np.diag(noise_vec / varK)
            
            if b_normlz_w_varK is False:
                Kcov = varK * Kern_w_eta
            else:
                Kcov = 1.0 * Kern_w_eta
            
            Kcor        = P1inv @ Kern @ P1inv
            Kcor_w_eta  = P1inv @ Kern_w_eta @ P1inv
            
            # Apply 2nd preconditioning to handle case with noisy obj or grad
            # This ensures the diagonal is 1 + nugget
            p2vec = np.sqrt(np.diag(Kcor_w_eta) / (1 + nugget))
            P2inv = np.diag(1 / p2vec)
            Pall  = np.diag(p2vec) @ P1
            
            Kscld = P2inv @ Kcor_w_eta @ P2inv
            
            if calc_cond:
                condK = np.linalg.cond(Kscld, p = self.condnum_norm) 
                
                if condK > (1.1 * self.cond_max):
                    print(f'*** WARNING: condK = {condK:.2e} which is greater than cond_max = {self.cond_max:.2e}')
            else:
                condK = None
            
            time_chofac_start = time.time()
            
            if calc_chofac:
                try:
                    if b_normlz_w_varK:
                        Kscld_chofac = cho_factor(Kscld, lower = True)
                    else:
                        Kscld_chofac = cho_factor(varK * Kscld, lower = True)
                        
                    Kcov_chofac = (Pall @ Kscld_chofac[0], Kscld_chofac[1])
                    
                    # Kcor_chofac = cho_factor(Kcor_w_eta, lower = True)
                    # Kcov_chofac = (np.sqrt(sigK) * (P1 @ Kcor_chofac[0]), Kcor_chofac[1])
                except:
                    Kcov_chofac = None
                    
                    eig_val = np.linalg.eigvalsh(Kcor_w_eta)
                    min_eig = np.min(eig_val)
                    cond_L2 = np.max(eig_val) / min_eig
                    
                    log10th = np.log10(theta)
                    print(f'Failed Cho factorization of Kcor_w_eta, cond_L2 = {cond_L2:.2e}, eig min = {min_eig:.2e}, log10(th) = {log10th}')
            else:
                Kcov_chofac = None
                
        else:
            Kcor = None
            Kern = self.calc_Kern(Rtensor, theta, hp_kernel, self.bvec_use_grad, self.bvec_use_grad)
            
            if b_normlz_w_varK:
                Kcov = Kern + np.diag(noise_vec / varK)
            else:
                Kcov = varK * Kern + np.diag(noise_vec)
        
            if calc_cond:
                condK = np.linalg.cond(Kcov, p = self.condnum_norm)  
                
                if condK > self.cond_max_abs:
                    calc_chofac = False
            else:
                condK = None
        
            time_chofac_start = time.time()
        
            if calc_chofac:
                try:
                    Kcov_chofac = cho_factor(Kcov)
                except:
                    Kcov_chofac = None
                    
                    eig_val = np.linalg.eigvalsh(Kcov)
                    min_eig = np.min(eig_val)
                    cond    = np.max(eig_val) / min_eig
                    
                    log10th = np.log10(theta)
                    print(f'Failed Cho factorization of Kcov, cond = {cond:.2e}, eig min = {min_eig:.2e}, log10(th) = {log10th}')
            else:
                Kcov_chofac = None
                
        time_chofac = time.time() - time_chofac_start
        self._time_chofac += time_chofac
        
        return Kern, Kcor, Kcov, Kcov_chofac, condK

    def make_vec_var_noise(self, hp_vals, nugget, varK):
        '''
        Parameters
        ----------
        hp_vals : HparaOptzVal from GpHpara
            Class that contains all of the hyperparameters
        nugget : float
            Minimum value to be added to the diagonal of the kernel matrix to 
            regularize it.
        varK : float
            Hyperparameter.

        Returns
        -------
        var_noise : 1D numpy array
            Array of noise values.
        '''
        
        ''' Preliminary '''
        
        assert not np.isnan(varK), f'We have varK = {varK}'
        
        theta = hp_vals.theta 
        std_fval, _, std_fgrad = self.get_scl_eval_data(theta)[1:]
        
        if self.use_grad == False:
            var_noise = np.maximum(std_fval, nugget * varK)
        else:
            var_noise = np.zeros(self.n_data)
            
            ''' Make noise vector '''
            
            if self.known_eps_fval:
                assert hp_vals.var_fval is None
                assert not np.any(np.isnan(std_fval)), f'There are nan values: std_fval = {std_fval}'
                var_noise[:self.n_eval] = std_fval**2
            else:
                var_noise[:self.n_eval] = hp_vals.var_fval
            
            if self.known_eps_fgrad:
                assert hp_vals.var_fgrad is None
                assert not np.any(np.isnan(std_fgrad)), f'There are nan values: std_fgrad = {std_fgrad}'
                var_noise[self.n_eval:] = (std_fgrad**2).reshape(std_fgrad.size, order='f')
            else:
                var_noise[self.n_eval:] = hp_vals.var_fgrad
    
            ''' Make nugget vector '''
            
            if self.wellcond_mtd == 'precon':
                pvec       = self.calc_Kern_precon(self.n_eval, self.n_grad, theta, b_return_vec = True)[0]
                nugget_vec = pvec**2 * nugget * varK
            else:
                nugget_vec = np.full(self.n_data, nugget * varK)
            
            # Scale and set minimum noise with the nugget
            var_noise = np.maximum(var_noise, nugget_vec)
            
            if np.any(np.isnan(var_noise)):
                print(f'There are nan in var_noise. varK = {varK:.1e}, theta = {theta}')
                print(f'var_noise = \n{var_noise}')
        
        return var_noise
