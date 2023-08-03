#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:39:17 2022

@author: andremarchildon
"""

import numpy as np
from scipy import linalg

from dataclasses import dataclass

@dataclass
class LkdInfo:  
    hp_beta:          np.array = None # Coefficents of the mean function
    hp_beta_grad:     np.array = None
    hp_varK:          float    = None # Hyperparameter varK
    hp_varK_grad:     np.array = None
    ln_det_Kmat:      float    = None # Log determinant of either the kernel or covariance matrix
    ln_det_Kmat_grad: np.array = None
    ln_lkd:           float    = None # Marginal log-likelihood
    ln_lkd_grad:      np.array = None
    cond:             float    = None # Condition number of either the kernel or correlation matrix 
    cond_grad:        np.array = None
    data_vec:         np.array = None # Function and gradient evaluations in a 1D numpy array

class calcLkdWoNoise:
    
    def calc_lkd_all_wo_noise(self, theta, Kern_chofac, KernGrad_hp = None, calc_lkd = True):    
        '''
        Parameters
        ----------
        theta : 1d numpy array of floats of length dim
            Hyperparameters for the kernel related to the characteristic length.
        Kern_chofac : scipy.linalg.cho_factor
            Cholesky factorization of the regularized kernel matrix.
        KernGrad_hp : 3d numpy array of size [dim, n_eval, n_eval], optional
            The derivative of the kernel matrix wrt the hyperparameters. The default is None.
        calc_lkd : bool, optional
            Set to True to calculate the marginal log-likelihood. The default is True.

        Returns
        -------
        class of type LkdInfo
        '''
        
        fval_scl, std_fval, grad_scl, std_fgrad = self.get_scl_eval_data(theta)

        ''' Calculate individual terms '''
        
        data_vec = self.make_data_vec(fval_scl, grad_scl)
        
        mean_model_val, mean_model_hp_grad, hp_beta, hp_beta_grad \
            = self.calc_model_max_lkd(Kern_chofac, data_vec, KernGrad_hp)
        
        hp_varK, hp_varK_grad \
            = self.calc_lkd_opt_varK(data_vec,     mean_model_val,
                                     Kern_chofac,  mean_model_hp_grad,
                                     KernGrad_hp)
        
        ''' Calculate likelihood '''
        
        if calc_lkd:
            ln_det_Kern, ln_det_KernGrad = self.calc_detKmat(Kern_chofac, KernGrad_hp)
    
            ln_lkd, ln_lkd_grad \
                = self.calc_lkd_w_Kern(self.n_data, hp_varK, theta, ln_det_Kern, 
                                       hp_varK_grad, ln_det_KernGrad)
        else:
            ln_det_Kern = ln_det_KernGrad = None
            ln_lkd      = ln_lkd_grad     = None

        ''' Return calculations '''
        
        return LkdInfo(hp_beta     = hp_beta,     hp_beta_grad     = hp_beta_grad, 
                       hp_varK     = hp_varK,     hp_varK_grad     = hp_varK_grad,
                       ln_det_Kmat = ln_det_Kern, ln_det_Kmat_grad = ln_det_KernGrad, 
                       ln_lkd      = ln_lkd,      ln_lkd_grad      = ln_lkd_grad)
    
    @staticmethod
    def calc_lkd_w_Kern(n_data, hp_varK, theta, ln_det_Kern, 
                        hp_varK_grad = None, ln_det_KernGrad = None):
        
        ln_lkd = -(n_data * np.log(hp_varK) + ln_det_Kern) /2
        
        if (hp_varK_grad is None) or (ln_det_KernGrad is None):
            ln_lkd_grad = None
        else:
            ln_lkd_grad = -(n_data * (hp_varK_grad / hp_varK) + ln_det_KernGrad) /2
        
        return ln_lkd, ln_lkd_grad
    
    @staticmethod
    def calc_lkd_opt_varK(data_vec, mean_model_val, Kern_chofac,  
                          mean_model_hp_grad = None, KernGrad_hp = None, varK_min = 1e-8):
        
        n_data      = data_vec.size
        denominator = n_data # For unbiased varK we would require denominator(n_data - self.beta_var_npara)
        
        res         = data_vec - mean_model_val 
        invKern_res = linalg.cho_solve(Kern_chofac, res)
        
        # Have varK_min to ensure varK is positive
        varK = np.max((varK_min, np.dot(res, invKern_res) / denominator))
        
        if (mean_model_hp_grad is None) or (KernGrad_hp is None):
            varK_grad = None
        else:
            varK_grad = (-2 * invKern_res.T @ mean_model_hp_grad 
                         - np.einsum('i,kij,j', invKern_res, KernGrad_hp, invKern_res)) / denominator
            
        return varK, varK_grad
    
class calcLkdWNoise:   
        
    def calc_lkd_all_w_noise(self, theta, Kcov_chofac, Kcov_grad_hp = None, calc_lkd = True):    
        '''
        Parameters
        ----------
        theta : 1d numpy array of floats of length dim
            Hyperparameters for the kernel related to the characteristic length.
        Kcov_chofac : scipy.linalg.cho_factor
            Cholesky factorization of the covariance matrix.
        Kcov_grad_hp : 3d numpy array of size [dim, n_eval, n_eval], optional
            The derivative of the covariance matrix wrt the hyperparameters. The default is None.
        calc_lkd : bool, optional
            Set to True to calculate the marginal log-likelihood. The default is True.

        Returns
        -------
        class of type LkdInfo
        '''
        
        fval_scl, std_fval, grad_scl, std_fgrad = self.get_scl_eval_data(theta)
        
        ''' Calculate individual terms '''
        
        data_vec = self.make_data_vec(fval_scl, grad_scl)
        
        mean_model_val, mean_model_hp_grad, hp_beta, hp_beta_grad \
            = self.calc_model_max_lkd(Kcov_chofac, data_vec, Kcov_grad_hp)
        
        # Calculate data_diff, ie the diff between the mean function and the data
        data_diff          = data_vec - mean_model_val
        sol_King_data_diff = sol_King_data_diff = linalg.cho_solve(Kcov_chofac, data_diff)
        
        ''' Calculate likelihood '''
        
        if calc_lkd:
            ln_det_Kcov, ln_det_Kcov_grad = self.calc_detKmat(Kcov_chofac, Kcov_grad_hp)
    
            ln_lkd, ln_lkd_grad \
                = self.calc_lkd_w_Kcov(data_diff, sol_King_data_diff, ln_det_Kcov, theta,
                                       Kcov_grad_hp, ln_det_Kcov_grad, mean_model_hp_grad)
        else:
            ln_det_Kcov = ln_det_Kcov_grad = None
            ln_lkd      = ln_lkd_grad      = None

        ''' Return calculations '''
        
        return LkdInfo(hp_beta     = hp_beta,     hp_beta_grad     = hp_beta_grad, 
                       ln_det_Kmat = ln_det_Kcov, ln_det_Kmat_grad = ln_det_Kcov_grad, 
                       ln_lkd      = ln_lkd,      ln_lkd_grad      = ln_lkd_grad, 
                       data_vec    = data_vec)
    
    @staticmethod
    def calc_lkd_w_Kcov(data_diff, sol_King_data_diff, ln_det_Kcov, theta,
                        Kcov_grad_hp = None, ln_det_Kcov_grad = None, mean_model_hp_grad = None):
        
        ln_lkd = -(ln_det_Kcov + np.dot(data_diff, sol_King_data_diff)) /2
        
        if (Kcov_grad_hp is None) or (ln_det_Kcov_grad is None) or (mean_model_hp_grad is None):
            ln_lkd_grad = None
        else:
            ln_lkd_grad = (-0.5 * ln_det_Kcov_grad 
                           + 0.5*np.einsum('i,kij,j', sol_King_data_diff, Kcov_grad_hp, sol_King_data_diff) 
                           + sol_King_data_diff @ mean_model_hp_grad)
        
        return ln_lkd, ln_lkd_grad

class CalcLkd(calcLkdWNoise, calcLkdWoNoise):
       
    def calc_lkd_all(self, hp_vals, calc_lkd = True, calc_cond = False, calc_grad = False):
        '''
        Parameters
        ----------
        hp_vals : HparaOptzVal from class GpHpara
            class with the hyperparameters.
        calc_lkd : bool, optional
            Set to True to calculate the marginal log-likelihood. The default is True.
        calc_cond : bool, optional
            Set to True to calculate the condition number. The default is False.
        calc_grad : bool, optional
            Set to True to calculate the gradients of the lkd and cond. The default is False.

        Returns
        -------
        lkd_info : class of type LkdInfo
            Holds values and gradients that were calculated 
        b_chofac_good : bool
            True if the Cholesky factorization was successful, False, otherwise.
        '''
        
        x_scl, Rtensor = self.get_scl_x_w_dist()
        # fval_scl, std_fval, grad_scl, std_fgrad = self.get_scl_eval_data(hp_vals.theta)
        
        # If some of the data is noisy then the hyperparameter varK must be 
        # calculated numerically
        if self.b_has_noisy_data:
            Kern, _, Kcov, Kcov_chofac, cond_K \
                = self.calc_all_K_w_chofac(Rtensor, hp_vals, calc_cond = calc_cond)
            
            if calc_grad:
                Kcov_grad_hp = self.calc_Kcov_grad_hp(self.hp_info_optz_lkd, hp_vals, Kern, Rtensor)
            else:
                Kcov_grad_hp = None
            
            if Kcov_chofac is None:
                b_chofac_good   = False
                cond, cond_grad = self.calc_cond_L2_w_grad(Kcov, Kcov_grad_hp)
                lkd_info        = LkdInfo(cond = cond, cond_grad = cond_grad)
            else:
                b_chofac_good   = True
                lkd_info        = self.calc_lkd_all_w_noise(hp_vals.theta, Kcov_chofac, 
                                                            Kcov_grad_hp)
            
            if calc_cond and calc_grad:
                lkd_info.cond, lkd_info.cond_grad = self.calc_cond_w_grad(Kcov, Kcov_chofac, Kcov_grad_hp)
            else:
                lkd_info.cond = cond_K
        else:
            Kern, _, Kern_w_eta, Kern_chofac, cond_K \
                = self.calc_Kern_w_chofac(Rtensor, hp_vals, calc_cond = calc_cond)
            
            if calc_grad:
                KernGrad_hp = self.calc_KernGrad_hp(self.hp_info_optz_lkd, hp_vals, Rtensor)
            else:
                KernGrad_hp = None
            
            if Kern_chofac is None:
                b_chofac_good   = False
                cond, cond_grad = self.calc_cond_L2_w_grad(Kern_w_eta, KernGrad_hp)
                lkd_info        = LkdInfo(cond = cond, cond_grad = cond_grad)
            else:
                b_chofac_good   = True
                
                lkd_info = self.calc_lkd_all_wo_noise(hp_vals.theta, Kern_chofac, 
                                                      KernGrad_hp)
                
                if calc_cond and calc_grad:
                    lkd_info.cond, lkd_info.cond_grad = self.calc_cond_w_grad(Kern_w_eta, Kern_chofac, KernGrad_hp)
                else:
                    lkd_info.cond = cond_K
        
        return lkd_info, b_chofac_good
    
    @staticmethod
    def calc_detKmat(Kmat_chofac, Kmat_grad_hp=None):

        L = Kmat_chofac[0]
        
        # ln_det_Kmat = np.log(np.prod(np.diag(L)**2))
        ln_det_Kmat = 2*np.sum(np.log(np.diag(L))) # More accurate computationally

        if Kmat_grad_hp is None:
            ln_det_Kmat_grad_hp = None
        else:

            n_der = Kmat_grad_hp.shape[0]

            ln_det_Kmat_grad_hp = np.zeros(n_der)

            for i in range(n_der):
                ln_det_Kmat_grad_hp[i] = np.trace(linalg.cho_solve(Kmat_chofac, Kmat_grad_hp[i,:,:]))

        return ln_det_Kmat, ln_det_Kmat_grad_hp
   