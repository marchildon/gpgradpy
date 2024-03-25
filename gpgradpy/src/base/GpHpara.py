#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:52:48 2021

@author: andremarchildon
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds
from smt.sampling_methods import LHS


@dataclass
class HparaOptzVal:
    beta:       np.array = None # Coefficients for the mean function
    theta:      np.array = None # Hyperparameters related to the characteristic length for the kernel
    kernel:     float    = None # Extra hyperparameter for the kernel
    varK:       float    = None # Variance of the stationary residual error
    var_fval:   float    = None # Variance of the noise on the value of the function of interest
    var_fgrad:  float    = None # Variance of the noise on the gradient of the function of interest
    
class GpHpara:
    
    # Keeps track of the last point where the acq value, gradient and Hessian 
    # have been calculated. This ensures these parameters do not need to be 
    # calculated again
    _last_hp_vec = None

    def make_hp_class(self, beta = None, theta = None, kernel = None, 
                      varK = None, var_fval = None, var_fgrad = None):
        
        return HparaOptzVal(beta, theta, kernel, varK, var_fval, var_fgrad)
    
    def set_hp_from_idx(self, i_optz):
        
        if np.isnan(self.hp_var_fval_all[i_optz]):
            self.known_eps_fval = True
            hp_var_fval = None
        else:
            self.known_eps_fval = False
            hp_var_fval = self.hp_var_fval_all[i_optz]
            
        if np.isnan(self.hp_var_fgrad_all[i_optz]):
            self.known_eps_fgrad = True
            hp_var_fgrad = None
        else:
            self.known_eps_fgrad = False
            hp_var_fgrad = self.hp_var_fgrad_all[i_optz]
        
        hp_vals = self.make_hp_class(
            self.hp_beta_all[i_optz,:], self.hp_theta_all[i_optz,:], 
            self.hp_kernel_all[i_optz], self.hp_varK_all[i_optz], 
            hp_var_fval,                hp_var_fgrad)
        
        self.hp_vals = hp_vals
        
    def hp_vec2dataclass(self, hp_optz_info, hp_vec):
        '''
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from the file GpHparaOptz
            Holds info on what hyperparameters to optimize numerically.
        hp_vec : 1D numpy array
            Array with values of the hyperparameters being optimized numerically.

        Returns
        -------
        hp_vals : data class of type HparaOptzVal
            Holds the hyperparameters in a data class.
        '''
        
        hp_vec_mod       = np.copy(hp_vec)
        bvec             = hp_optz_info.bvec_log_optz
        hp_vec_mod[bvec] = 10**(hp_vec_mod[bvec])
        
        if hp_optz_info.has_theta:
            theta = hp_vec_mod[hp_optz_info.idx_theta]
        else:
            theta = None
        
        if hp_optz_info.has_kernel:
            hp_kernel = hp_vec_mod[hp_optz_info.idx_kernel]
        else:
            hp_kernel = None
        
        if hp_optz_info.has_varK:
            assert self.b_has_noisy_data
            varK = hp_vec_mod[hp_optz_info.idx_varK]
        else:
            varK = None
            
        if hp_optz_info.has_var_fval:
            assert self.b_optz_var_fval
            var_fval = hp_vec_mod[hp_optz_info.idx_var_fval]
        else:
            var_fval = None

        if hp_optz_info.has_var_fgrad:
            assert self.b_optz_var_fgrad
            var_fgrad = hp_vec_mod[hp_optz_info.idx_var_fgrad]
        else:
            var_fgrad = None
        
        return self.make_hp_class(None, theta, hp_kernel, varK, var_fval, var_fgrad)

    def set_custom_hp(self, beta = None, theta = None, kernel = None, varK = None, 
                      var_fval = None, var_fgrad = None):
        
        ''' 
        See definitions of inputs from the data class HparaOptzVal
        '''
        
        if varK is not None:
            assert varK > 0, f'varK must be positive but it is {varK}'
        
        hp_vals      = self.make_hp_class(beta, theta, kernel, varK, var_fval, var_fgrad)
        self.hp_vals = hp_vals
