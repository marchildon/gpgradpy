#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:11:00 2023

@author: andremarchildon
"""

import numpy as np

class KernelCommon:
    
    @staticmethod
    def calc_grad_precon_matrix(n_eval, n_grad, gamma_grad_theta, b_return_vec):
        '''
        Parameters
        ----------
        n_eval : int 
            No. of data points evaluated.
        gamma_grad_theta : 1d np array of length dim
            derivative of gamma with respect to theta
        b_return_vec : Bool
            If true, the preconditioner is returned as a 1d array, if false, 
            it is returned as a diagonal matrix.
        '''
        
        dim    = gamma_grad_theta.size
        n_data = n_eval + n_grad * dim
        
        if b_return_vec:
            pvec_grad_theta = np.zeros((n_data, dim))
            
            for i in range(dim):
                a = n_eval + i * n_grad
                b = a + n_grad
                pvec_grad_theta[a:b ,i] = gamma_grad_theta[i]
            
            return pvec_grad_theta
        else:
            P_grad_theta = np.zeros((dim, n_data, n_data))
            
            for i in range(dim):
                a = n_eval + i * n_grad
                b = a + n_grad
                vec                 = np.zeros(n_data)
                vec[a:b]            = gamma_grad_theta[i]
                P_grad_theta[i,:,:] = np.diag(vec)
                
            return P_grad_theta
