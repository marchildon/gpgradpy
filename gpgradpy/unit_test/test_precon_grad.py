#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:19:56 2023

@author: andremarchildon
"""

import unittest
import numpy as np

from gpgradpy import GaussianProcess

'''  
Check the gradient for the precondition matrix with respect to theta
'''

# np.set_printoptions(precision=2)

''' Set parameters '''

# Testing tolerances
rtol    = 1e-4
atol    = 1e-8

# Default parameters
eps     = 1e-6
dim     = 2
n_eval  = 1
n_data  = n_eval * (dim+1)
theta   = np.linspace(2.5, 3, dim)

# Not to be changed
use_grad        = True
wellcond_method = 'precon' 

print_results   = False

def check_precon_grad(kernel_type, b_return_vec):

    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_method)
    pvec, pvec_inv, grad_precon = GP.calc_Kern_precon(n_eval, n_eval, theta, 
                                                      calc_grad = True, b_return_vec = b_return_vec)
    
    if b_return_vec:
        fd_grad = np.zeros((n_data, dim))
    else:
        fd_grad = np.zeros((dim, n_data, n_data))
    
    for i in range(dim):
        
        theta_w_eps = theta * 1.0 
        theta_w_eps[i] += eps 
        
        pvec_w_eps = GP.calc_Kern_precon(n_eval, n_eval, theta_w_eps, calc_grad = False, b_return_vec = b_return_vec)[0]
        
        if b_return_vec:
            fd_grad[:,i]   = (pvec_w_eps - pvec) / eps
        else:
            fd_grad[i,:,:] = (pvec_w_eps - pvec) / eps
        
    diff_grad_precond = grad_precon - fd_grad
    max_abs_diff = np.max(np.abs(diff_grad_precond))
    
    if print_results:
        print(f'kernel: = {kernel_type}, b_return_vec = {b_return_vec}')
        print(f'  max_abs_diff = {max_abs_diff:.2e}')
    
    return grad_precon, fd_grad

class TestGradKmat(unittest.TestCase):
    
    def test_sq_exp_kernel(self):
        kernel_type = 'SqExp'
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = True)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = False)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
    def test_matern_ff2_kernel(self):
        kernel_type = 'Ma5f2'
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = True)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = False)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
    def test_RatQu_kernel(self):
        kernel_type = 'RatQu'
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = True)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
        grad_precon, fd_grad = check_precon_grad(kernel_type, b_return_vec = False)
        np.testing.assert_allclose(grad_precon, fd_grad, rtol = rtol, atol = atol)  
        
if __name__ == '__main__':
    unittest.main()
    