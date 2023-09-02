#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:52:38 2022

@author: andremarchildon
"""

import unittest
import numpy as np

from gpgradpy.base import CommonFun
from gpgradpy import GaussianProcess

''' Set parameters '''

np.set_printoptions(precision=2)

list_kernel_types  = ['SqExp', 'Ma5f2', 'RatQu']
list_wellcond_mtds = [None, 'req_vmin', 'precon']

# list_kernel_types  = ['SqExp']
# list_wellcond_mtds = [None]

x_eval      = np.array([[0,0], [1,1]], dtype=float)
n_eval, dim = x_eval.shape

eps         = 1e-6
use_grad    = True
theta       = 1 * np.linspace(1, 2, dim)

# Testing tolerances
rtol    = 1e-4
atol    = 1e-4 # Lower than usual since second derivatives are compared

print_results = False

Rtensor = CommonFun.calc_Rtensor(x_eval, x_eval, 1)

''' Setup GP '''

def calc_Kern_mat(kernel_type, wellcond_method):

    GP_base = GaussianProcess(dim, use_grad, kernel_type, wellcond_method)
    
    fh_calc_KernBase = GP_base.calc_KernBase
    fh_calc_KernGrad = GP_base.calc_KernGrad
        
    if kernel_type == 'SqExp' or kernel_type == 'Ma5f2':
        hp_kernel = np.nan 
    elif kernel_type == 'RatQu':
        hp_kernel = 1.5
    else:
        raise Exception('Unknown kernel type')
    
    Kbase   = fh_calc_KernBase(Rtensor, theta, hp_kernel)
    Kfull   = fh_calc_KernGrad(Rtensor, theta, hp_kernel)
    
    Kfull_fd = np.zeros(Kfull.shape)
    Kfull_fd[:n_eval, :n_eval] = Kbase
    
    # Calc first derivative
    for i in range(1, dim+1):
        r1 = i * n_eval
        r2 = r1 + n_eval
        
        x1 = x_eval * 1
        x1[:,i-1] += eps
        
        Rtensor_mod = CommonFun.calc_Rtensor(x1, x_eval, 1)
        Kbase_mod   = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
        Kfull_fd[r1:r2, :n_eval] = (Kbase_mod - Kbase) / eps
        
        Rtensor_mod = CommonFun.calc_Rtensor(x_eval, x1, 1)
        Kbase_mod   = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
        Kfull_fd[:n_eval, r1:r2] = (Kbase_mod - Kbase) / eps
    
    for i in range(1, dim+1):
        r1 = i * n_eval
        r2 = r1 + n_eval
        
        x1m = x_eval * 1
        x1m[:,i-1] -= eps
        
        x1p = x_eval * 1
        x1p[:,i-1] += eps
        for j in range(1, dim+1):
            c1 = j * n_eval
            c2 = c1 + n_eval
            
            x2m = x_eval * 1
            x2m[:,j-1] -= eps
            
            x2p = x_eval * 1
            x2p[:,j-1] += eps
            
            Rtensor_mod = CommonFun.calc_Rtensor(x1m, x2m, 1)
            Kbase_mm    = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
            
            Rtensor_mod = CommonFun.calc_Rtensor(x1m, x2p, 1)
            Kbase_mp    = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
            
            Rtensor_mod = CommonFun.calc_Rtensor(x1p, x2m, 1)
            Kbase_pm    = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
            
            Rtensor_mod = CommonFun.calc_Rtensor(x1p, x2p, 1)
            Kbase_pp    = fh_calc_KernBase(Rtensor_mod, theta, hp_kernel)
            
            Kfull_fd[r1:r2, c1:c2] = (Kbase_pp + Kbase_mm - Kbase_pm - Kbase_mp) / (4*eps**2)
            
    Kfull_diff = Kfull - Kfull_fd
    max_diff   = np.max(np.abs(Kfull_diff))
    
    if print_results:
        print(f'kernel_type = {kernel_type}, wellcond_method = {wellcond_method}')
        print(f'max_diff = {max_diff:.3e}')
            
    return Kfull, Kfull_fd

class TestKernelMatrix(unittest.TestCase):
    
    def test_(self):
        
        for kernel_type in list_kernel_types:
            for wellcond_mtd in list_wellcond_mtds:
        
                Kfull, Kfull_fd = calc_Kern_mat(kernel_type, wellcond_mtd)
        
                np.testing.assert_allclose(Kfull, Kfull_fd, rtol = rtol, atol = atol)
        
if __name__ == '__main__':
    unittest.main()
