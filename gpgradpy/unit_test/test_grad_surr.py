#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:40:59 2021

@author: andremarchildon
"""

import unittest
import numpy as np

from gpgradpy.src import GaussianProcess

''' Set parameters '''

eps      = 1e-4
use_grad = True

xmin = 0 
xmax = 2

kernel_type_vec  = ['SqExp', 'RatQu', 'Ma5f2']
wellcond_mtd_vec = [None, 'req_vmin', 'precon']

method2set_hp = 'set'

calc_hess = True

# Testing tolerances
rtol    = 1e-4
atol    = 1e-8

print_results = False

''' Set nodal parameters and evaluate the function at those points '''

node_option = 1

if node_option == 1:
    x_eval      = np.array([xmin, xmax], dtype=float)
    x_eval      = x_eval[:,None]

elif node_option == 2:
    n = 5
    x_eval      = np.zeros((n, 2))
    x_eval[:,0] = np.linspace(xmin, xmax, n)
    x_eval[:,1] = x_eval[:,0]
elif node_option == 3:
    x_eval      = 3*np.array([[0, 0], [1, 0], [0, 1], [1,1]])
    x_eval      = 3*np.array([[0, 0], [1, 0]])
elif node_option == 4:
    dim  = 3
    dist = 1
    x_eval      = np.zeros((2, dim))
    x_eval[0,:] = np.linspace(xmin, xmax-dist, dim)
    x_eval[1,:] = x_eval[0,:] + dist
else:
    raise Exception('Option not available')

frac = 0.4 

diff_eval = x_eval[1,:] - x_eval[0,:]
x2test    = x_eval[0,:] + frac * diff_eval
x2test    = x2test[None,:]

n_x_model = x2test.shape[0]

n, dim    = x_eval.shape
n_der     = dim

hp_theta = 0.1 * np.linspace(1, 2, dim)


def call_fun(x_in, calc_grad):

    n_in, dim_in = x_in.shape
    fun_out      = np.sum(x_in**2, axis=1)

    if calc_grad:
        grad_out = np.zeros((n_in, dim_in))

        for d in range(dim_in):
            grad_out[:,d] = 2*x_in[:,d]
    else:
        grad_out = None

    return fun_out, grad_out

obj, grad = call_fun(x_eval, use_grad)
std_obj   = np.zeros(obj.shape)

if use_grad:
    std_grad = np.zeros(x_eval.shape)
else:
    std_grad = None

''' Calculate Kriging parameters '''

i_optz = 0

def make_GP(wellcond_mtd, kernel_type):
    GP = GaussianProcess(dim, use_grad=use_grad, kernel_type=kernel_type, wellcond_mtd=wellcond_mtd)
    GP.init_optz_surr(2)

    if kernel_type == 'SqExp' or kernel_type == 'Ma5f2':
        hp_kernel = np.nan 
    elif kernel_type == 'RatQu':
        hp_kernel = 1.5
    else:
        raise Exception('Unknown kernel type')

    if method2set_hp == 'set':
        mean_obj = np.atleast_1d(np.mean(obj))
        
        hp_vals  = GP.make_hp_class(beta = mean_obj, kernel = hp_kernel,
                                    theta = hp_theta, varK = 1e3)
    else:
        hp_vals = None
    
    if use_grad:
        GP.set_data(x_eval, obj, std_obj, grad, std_grad)
    else:
        GP.set_data(x_eval, obj, std_obj)
    
    GP.set_hpara(method2set_hp, i_optz, hp_vals = hp_vals)

    GP.setup_eval_model()
    
    return GP

def eval_GP(GP, x2test):

    mu_cent, sig_cent, dmu_dx_cent, dsig_dx_cent, d2mu_dx2, d2sig_dx2 \
        = GP.eval_model(x2test, calc_grad=True, calc_hess=calc_hess)
    
    dmu_dx_cent  = dmu_dx_cent[0,:]
    dsig_dx_cent = dsig_dx_cent[0,:]
    
    fd_dmu_dx    = np.full((dim), np.nan)
    fd_dsig_dx   = np.full((dim), np.nan)
    
    fd_d2mu_dx2  = np.full((dim, dim), np.nan)
    fd_d2sig_dx2 = np.full((dim, dim), np.nan)
    
    for i in range(dim):
        for j in range(i,dim):
            
            # Evaluate at ++
            x_model_pp       = x2test * 1.0
            x_model_pp[0,i] += eps
            x_model_pp[0,j] += eps
            mu_pp, sig_pp    = GP.eval_model(x_model_pp)[:2]
            
            # Evaluate at +-
            x_model_pm       = x2test * 1.0
            x_model_pm[0,i] += eps
            x_model_pm[0,j] -= eps
            mu_pm, sig_pm    = GP.eval_model(x_model_pm)[:2]
            
            # Evaluate at -+
            x_model_mp       = x2test * 1.0
            x_model_mp[0,i] -= eps
            x_model_mp[0,j] += eps
            mu_mp, sig_mp    = GP.eval_model(x_model_mp)[:2]
            
            # Evaluate at --
            x_model_mm       = x2test * 1.0
            x_model_mm[0,i] -= eps
            x_model_mm[0,j] -= eps
            mu_mm, sig_mm    = GP.eval_model(x_model_mm)[:2]
            
            fd_d2mu_dx2[i,j]  = (mu_pp  - mu_pm  - mu_mp  + mu_mm) / (4 * eps**2)
            fd_d2mu_dx2[j,i]  = fd_d2mu_dx2[i,j]
            
            fd_d2sig_dx2[i,j] = (sig_pp - sig_pm - sig_mp + sig_mm) / (4 * eps**2)
            fd_d2sig_dx2[j,i] = fd_d2sig_dx2[i,j]
            
            if i == j:            
                fd_dmu_dx[i]  = (mu_pp  - mu_mm)  / (2*(2*eps))
                fd_dsig_dx[i] = (sig_pp - sig_mm) / (2*(2*eps))
                
    return fd_dmu_dx, fd_dsig_dx, fd_d2mu_dx2, fd_d2sig_dx2

class TestGrad(unittest.TestCase):
    
    def test_GP_grad(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            for kernel_type in kernel_type_vec:
                GP = make_GP(wellcond_mtd, kernel_type)
                
                dmu_dx, dsig_dx = GP.eval_model(x2test, calc_grad=True, calc_hess=False)[2:4]
                dmu_dx  = dmu_dx[0,:]
                dsig_dx = dsig_dx[0,:]
                
                fd_dmu_dx, fd_dsig_dx, fd_d2mu_dx2, fd_d2sig_dx2 = eval_GP(GP, x2test)
                
                if print_results: 
                    diff_dmu_dx  = dmu_dx  - fd_dmu_dx
                    diff_dsig_dx = dsig_dx - fd_dsig_dx
                    
                    print(f'\n** wellcond_mtd = {wellcond_mtd}, kernel_type = {kernel_type} **')
                    print('Gradient model mean')
                    print(f' analytical = {dmu_dx}')
                    print(f' FD         = {fd_dmu_dx}')
                    print(f' diff       = {diff_dmu_dx}')
                    
                    print('\nGradient model std')
                    print(f' analytical = {dsig_dx}')
                    print(f' FD         = {fd_dsig_dx}')
                    print(f' diff       = {diff_dsig_dx}')
                
                np.testing.assert_allclose(dmu_dx,  fd_dmu_dx,  rtol = rtol, atol = atol)
                np.testing.assert_allclose(dsig_dx, fd_dsig_dx, rtol = rtol, atol = atol)
                
    def test_GP_hess(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            for kernel_type in kernel_type_vec:
                GP = make_GP(wellcond_mtd, kernel_type)
                
                d2mu_dx2, d2sig_dx2 = GP.eval_model(x2test, calc_grad=True, calc_hess=True)[4:]
                d2mu_dx2  = d2mu_dx2[0,:,:]
                d2sig_dx2 = d2sig_dx2[0,:,:]
                
                fd_d2mu_dx2, fd_d2sig_dx2 = eval_GP(GP, x2test)[2:]
                
                if print_results:
                    diff_d2mu_dx2  = d2mu_dx2 - fd_d2mu_dx2
                    diff_d2sig_dx2 = d2sig_dx2 - fd_d2sig_dx2
                    
                    print(f'\n** wellcond_mtd = {wellcond_mtd}, kernel_type = {kernel_type} **')
                    print('\nHessian model mean')
                    print(f' analytical = \n{d2mu_dx2}')
                    print(f' FD         = \n{fd_d2mu_dx2}')
                    print(f' diff       = \n{diff_d2mu_dx2}')
                    
                    print('\nHessian model std')
                    print(f' analytical = \n{d2sig_dx2}')
                    print(f' FD         = \n{fd_d2sig_dx2}')
                    print(f' diff       = \n{diff_d2sig_dx2}')
                
                np.testing.assert_allclose(d2mu_dx2,  fd_d2mu_dx2,  rtol = rtol, atol = atol)
                np.testing.assert_allclose(d2sig_dx2, fd_d2sig_dx2, rtol = rtol, atol = atol)
        
if __name__ == '__main__':
    unittest.main()
