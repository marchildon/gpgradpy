#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:12:40 2022

@author: andremarchildon
"""

import unittest
import numpy as np

from gpgradpy.src import GaussianProcess

''' Set parameters '''

 # req_vmin is not setup for cases with noise on the objective or gradient 

# wellcond_mtd_vec = [None, 'req_vmin', 'precon']
wellcond_mtd_vec = [None, 'precon']
# wellcond_mtd_vec = ['req_vmin']

kernel_type_vec  = ['SqExp', 'Ma5f2', 'RatQu']

eps      = 1e-8
use_grad = True

# x_eval   = np.random.rand(10,1)
x_eval = 3*np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])

n_eval, dim  = x_eval.shape
n_der    = dim

eta_dflt = 1e-1

# Testing tolerances
rtol    = 1e-5
atol    = 1e-6

b_print_info  = False

bvec_use_grad     = np.ones(n_eval, dtype=bool)
bvec_use_grad[-1] = False

def get_hp_kernel(kernel_type):
    
    if (kernel_type == 'SqExp') or (kernel_type == 'Ma5f2'):
        hp_kernel     = np.nan
    elif kernel_type == 'RatQu':
        hp_kernel     = 2.5
    else:
        raise Exception('Unknown kernel')
        
    return hp_kernel

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

theta_vec       = np.linspace(1.5, 3, dim) 
varK_init       = 4
var_fval_init   = 3
var_fgrad_init  = 4

''' Calculate data '''

obj_eval, grad_eval_all = call_fun(x_eval, use_grad)
grad_eval = grad_eval_all[bvec_use_grad,:]

def make_hp_vals_wo_noise(GP_w0_noise, hp_kernel):
    
    return GP_w0_noise.make_hp_class(None, np.copy(theta_vec), hp_kernel, 
                                     None, None, None)

def make_hp_vals_w_noise(GP_w_noise, hp_kernel):
    
    return GP_w_noise.make_hp_class(None, np.copy(theta_vec), hp_kernel, 
                                    varK_init, var_fval_init, var_fgrad_init)

def setup_GP_wo_noise(wellcond_mtd, kernel_type):
    
    # hp_kernel   = get_hp_kernel(kernel_type)
    std_fval    = np.zeros(obj_eval.shape)
    std_fgrad   = np.zeros(grad_eval.shape)
    GP_w0_noise = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP_w0_noise.set_data(x_eval, obj_eval, std_fval, grad_eval, std_fgrad, bvec_use_grad)
    
    GP_w0_noise._etaK = eta_dflt
    
    return GP_w0_noise

def setup_GP_w_noise(wellcond_mtd, kernel_type):
    
    std_fval   = std_fgrad = None
    GP_w_noise = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP_w_noise.set_data(x_eval, obj_eval, std_fval, grad_eval, std_fgrad, bvec_use_grad)
    
    GP_w_noise._etaK = eta_dflt
    
    return GP_w_noise

def calc_GP_lkd_w_grad(GP, hp_vals1):
    
    if GP.wellcond_mtd == 'precon':
        calc_cond = False
    else:
        calc_cond = True
        
    lkd_info        = GP.calc_lkd_all(hp_vals1, calc_cond = calc_cond, calc_grad = True)[0]

    lkd             = lkd_info.ln_lkd
    lkd_der         = lkd_info.ln_lkd_grad

    ln_det_Kmat     = lkd_info.ln_det_Kmat
    ln_det_Kmat_der = lkd_info.ln_det_Kmat_grad

    hp_beta         = lkd_info.hp_beta
    hp_beta_der     = lkd_info.hp_beta_grad

    cond            = lkd_info.cond
    cond_der        = lkd_info.cond_grad
    
    return (lkd, ln_det_Kmat, hp_beta, cond), (lkd_der, ln_det_Kmat_der, hp_beta_der, cond_der)

''' Check finite difference and analytical gradients '''

def calc_fd_lkd(GP_in, wellcond_mtd, kernel_type, hp_vals_in, idx_in, text2print, b_noise_free):
    
    lkd_info_in = GP_in.calc_lkd_all(hp_vals_in, calc_cond = True, calc_grad = False)[0]
        
    if b_noise_free:
        hp_kernel        = get_hp_kernel(kernel_type)
        GP_wo_noise      = setup_GP_wo_noise(wellcond_mtd, kernel_type)
        hp_vals_wo_noise = make_hp_vals_wo_noise(GP_wo_noise, hp_kernel)
        
        ((lkd, ln_det_Kmat, hp_beta, cond), 
         (lkd_der, ln_det_Kmat_der, hp_beta_der, cond_der)) \
            = calc_GP_lkd_w_grad(GP_wo_noise, hp_vals_wo_noise)
    else:
        hp_kernel       = get_hp_kernel(kernel_type)
        GP_w_noise      = setup_GP_w_noise(wellcond_mtd, kernel_type)
        hp_vals_w_noise = make_hp_vals_w_noise(GP_w_noise, hp_kernel)

        ((lkd, ln_det_Kmat, hp_beta, cond), 
         (lkd_der, ln_det_Kmat_der, hp_beta_der, cond_der)) \
            = calc_GP_lkd_w_grad(GP_w_noise, hp_vals_w_noise)
    
    if hp_beta_der.size > 0:
        beta_fd          = (lkd_info_in.hp_beta - hp_beta) / eps 
        beta_der_diff    = hp_beta_der[0,idx_in] - beta_fd[0]
        np.testing.assert_allclose(hp_beta_der[0,idx_in], beta_fd[0], rtol = rtol, atol = atol)
        
    
    ln_detKmat_fd       = (lkd_info_in.ln_det_Kmat - ln_det_Kmat) / eps
    ln_detKmat_der_diff = ln_det_Kmat_der[idx_in] - ln_detKmat_fd
    ln_detKmat_frac     = np.abs(ln_det_Kmat_der[idx_in]) / np.max((1e-16, ln_detKmat_fd))
    
    lkd_fd           = (lkd_info_in.ln_lkd - lkd) / eps
    lkd_der_diff     = lkd_der[idx_in] - lkd_fd
    lkd_der_frac     = np.abs(lkd_der[idx_in]) / np.max((1e-16, lkd_fd))
    
    if wellcond_mtd != 'precon':
        calc_cond    = True
        cond_fd      = (lkd_info_in.cond - cond) / eps 
        cond_fd_diff = cond_der[idx_in] - cond_fd
        cond_fd_frac = np.abs(cond_der[idx_in]) / np.max((1e-16, np.abs(cond_fd)))
    else:
        calc_cond = False
        
    if b_print_info:
        print('\n' + text2print)
        
        print(f'detK: exa der = {ln_det_Kmat_der[idx_in]:.3e}, fd = {ln_detKmat_fd:.3e}, der diff = {ln_detKmat_der_diff:.3e}')
        print(f'Lkd:  exa der = {lkd_der[idx_in]:.3e}, fd = {lkd_fd:.3e}, der diff = {lkd_der_diff:.3e}')
        
        if calc_cond:
            print(f'Cond: exa der = {cond_der[idx_in]:.3e}, fd = {cond_fd:.3e}, der diff = {cond_fd_diff:.3e}')
        
        print(f'ln_detKmat_frac = {ln_detKmat_frac:.3}, lkd_der_frac = {lkd_der_frac:.3}, cond_fd_frac = {cond_fd_frac:.3}')
    
        if hp_beta_der.size > 0:
            print(f'beta: exa der = {hp_beta_der[0,idx_in]:.3e}, fd = {beta_fd[0]:.3e}, der diff = {beta_der_diff:.3e}')
        
    np.testing.assert_allclose(ln_det_Kmat_der[idx_in], ln_detKmat_fd, rtol = rtol, atol = atol)
    np.testing.assert_allclose(lkd_der[idx_in],         lkd_fd,        rtol = rtol, atol = atol)
    
    if calc_cond:
        np.testing.assert_allclose(cond_der[idx_in], cond_fd, rtol = rtol, atol = atol) 
        

class TestGrad(unittest.TestCase):
    
    ''' Noise-free test cases '''
    
    def test_grad_theta_wo_noise(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            for kernel_type in kernel_type_vec:
            
                b_noise_free = True
                GP_wo_noise  = setup_GP_wo_noise(wellcond_mtd, kernel_type)
                hp_kernel    = get_hp_kernel(kernel_type)
                
                hp_vals_mod           = make_hp_vals_wo_noise(GP_wo_noise, hp_kernel)
                hp_vals_mod.theta[0] += eps
                
                idx = np.min(GP_wo_noise.hp_info_optz_lkd.idx_theta)
                calc_fd_lkd(GP_wo_noise, wellcond_mtd, kernel_type, hp_vals_mod, idx, 'Hpara: theta wo noise', b_noise_free)   
     
    ''' Noisy test cases '''   
     
    def test_grad_theta_w_noise(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            
            if wellcond_mtd == 'req_vmin':
                continue
            
            for kernel_type in kernel_type_vec:
                
                b_noise_free = False
                GP_w_noise   = setup_GP_w_noise(wellcond_mtd, kernel_type)
                hp_kernel    = get_hp_kernel(kernel_type)
                
                idx = np.min(GP_w_noise.hp_info_optz_lkd.idx_theta)
                
                hp_vals_mod           = make_hp_vals_w_noise(GP_w_noise, hp_kernel)
                hp_vals_mod.theta[0] += eps
                
                calc_fd_lkd(GP_w_noise, wellcond_mtd, kernel_type, hp_vals_mod, idx, 'Hpara: theta w noise', b_noise_free)   
        
    def test_grad_varK_w_noise(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            
            if wellcond_mtd == 'req_vmin':
                continue
            
            for kernel_type in kernel_type_vec:
                
                b_noise_free = False
                GP_w_noise   = setup_GP_w_noise(wellcond_mtd, kernel_type)
                hp_kernel    = get_hp_kernel(kernel_type)
                
                idx = GP_w_noise.hp_info_optz_lkd.idx_varK
                
                hp_vals_mod       = make_hp_vals_w_noise(GP_w_noise, hp_kernel)
                hp_vals_mod.varK += eps
                
                calc_fd_lkd(GP_w_noise, wellcond_mtd, kernel_type, hp_vals_mod, idx, 'Hpara: varK w noise', b_noise_free)   
        
    def test_grad_var_fval_w_noise(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            
            if wellcond_mtd == 'req_vmin':
                continue
            
            for kernel_type in kernel_type_vec:
                
                b_noise_free = False
                GP_w_noise   = setup_GP_w_noise(wellcond_mtd, kernel_type)
                hp_kernel    = get_hp_kernel(kernel_type)
                
                idx = GP_w_noise.hp_info_optz_lkd.idx_var_fval
                
                hp_vals_mod = make_hp_vals_w_noise(GP_w_noise, hp_kernel)
                hp_vals_mod.var_fval += eps
                
                calc_fd_lkd(GP_w_noise, wellcond_mtd, kernel_type, hp_vals_mod, idx, 'Hpara: var_fval w noise', b_noise_free)    

    def test_grad_var_fgrad_w_noise(self):
        
        for wellcond_mtd in wellcond_mtd_vec:
            
            if wellcond_mtd == 'req_vmin':
                continue
            
            for kernel_type in kernel_type_vec:
                
                b_noise_free = False
                GP_w_noise   = setup_GP_w_noise(wellcond_mtd, kernel_type)
                hp_kernel    = get_hp_kernel(kernel_type)
                
                idx = GP_w_noise.hp_info_optz_lkd.idx_var_fgrad
                
                hp_vals_mod = make_hp_vals_w_noise(GP_w_noise, hp_kernel)
                hp_vals_mod.var_fgrad += eps
                
                calc_fd_lkd(GP_w_noise, wellcond_mtd, kernel_type, hp_vals_mod, idx, 'Hpara: var_fgrad w noise', b_noise_free)   

if __name__ == '__main__':
    unittest.main()
