#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:21:12 2022

@author: andremarchildon
"""

import unittest
import numpy as np

from gpgradpy.base import CommonFun
from gpgradpy import GaussianProcess

# np.set_printoptions(precision=2)

''' Parameters to vary '''

kernel_type_vec     = ['SqExp', 'Ma5f2', 'RatQu']

list_wellcond_mtd_wo_noise = [None, 'precon'] # [None, 'precon', 'req_vmin',]
list_wellcond_mtd_w_noise  = [None, 'precon'] # req_vmin is not setup for noisy problems

''' Set parameters '''

eps  = 1e-6

eta_dflt     = 1e-2
use_grad     = True
wellcond_mtd = 'precon' 

x_eval   = 1e-3 * np.array([[0, 0], [1, 0.5], [0.5, 1.0]])
n_eval, dim = x_eval.shape

bvec_use_grad     = np.ones(n_eval, dtype=bool)
bvec_use_grad[-1] = False

# Testing tolerances
rtol    = 1e-5
atol    = 1e-7

''' Setup GP '''

n_data      = n_eval * (dim + 1) if use_grad else n_eval
dist_all    = CommonFun.calc_Rtensor(x_eval, x_eval, 1)

# Data 
obj      = np.zeros(n_eval) 
grad_all = np.zeros(x_eval.shape)
grad     = grad_all[bvec_use_grad,:]

# Default inconsequential required parameters 
i_optz = 1

# Default parameters
beta_dflt      = np.array([np.mean(obj)])
theta          = 10 * np.linspace(1.5, 3, dim)

varK_dflt      = 2.0 # 0.3
var_fval_dflt  = 0.2
var_fgrad_dflt = 0.3

method2set_hp = 'current' # 'set'

def get_hp_kernel(kernel_type):
    
    if kernel_type == 'RatQu':
        hp_kernel = 2.5
    else:
        hp_kernel = None
        
    return hp_kernel

def make_hp_vals(wellcond_mtd, kernel_type, varK_in, hp_var_fval_in, hp_var_fgrad_in):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP_base   = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    
    return GP_base.make_hp_class(None, np.copy(theta), hp_kernel, 
                                 varK_in, hp_var_fval_in, hp_var_fgrad_in)

def calc_err_cutoff(GP, eta):
    
    # Cutoff for var_fval
    cutoff_var_fval  = varK_dflt * eta 
    
    gamma = GP.theta2gamma(theta)
    
    min_gamma2 = np.min(gamma)**2
    max_gamma2 = np.max(gamma)**2
    
    cutoff_var_fgrad = varK_dflt * eta * np.array([min_gamma2, max_gamma2])
    
    return cutoff_var_fval, cutoff_var_fgrad

def setup_GP_w_zero_err(wellcond_mtd, kernel_type):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.init_optz_surr(2)
    
    std_fval_zero  = np.zeros(n_eval)
    std_fgrad_zero = np.zeros(grad.shape)
    
    GP.set_custom_hp(beta_dflt, theta, hp_kernel, varK_dflt, None, None)
    
    if use_grad:
        GP.set_data(x_eval, obj, std_fval_zero, grad, std_fgrad_zero, bvec_use_grad)
    else:
        GP.set_data(x_eval, obj, std_fval_zero)

    GP.set_hpara(method2set_hp, i_optz)
    
    GP._etaK = eta_dflt
    hp_vals = make_hp_vals(wellcond_mtd, kernel_type, varK_dflt, None, None)
    
    return GP, hp_vals

def setup_GP_w_small_known_err(wellcond_mtd, kernel_type):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.init_optz_surr(2)
    
    etaK = GP.calc_nugget(n_eval)[1]
    cutoff_var_fval, cutoff_var_fgrad = calc_err_cutoff(GP, etaK)
    
    cutoff_std_fval  = np.sqrt(cutoff_var_fval)
    cutoff_std_fgrad = np.sqrt(np.min(cutoff_var_fgrad))
    
    std_fval_zero  = np.ones(n_eval) * (cutoff_std_fval / 5)
    std_fgrad_zero = np.ones(grad.shape) * (cutoff_std_fgrad / 5)
    
    GP.set_custom_hp(beta_dflt, theta, hp_kernel, varK_dflt, None, None)
    
    GP.set_data(x_eval, obj, std_fval_zero, grad, std_fgrad_zero, bvec_use_grad)
    GP.set_hpara(method2set_hp, i_optz)
    GP._etaK = eta_dflt
    
    hp_vals = make_hp_vals(wellcond_mtd, kernel_type, varK_dflt, None, None)
    
    assert GP._etaK == eta_dflt
    
    return GP, hp_vals

def setup_GP_w_big_known_err(wellcond_mtd, kernel_type):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.init_optz_surr(2)
    
    etaK     = eta_dflt
    # etaK   = GP.calc_nugget(n_eval)[1]
    
    cutoff_var_fval, cutoff_var_fgrad = calc_err_cutoff(GP, etaK)
    
    cutoff_std_fval = np.sqrt(cutoff_var_fval)
    cutoff_std_fgrad = np.sqrt(np.max(cutoff_var_fgrad))
    
    std_fval_zero  = np.ones(n_eval) * (cutoff_std_fval * 100)
    std_fgrad_zero = np.ones(grad.shape) * (cutoff_std_fgrad * 100)
    
    GP.set_data(x_eval, obj, std_fval_zero, grad, std_fgrad_zero, bvec_use_grad)
    GP.set_custom_hp(beta_dflt, theta, hp_kernel, varK_dflt, None, None)

    hp_vals = make_hp_vals(wellcond_mtd, kernel_type, varK_dflt, None, None)
    
    GP._etaK = eta_dflt
    assert GP._etaK == eta_dflt
    
    return GP, hp_vals

def setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.init_optz_surr(2)
    
    cutoff_var_fval, cutoff_var_fgrad = calc_err_cutoff(GP, eta_dflt)
    
    hp_var_fval     = cutoff_var_fval / 5
    hp_var_fgrad    = np.min(cutoff_var_fgrad) / 5
    
    std_fval_zero   = None
    std_fgrad_zero  = None
    
    GP.set_data(x_eval, obj, std_fval_zero, grad, std_fgrad_zero, bvec_use_grad)
    GP.set_custom_hp(beta_dflt, theta, hp_kernel, varK_dflt, hp_var_fval, hp_var_fgrad)
    
    GP._etaK = eta_dflt
    
    hp_vals = make_hp_vals(wellcond_mtd, kernel_type, varK_dflt, hp_var_fval, hp_var_fgrad)
    
    return GP, hp_vals

def setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type):
    
    hp_kernel = get_hp_kernel(kernel_type)
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.init_optz_surr(2)
    
    etaK = GP.calc_nugget(n_eval)[1]
    cutoff_var_fval, cutoff_var_fgrad = calc_err_cutoff(GP, etaK)
    
    hp_var_fval     = cutoff_var_fval * 5
    hp_var_fgrad    = np.max(cutoff_var_fgrad) * 5
    
    std_fval_zero   = None
    std_fgrad_zero  = None
    
    GP.set_data(x_eval, obj, std_fval_zero, grad, std_fgrad_zero, bvec_use_grad)
    GP.set_custom_hp(beta_dflt, theta, hp_kernel, varK_dflt, hp_var_fval, hp_var_fgrad)
    
    GP._etaK = eta_dflt
    assert GP._etaK == eta_dflt
    
    hp_vals = make_hp_vals(wellcond_mtd, kernel_type, varK_dflt, hp_var_fval, hp_var_fgrad)
    
    return GP, hp_vals

''' Calculate grad and fd '''

def calc_grad_theta(GP, hp_vals):
    
    Kern, _, Kcov       = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]
    KernGrad_theta      = GP.calc_Kern_grad_theta(dist_all, hp_vals.theta, hp_vals.kernel, bvec_use_grad)
    Kcov_grad_theta     = GP.calc_Kcov_grad_theta(hp_vals, dist_all)

    hp_vals.theta[0] += eps
    Kern_w_eps, _, Kcov_w_eps \
        = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]

    Kern_fd = (Kern_w_eps - Kern)  / eps
    Kcov_fd = (Kcov_w_eps  - Kcov) / eps
    
    return KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd
  
def calc_grad_alpha(GP, hp_vals):
    
    Kern, _, Kcov       = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]
    KernGrad_alpha      = GP.calc_Kern_grad_alpha(dist_all, hp_vals.theta, hp_vals.kernel, bvec_use_grad)
    Kcov_grad_alpha     = GP.calc_Kcov_grad_alpha(hp_vals, dist_all)

    hp_vals.kernel += eps
    Kern_w_eps, _, Kcov_w_eps \
        = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]

    Kern_fd = (Kern_w_eps - Kern)  / eps
    Kcov_fd = (Kcov_w_eps  - Kcov) / eps
    
    return KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd
  
def calc_grad_varK(GP, hp_vals):
    
    Kern, _, Kcov   = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]
    Kcov_grad_varK  = GP.calc_Kcov_grad_varK(hp_vals, Kern)

    hp_vals.varK += eps
    Kern_w_eps, _, Kcov_w_eps = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[:3]

    Kcov_fd = (Kcov_w_eps  - Kcov)  / eps
    
    return Kcov_grad_varK, Kcov_fd
  
def calc_grad_var_fval(GP, hp_vals):
    
    Kcov = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[2]
    Kcov_grad_var_fval  = GP.calc_Kcov_grad_var_fval(hp_vals)

    hp_vals.var_fval += eps
    Kcov_w_eps = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[2]

    Kcov_fd = (Kcov_w_eps  - Kcov)  / eps
    
    return Kcov_grad_var_fval, Kcov_fd
  
def calc_grad_var_fgrad(GP, hp_vals):
    
    Kcov         = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[2]
    Kcov_grad_var_fgrad = GP.calc_Kcov_grad_var_fgrad(hp_vals)

    hp_vals.var_fgrad += eps
    Kcov_w_eps = GP.calc_all_K_w_chofac(dist_all, hp_vals, calc_chofac = False)[2]

    Kcov_fd = (Kcov_w_eps  - Kcov)  / eps
    
    return Kcov_grad_var_fgrad, Kcov_fd
  
''' Run tests '''

class TestGradKmat(unittest.TestCase):
    
    ''' Grad theta '''
    
    def test_grad_theta_zero_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_wo_noise:
            for kernel_type in kernel_type_vec:
            
                GP, hp_vals = setup_GP_w_zero_err(wellcond_mtd, kernel_type)
                KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd = calc_grad_theta(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_theta[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_theta[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_theta_small_known_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
            
                GP, hp_vals = setup_GP_w_small_known_err(wellcond_mtd, kernel_type)
                KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd = calc_grad_theta(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_theta[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_theta[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_theta_big_known_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_big_known_err(wellcond_mtd, kernel_type)
                KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd = calc_grad_theta(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_theta[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_theta[0,:,:], rtol = rtol, atol = atol)
     
    def test_grad_theta_small_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
                
                GP, hp_vals = setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type)
                KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd = calc_grad_theta(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_theta[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_theta[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_theta_big_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type)
                KernGrad_theta, Kcov_grad_theta, Kern_fd, Kcov_fd = calc_grad_theta(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_theta[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_theta[0,:,:], rtol = rtol, atol = atol)
        
    ''' Grad alpha '''
    
    def test_grad_alpha_zero_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_wo_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_zero_err(wellcond_mtd, kernel_type)
                
                if hp_vals.kernel is None:
                    return
                
                KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd = calc_grad_alpha(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_alpha[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_alpha[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_alpha_small_known_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
            
                GP, hp_vals = setup_GP_w_small_known_err(wellcond_mtd, kernel_type)
                
                if hp_vals.kernel is None:
                    continue 
                
                KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd = calc_grad_alpha(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_alpha[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_alpha[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_alpha_big_known_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_big_known_err(wellcond_mtd, kernel_type)
                
                if hp_vals.kernel is None:
                    continue
                
                KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd = calc_grad_alpha(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_alpha[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_alpha[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_alpha_small_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type)
                
                if hp_vals.kernel is None:
                    continue
                
                KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd = calc_grad_alpha(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_alpha[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_alpha[0,:,:], rtol = rtol, atol = atol)
        
    def test_grad_alpha_big_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type)
                
                if hp_vals.kernel is None:
                    continue
                
                KernGrad_alpha, Kcov_grad_alpha, Kern_fd, Kcov_fd = calc_grad_alpha(GP, hp_vals)
        
                np.testing.assert_allclose(Kern_fd, KernGrad_alpha[0,:,:],  rtol = rtol, atol = atol)
                np.testing.assert_allclose(Kcov_fd, Kcov_grad_alpha[0,:,:], rtol = rtol, atol = atol)
        
    ''' Grad wrt varK ''' 
     
    def test_grad_varK_small_known_err(self):
         
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_small_known_err(wellcond_mtd, kernel_type)
                Kcov_grad_varK, Kcov_fd = calc_grad_varK(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_varK,  rtol = rtol, atol = atol)
     
    def test_grad_varK_big_known_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
        
                GP, hp_vals = setup_GP_w_big_known_err(wellcond_mtd, kernel_type)
                Kcov_grad_varK, Kcov_fd = calc_grad_varK(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_varK,  rtol = rtol, atol = atol)   
     
    def test_grad_varK_small_unknown_err(self):
         
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
            
                GP, hp_vals = setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_varK, Kcov_fd = calc_grad_varK(GP, hp_vals)
                  
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_varK,  rtol = rtol, atol = atol)
     
    def test_grad_varK_big_unknown_err(self):
         
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
                
                GP, hp_vals = setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_varK, Kcov_fd = calc_grad_varK(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_varK,  rtol = rtol, atol = atol)      
     
    ''' Grad wrt var_fval '''
    
    def test_grad_var_fval_small_unknown_err(self):
    
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
            
                GP, hp_vals = setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_var_fval, Kcov_fd = calc_grad_var_fval(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_var_fval,  rtol = rtol, atol = atol)
     
    def test_grad_var_fval_big_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
         
                GP, hp_vals = setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_var_fval, Kcov_fd = calc_grad_var_fval(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_var_fval,  rtol = rtol, atol = atol)   
    
    ''' Grad wrt var_fgrad '''
    
    def test_grad_var_fgrad_small_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
         
                GP, hp_vals = setup_GP_w_small_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_var_fgrad, Kcov_fd = calc_grad_var_fgrad(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_var_fgrad,  rtol = rtol, atol = atol)
     
    def test_grad_var_fgrad_big_unknown_err(self):
        
        for wellcond_mtd in list_wellcond_mtd_w_noise:
            for kernel_type in kernel_type_vec:
                
                GP, hp_vals = setup_GP_w_big_unknown_err(wellcond_mtd, kernel_type)
                Kcov_grad_var_fgrad, Kcov_fd = calc_grad_var_fgrad(GP, hp_vals)
                
                np.testing.assert_allclose(Kcov_fd,  Kcov_grad_var_fgrad,  rtol = rtol, atol = atol)   
     
if __name__ == '__main__':
    unittest.main()
