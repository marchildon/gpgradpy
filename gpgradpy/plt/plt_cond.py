#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:16:36 2023

@author: andremarchildon
"""

import numpy as np 
from os import path, makedirs
import matplotlib.pyplot as plt
from scipy.spatial import distance
from smt.sampling_methods import LHS

from gpgradpy import GaussianProcess  
 
''' Set options '''

n_eval      = 10
cond_max    = 1e10 # Maximum condition number

# Plotting options
n_gamma     = 100
range_gamma = np.array([1e-4, 1e8])

save_fig    = True
plt_nugget  = False
folder_all  = path.join('figures', '2D_cond')

label_fs    = 12
tick_fs     = 11
markersize  = 12

cmap_cond   = 'jet' # 'viridis' 'jet', 'rainbow', 'turbo'
cmap_lkd    = 'viridis' #'viridis' # 'Greens', 'BuGn'

lvl_lkd     = np.linspace(0, 1, 11)
lvl_cond    = np.linspace(0, 12, 7)
lvl_eta     = np.linspace(0, 1, 9)
# lvl_eta     = np.linspace(1, 3, 9)

''' Define the function of interest '''

# This does not impact the condition number of the covariance matrix but it 
# impacts the values of the hyperparameters that maximize the marginal log-likelihood

def calc_obj(xy, a = 10):
    # Rosenbrock function eval
    
    n_eval, dim = xy.shape
    obj = np.zeros(n_eval)
    
    for d in range(dim-1):
        obj += a*(xy[:,d+1] - xy[:,d]**2)**2 + (1 - xy[:,d])**2

    return obj

def calc_grad(xy, a = 10):
    # Rosenbrock function grad
    
    n_eval, dim = xy.shape
    
    grad = np.zeros([n_eval, dim])

    x1        = xy[:,0]
    x2        = xy[:,1]
    grad[:,0] = -2*(1 - x1) - 4*a* x1 *(x2 - x1**2)

    grad[:,-1] = 2*a * (xy[:,-1] - xy[:,-2]**2)

    for d in range(1, dim-1):
        xd_n1 = xy[:,d-1]
        xd    = xy[:,d]
        xd_p1 = xy[:,d+1]

        grad[:,d] += -2*(1 - xd) - 4*a * xd *(xd_p1 - xd**2) + 2*a * (xd - xd_n1**2)

    return grad

''' Preliminary calculations '''

varK = 0.1

dim = 2 # This function is must use dim = 2

if path.exists(folder_all) is False:
    makedirs(folder_all)

# Calculate the required nugget for the preconditioning method for the 
# squared exponential kernel. This nugget is used for all kernels and all 
# wellcond_mtd methods 
GP     = GaussianProcess(dim, use_grad = True, kernel_type = 'SqExp', wellcond_mtd = 'precon')
nugget = GP.calc_nugget(n_eval)[1]

def make_lhs(para_min, para_max, n_eval):
    
    para_limit      = np.array([para_min, para_max]).T
    xdata_sampling  = LHS(xlimits=para_limit, random_state = 2)
    x_eval          = xdata_sampling(n_eval) 
    
    return x_eval

# Set Latin hypercube of nodes
center    = 1.0 
factor    = 1e-2
para_min  = center - factor * np.ones(dim)
para_max  = center + factor * np.ones(dim)
x_eval    = make_lhs(para_min, para_max, n_eval)

# Evaluate the function of interest
obj_eval  = calc_obj(x_eval)
grad_eval = calc_grad(x_eval)

# For the plots
para_tick_loc = 10**np.arange(np.log10(range_gamma[0]), np.log10(range_gamma[1])+0.1, 2)

# Print info 
dist_x_eval = distance.cdist(x_eval, x_eval, 'euclidean')
dist_min    = np.nanmin(dist_x_eval + np.diag(np.full(n_eval, np.nan)))
print(f'dim = {dim}, n_eval = {n_eval}, nugget = {nugget:.2e}, dist_min = {dist_min:.2e}')

def calc_n_plot(use_grad, kernel_type, wellcond_mtd, 
                use_noisy_precon = True, use_const_eta   = False, 
                std_fval_scalar  = 0,    std_grad_scalar = 0):
    
    std_fval_vec  = std_fval_scalar * np.ones(obj_eval.shape)
    std_grad_vec  = std_grad_scalar * np.ones(grad_eval.shape)
    
    print(f'use_grad = {use_grad}, kernel_type = {kernel_type}, wellcond_mtd = {wellcond_mtd}')

    ''' Create the GP and evaluate the surrogate'''
    
    if std_fval_scalar > 0:
        str_noise = f'StdObj_{std_fval_scalar:.1e}'.replace('.', 'p')
    else:
        str_noise = ''
    
    if use_grad:
        if wellcond_mtd == 'precon':
            if use_noisy_precon:
                str_wellcond_mtd = 'precon2'
            else:
                str_wellcond_mtd = 'precon1'
        else:
            str_wellcond_mtd = wellcond_mtd
        
        if std_grad_scalar > 0:
            str_noise += f'StdGrad_{std_grad_scalar:.1e}'.replace('.', 'p')
        
        base_file_name = f'Kern_{kernel_type}_mtd_{str_wellcond_mtd}_grad_T_n{n_eval}_{str_noise}.png'
    else:
        base_file_name = f'Kern_{kernel_type}_grad_F_n{n_eval}_{str_noise}.png'
        assert wellcond_mtd is None, 'If use_grad is False, then wellcond_mtd must be None'
        
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.use_noisy_precon = use_noisy_precon
    GP.use_const_eta = False
    
    if use_grad:
        GP.set_data(x_eval, obj_eval, std_fval_vec, grad_eval, std_grad_vec)
    else:
        GP.set_data(x_eval, obj_eval, std_fval_vec)
        
    GP._etaK      = nugget
    GP._eta_Kgrad = nugget
    GP.cond_max   = cond_max
    
    hp_kernel   = GP.hp_kernel_default
    
    range_theta = GP.gamma2theta(range_gamma)
    theta_vec   = np.logspace(np.log10(range_theta[0]), np.log10(range_theta[1]), n_gamma) 
    gamma_vec   = GP.theta2gamma(theta_vec)
    
    lkd_all     = np.full((n_gamma, n_gamma), np.nan)
    condK_all   = np.full((n_gamma, n_gamma), np.nan)
    # etaK_all    = np.full((n_gamma, n_gamma), np.nan)
    
    eta_Gcircle_all = np.full((n_gamma, n_gamma), np.nan)
    eta_eig_all     = np.full((n_gamma, n_gamma), np.nan)
    
    for i in range(n_gamma):
        for j in range(n_gamma):
            theta    = np.array([theta_vec[i], theta_vec[j]])
            hp_vals  = GP.make_hp_class(varK = varK, theta = theta, kernel = hp_kernel)
            lkd_info = GP.calc_lkd_all(hp_vals, calc_lkd = True, calc_cond = True, calc_grad = False)[0]
            
            if GP.Rtensor_init is not None:
                Kcor    = GP.calc_all_K_w_chofac(GP.Rtensor_init, hp_vals)[1]
                eigall  = np.linalg.eigvalsh(Kcor)
                
                eig_max_est          = np.max(np.sum(np.abs(Kcor), axis=1))
                eta_Gcircle_all[i,j] = eig_max_est / (cond_max - 1)
                
                eta_eig_all[i,j]     = np.max((0, (np.max(eigall) - np.min(eigall) * cond_max) / (cond_max - 1)))
            
            lkd_all[i,j]   = lkd_info.ln_lkd
            condK_all[i,j] = lkd_info.cond
            
    log10_condK_all = np.log10(condK_all)
    # etaK_scl_all    = GP._etaK / etaK_all
    print(f'wellcond_mtd = {wellcond_mtd}, cond: min = {np.min(log10_condK_all):.2}, max = {np.max(log10_condK_all):.2}')
            
    min_lkd = np.nanmin(lkd_all)
    max_lkd = np.nanmax(lkd_all)
    scl_lkd_all = (lkd_all - min_lkd) / (max_lkd - min_lkd)    
    
    ''' Identify max likelihood ''' 
    
    if np.sum(np.isnan(scl_lkd_all)) == n_gamma**2:
        loc_max_lkd  = None
    else:
        max_each_row = np.nanmax(scl_lkd_all, axis=1)
        idx_max_col  = np.nanargmax(max_each_row)
        idx_max_row  = np.nanargmax(scl_lkd_all[idx_max_col, :])
        
        loc_max_lkd  = np.array([gamma_vec[idx_max_row], gamma_vec[idx_max_col]])
    
    def setup_plot():
    
        fig, ax = plt.subplots(figsize=(4, 4))
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        if dim != 2:
            raise Exception('Not considered')
        
        ax.set_xlabel(r'$\gamma_1$', size=label_fs)
        ax.set_ylabel(r'$\gamma_2$', size=label_fs, rotation=0)
        
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=tick_fs)
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xticks(para_tick_loc)
        ax.set_yticks(para_tick_loc)
            
        return fig, ax
        
    ''' Plot the condition number with eta '''
    
    Xmat_para, Ymat_para  = np.meshgrid(gamma_vec, gamma_vec)
    
    fig0, ax0 = setup_plot()
    
    cs = ax0.contourf(Xmat_para, Ymat_para, log10_condK_all, levels=lvl_cond, cmap=cmap_cond, extend='max')
    
    if loc_max_lkd is not None:
        ax0.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
            
    fig0.colorbar(cs, pad = 0.08)
    
    fig0.tight_layout()
    plt.show() 
    
    if save_fig:
        file_name = 'Cond_' + base_file_name
        full_path = path.join(folder_all, file_name)
        fig0.savefig(full_path, dpi = 800, format='png')
    
    ''' Plot the likelihood number '''
    
    fig2, ax2 = setup_plot()
    
    cs = ax2.contourf(Xmat_para, Ymat_para, scl_lkd_all, levels=lvl_lkd, cmap=cmap_lkd)
    
    if loc_max_lkd is not None:
        ax2.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
        
    if np.max(log10_condK_all) > np.log10(cond_max):
        ax2.contour(Xmat_para, Ymat_para, scl_lkd_all, levels=[np.log10(cond_max)])
    
    fig2.colorbar(cs, pad = 0.08)
    
    plt.show() 
    
    if save_fig:
        file_name = 'Lkd_' + base_file_name
        full_path = path.join(folder_all, file_name)
        fig2.savefig(full_path, dpi = 800, format='png')
        
    ''' Plot the nugget fraction etaK(X) / etaK_const '''
    
    if wellcond_mtd == 'precon':
        # eta_Gcircle_scl_all = GP._etaK / eta_Gcircle_all
        # eta_eig_scl_all = eta_Gcircle_all / eta_eig_all
        
        eta_Gcircle_scl_all = eta_Gcircle_all / GP._etaK
        eta_eig_scl_all     = eta_eig_all / GP._etaK
        
        eta_Gcircle_scl_at_lkdmax = eta_Gcircle_scl_all[idx_max_row, idx_max_col]
        eta_eig_scl_at_lkdmax     = eta_eig_scl_all[idx_max_row, idx_max_col]
        
        print(f'eta_Gcircle_scl_at_lkdmax = {eta_Gcircle_scl_at_lkdmax:.3}')
        print(f'eta_eig_scl_at_lkdmax     = {eta_eig_scl_at_lkdmax:.3}')
        
        if not plt_nugget:
            return
        
        # Plot eta_Gcircle_scl_all
        fig1, ax1 = setup_plot()
        
        cs = ax1.contourf(Xmat_para, Ymat_para, eta_Gcircle_scl_all, levels=lvl_eta, cmap=cmap_lkd)
        
        if loc_max_lkd is not None:
            ax1.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
                
        fig1.colorbar(cs, pad = 0.08)
        
        fig1.tight_layout()
        plt.show() 
        
        if save_fig:
            file_name = 'EtaGcircle_' + base_file_name
            full_path = path.join(folder_all, file_name)
            fig1.savefig(full_path, dpi = 800, format='png')
            
        # Plot eta_eig_scl_all
        fig1, ax1 = setup_plot()
        
        cs = ax1.contourf(Xmat_para, Ymat_para, eta_eig_scl_all, levels=lvl_eta, cmap=cmap_lkd)
        
        if loc_max_lkd is not None:
            ax1.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
                
        fig1.colorbar(cs, pad = 0.08)
        
        fig1.tight_layout()
        plt.show() 
        
        if save_fig:
            file_name = 'EtaEig_' + base_file_name
            full_path = path.join(folder_all, file_name)
            fig1.savefig(full_path, dpi = 800, format='png')
        
    # return eta_Gcircle_scl_all, GP._etaK, etaK_all
    
''' Grad-free, noise-free '''

if False:
    use_grad         = False
    kernel_type_vec  = ['SqExp', 'Ma5f2', 'RatQu']
    wellcond_mtd     = None
    use_noisy_precon = True
    use_const_eta    = True
    std_fval_scalar  = 0
    std_grad_scalar  = 0
    
    for kernel_type in kernel_type_vec:
        calc_n_plot(use_grad,kernel_type, wellcond_mtd)

''' Gaussian kernel with gradients '''

# Noise free
if False:
    use_grad         = True
    kernel_type      = 'SqExp'
    wellcond_mtd     = 'precon'
    use_noisy_precon = True
    use_const_eta    = True
    std_fval_scalar  = 0
    std_grad_scalar  = 0
    
    calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, 
                std_fval_scalar, std_grad_scalar)

# Noise free without second precon
if False:
    use_grad         = True
    kernel_type      = 'SqExp'
    wellcond_mtd     = 'precon'
    use_noisy_precon = False
    use_const_eta    = True
    std_fval_scalar  = 0
    std_grad_scalar  = 0
    
    calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, 
                std_fval_scalar, std_grad_scalar)

# Noisy
if False:
    use_grad         = True
    kernel_type      = 'SqExp'
    wellcond_mtd     = 'precon'
    use_noisy_precon = True
    use_const_eta    = True
    std_fval_scalar  = 1e-6
    std_grad_scalar  = 1e-1
    
    calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, 
                std_fval_scalar, std_grad_scalar)
    
''' Non-Gaussian kernels with gradients '''

# Baseline method with constant nuggget
if False:
    use_grad         = True
    kernel_type_vec  = ['Ma5f2', 'RatQu']
    wellcond_mtd     = None
    use_noisy_precon = True
    use_const_eta    = False
    std_fval_scalar  = 0
    std_grad_scalar  = 0
    
    calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, 
                std_fval_scalar, std_grad_scalar)

# Precon method with variable nugget
if True:
    use_grad         = True
    kernel_type_vec  = ['Ma5f2', 'RatQu']
    wellcond_mtd     = 'precon'
    use_noisy_precon = True
    use_const_eta    = False
    std_fval_scalar  = 0
    std_grad_scalar  = 0
    
    for kernel_type in kernel_type_vec:
        calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, 
                    std_fval_scalar, std_grad_scalar)


# # Cases with gradients 
# use_grad         = True
# wellcond_mtd_vec = [None, 'req_vmin', 'precon']
# kernel_type_vec  = ['SqExp', 'Ma5f2', 'RatQu']

# for kernel_type in kernel_type_vec:
#     for wellcond_mtd in wellcond_mtd_vec:
#             calc_n_plot(use_grad, kernel_type, wellcond_mtd)
    
# # Cases without gradients 
# use_grad         = False
# wellcond_mtd     = None
# kernel_type_vec  = ['SqExp', 'Ma5f2', 'RatQu']

# for kernel_type in kernel_type_vec:
#     calc_n_plot(use_grad,kernel_type, wellcond_mtd)

# Cases with and without the second preconditioning
# use_grad        = True
# kernel_type     = 'SqExp'
# wellcond_mtd    = 'precon'
# std_fval_scalar = 1e-6
# std_grad_scalar = 0.1

# use_noisy_precon = True
# calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, std_fval_scalar, std_grad_scalar)

# use_noisy_precon = False
# calc_n_plot(use_grad, kernel_type, wellcond_mtd, use_noisy_precon, use_const_eta, std_fval_scalar, std_grad_scalar)
