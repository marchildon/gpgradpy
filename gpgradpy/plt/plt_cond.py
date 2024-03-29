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

from gpgradpy.src import GaussianProcess  
 
''' Set options '''

n_eval      = 20
cond_max    = 1e10 # Maximum condition number

# Plotting options
n_gamma     = 20
range_gamma = np.array([1e-4, 1e8])

save_fig    = False
folder_all  = path.join('figures', '2D_cond')

fs_tick     = 14
fs_label    = 18
fs_title    = 18

pad_title   = 10
markersize  = 12

cmap_cond   = 'jet' # 'viridis' 'jet', 'rainbow', 'turbo'
cmap_lkd    = 'viridis' #'viridis' # 'Greens', 'BuGn'

lvl_lkd     = np.linspace(0, 1, 11)
lvl_cond    = np.linspace(0, 12, 7)
lvl_eta     = np.linspace(0, 1, 9)
# lvl_eta     = np.linspace(1, 3, 9)

plt_shrink = 0.8
plt_aspect = 15

figsize = (4,4)

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
dim  = 2 # This function is must use dim = 2

if path.exists(folder_all) is False:
    makedirs(folder_all)

# Calculate the required nugget for the preconditioning method for the 
# squared exponential kernel. This nugget is used for all kernels and all 
# wellcond_mtd methods 
GP          = GaussianProcess(dim, use_grad = True, kernel_type = 'SqExp', wellcond_mtd = 'precon')
nugget_dflt = GP.calc_nugget(n_eval)[1]

def make_lhs(para_min, para_max, n_eval):
    
    para_limit      = np.array([para_min, para_max]).T
    xdata_sampling  = LHS(xlimits=para_limit, random_state = 2)
    x_eval          = xdata_sampling(n_eval) 
    
    return x_eval

# Set Latin hypercube of nodes
center    = 1.0 
factor    = 1e-1
para_min  = center - factor * np.ones(dim)
para_max  = center + factor * np.ones(dim)
x_eval    = make_lhs(para_min, para_max, n_eval)

# Evaluate the function of interest
obj_eval  = calc_obj(x_eval)
grad_eval = calc_grad(x_eval)

# For the plots
para_tick_loc = 10**np.arange(np.log10(range_gamma[0]), np.log10(range_gamma[1])+0.1, 4)
# para_tick_loc = 10**np.arange(np.log10(range_gamma[0]), np.log10(range_gamma[1])+0.1, 2)

# Print info 
dist_x_eval = distance.cdist(x_eval, x_eval, 'euclidean')
dist_min    = np.nanmin(dist_x_eval + np.diag(np.full(n_eval, np.nan)))
print(f'dim = {dim}, n_eval = {n_eval}, nugget_dflt = {nugget_dflt:.2e}, dist_min = {dist_min:.2e}')

range_theta = GP.gamma2theta(range_gamma)
theta_vec   = np.logspace(np.log10(range_theta[0]), np.log10(range_theta[1]), n_gamma) 
gamma_vec   = GP.theta2gamma(theta_vec)

Xmat_para, Ymat_para  = np.meshgrid(gamma_vec, gamma_vec)

def setup_plot(ax, set_labels = True):

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    if dim != 2:
        raise Exception('Not considered')
    
    if set_labels:
        ax.set_xlabel(r'$\gamma_1$', size=fs_label)
        ax.set_ylabel(r'$\gamma_2$', size=fs_label, rotation=0)
    
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=fs_tick)
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xticks(para_tick_loc)
    ax.set_yticks(para_tick_loc)

def calc_cond_lkd(use_grad, kernel_type, wellcond_mtd = 'precon', 
                  std_fval_scalar   = 0,    std_grad_scalar = 0, 
                  cond_eta_is_const = True, nugget          = None):
    
    std_fval_vec = std_fval_scalar * np.ones(obj_eval.shape)
    std_grad_vec = std_grad_scalar * np.ones(grad_eval.shape)
    
    if nugget is None:
        nugget = nugget_dflt
    
    print(f'use_grad = {use_grad}, kernel_type = {kernel_type}, wellcond_mtd = {wellcond_mtd}')

    ''' Create the GP and evaluate the surrogate'''
    
    GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
    GP.cond_eta_is_const = cond_eta_is_const
    
    if use_grad:
        GP.set_data(x_eval, obj_eval, std_fval_vec, grad_eval, std_grad_vec)
    else:
        GP.set_data(x_eval, obj_eval, std_fval_vec)
        
    GP._etaK      = nugget
    GP._eta_Kgrad = nugget
    GP.cond_max   = cond_max
    
    hp_kernel   = GP.hp_kernel_default
    
    lkd_all     = np.full((n_gamma, n_gamma), np.nan)
    condK_all   = np.full((n_gamma, n_gamma), np.nan)
    
    for i in range(n_gamma):
        for j in range(n_gamma):
            theta    = np.array([theta_vec[i], theta_vec[j]])
            hp_vals  = GP.make_hp_class(varK = varK, theta = theta, kernel = hp_kernel)
            lkd_info = GP.calc_lkd_all(hp_vals, calc_lkd = True, calc_cond = True, calc_grad = False)[0]
            
            lkd_all[i,j]   = lkd_info.ln_lkd
            condK_all[i,j] = lkd_info.cond
    
    log10_condK_all  = np.log10(condK_all)
    
    ''' Identify max likelihood ''' 
    
    if np.sum(np.isnan(lkd_all)) == n_gamma**2:
        loc_max_lkd  = None
    else:
        max_each_row = np.nanmax(lkd_all, axis=1)
        idx_max_col  = np.nanargmax(max_each_row)
        idx_max_row  = np.nanargmax(lkd_all[idx_max_col, :])
        
        loc_max_lkd  = np.array([gamma_vec[idx_max_row], gamma_vec[idx_max_col]])
    
    return log10_condK_all, lkd_all, loc_max_lkd

def calc_scl_data(data):
    
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_scl = (data - data_min) / (data_max - data_min)    
    
    return data_scl

def set_base_fig_name(kernel_type, std_fval_scalar, std_grad_scalar):
    
    if std_fval_scalar > 0:
        str_noise = f'_StdObj_{std_fval_scalar:.1e}'.replace('.', 'p')
    else:
        str_noise = ''
    
    if use_grad:        
        str_wellcond_mtd = wellcond_mtd
        
        if std_grad_scalar > 0:
            str_noise += f'StdGrad_{std_grad_scalar:.1e}'.replace('.', 'p')
        
        base_file_name = f'Kern_{kernel_type}_mtd_{str_wellcond_mtd}_grad_T_n{n_eval}{str_noise}.png'
    else:
        base_file_name = f'Kern_{kernel_type}_grad_F_n{n_eval}{str_noise}.png'
        assert wellcond_mtd is None, 'If use_grad is False, then wellcond_mtd must be None'
        
    return base_file_name

def plot_cond_number(fig, ax, log10_condK_all, loc_max_lkd = None, 
                     base_file_name = None, add_color_bar = True):
    
    # Xmat_para, Ymat_para  = np.meshgrid(gamma_vec, gamma_vec)
    
    cs = ax.contourf(Xmat_para, Ymat_para, log10_condK_all, levels=lvl_cond, cmap=cmap_cond, extend='max')
    
    if loc_max_lkd is not None:
        ax.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
         
    if add_color_bar:
        fig.colorbar(cs, pad = 0.08, shrink = plt_shrink, aspect = plt_aspect)
        fig.tight_layout()
    
    if save_fig and (base_file_name is not None):
        file_name = 'Cond_' + base_file_name
        full_path = path.join(folder_all, file_name)
        fig.savefig(full_path, dpi = 800, format='png', bbox_inches='tight')
        
    return cs

def plot_lkd(fig, ax, scl_lkd_all, log10_condK_all, loc_max_lkd = None, base_file_name = None):
    
    cs = ax.contourf(Xmat_para, Ymat_para, scl_lkd_all, levels=lvl_lkd, cmap=cmap_lkd)
    
    if loc_max_lkd is not None:
        ax.plot(loc_max_lkd[0], loc_max_lkd[1], 'k*', markersize = markersize)
        
    if log10_condK_all is not None:
        if np.max(log10_condK_all) > np.log10(cond_max):
            ax.contour(Xmat_para, Ymat_para, scl_lkd_all, levels=[np.log10(cond_max)])
    
    fig.colorbar(cs, pad = 0.08, shrink = plt_shrink, aspect = plt_aspect)
    
    fig.tight_layout()
    plt.show() 
    
    if save_fig and (base_file_name is not None):
        file_name = 'Lkd_' + base_file_name
        full_path = path.join(folder_all, file_name)
        fig.savefig(full_path, dpi = 800, format='png', bbox_inches='tight')

def calc_n_plot(use_grad, kernel_type, wellcond_mtd, 
                cond_eta_is_const = False, 
                std_fval_scalar   = 0,    std_grad_scalar = 0):
    
    ''' Calculations '''
    
    base_file_name = set_base_fig_name(kernel_type, std_fval_scalar, std_grad_scalar)
    
    log10_condK_all, lkd_all, loc_max_lkd \
        = calc_cond_lkd(use_grad, kernel_type, wellcond_mtd, 
                        std_fval_scalar, std_grad_scalar, 
                        cond_eta_is_const)
        
    scl_lkd_all     = calc_scl_data(lkd_all)
    
    ''' Plot '''
    
    fig0, ax0 = plt.subplots(figsize=figsize)
    setup_plot(ax0)
    plot_cond_number(fig0, ax0, log10_condK_all, loc_max_lkd, base_file_name)
    
    fig1, ax1 = plt.subplots(figsize=figsize)
    setup_plot(ax1)
    plot_lkd(fig1, ax1, scl_lkd_all, log10_condK_all, loc_max_lkd, base_file_name)
        
''' Grad-free, noise-free '''

# use_grad          = False
# kernel_type_vec   = ['SqExp']
# wellcond_mtd      = None
# cond_eta_is_const = True
# std_fval_scalar   = 0
# std_grad_scalar   = 0

# for kernel_type in kernel_type_vec:
#     calc_n_plot(use_grad,kernel_type, wellcond_mtd)

if False:
    use_grad          = False
    kernel_type_vec   = ['SqExp', 'Ma5f2', 'RatQu']
    wellcond_mtd      = None
    cond_eta_is_const = True
    std_fval_scalar   = 0
    std_grad_scalar   = 0
    
    for kernel_type in kernel_type_vec:
        calc_n_plot(use_grad,kernel_type, wellcond_mtd)

''' Gaussian kernel with gradients '''

# Noise-free
if False:
    use_grad          = True
    kernel_type       = 'SqExp'
    wellcond_mtd_vec  = [None, 'req_vmin', 'precon']
    cond_eta_is_const = True
    std_fval_scalar   = 0
    std_grad_scalar   = 0
    
    for wellcond_mtd in wellcond_mtd_vec:
        calc_n_plot(use_grad, kernel_type, wellcond_mtd, cond_eta_is_const, 
                    std_fval_scalar, std_grad_scalar)

# # Noisy
# if False:
#     use_grad          = True
#     kernel_type       = 'SqExp'
#     wellcond_mtd      = 'precon'
#     cond_eta_is_const = True
#     std_fval_scalar   = 1e-6
#     std_grad_scalar   = 1e-1
    
#     calc_n_plot(use_grad, kernel_type, wellcond_mtd, cond_eta_is_const, 
#                 std_fval_scalar, std_grad_scalar)
    
''' Non-Gaussian kernels with gradients '''

# Baseline method with constant nuggget
if False:
    use_grad          = True
    kernel_type       = 'SqExp'
    wellcond_mtd      = None
    cond_eta_is_const = False
    std_fval_scalar   = 0
    std_grad_scalar   = 0
    
    calc_n_plot(use_grad, kernel_type, wellcond_mtd, cond_eta_is_const, 
                std_fval_scalar, std_grad_scalar)

# Precon method with variable nugget
if True:
    use_grad          = True
    kernel_type_vec   = ['Ma5f2', 'RatQu']
    wellcond_mtd_vec  = ['precon', None]
    cond_eta_is_const = False
    std_fval_scalar   = 0
    std_grad_scalar   = 0
    
    for kernel_type in kernel_type_vec:
        for wellcond_mtd in wellcond_mtd_vec:
            calc_n_plot(use_grad, kernel_type, wellcond_mtd, cond_eta_is_const, 
                        std_fval_scalar, std_grad_scalar)

''' All kernels with grad, noisy-case '''

if False:
    use_grad          = True
    kernel_type_vec   = ['SqExp', 'Ma5f2', 'RatQu']
    wellcond_mtd      = 'precon'
    cond_eta_is_const = False
    std_fval_scalar   = 1e-6
    std_grad_scalar   = 1e-1
    
    for kernel_type in kernel_type_vec:
        calc_n_plot(use_grad, kernel_type, wellcond_mtd, cond_eta_is_const, 
                    std_fval_scalar, std_grad_scalar)

# ''' Plot with and without the use of a nugget '''

# kernel_type  = 'SqExp'
# wellcond_mtd = None

# use_grad = True
# log10_condK_wGrad_woEta_all, lkd_wGrad_woEta_all, loc_max_wGrad_woEta_lkd \
#     = calc_cond_lkd(use_grad, kernel_type, wellcond_mtd, nugget = 0)
    
# use_grad = False
# log10_condK_woGrad_woEta_all, lkd_woGrad_woEta_all, loc_max_woGrad_woEta_lkd \
#     = calc_cond_lkd(use_grad, kernel_type, wellcond_mtd, nugget = 0)
    
# use_grad = True
# log10_condK_wGrad_wEta_all, lkd_wGrad_wEta_all, loc_max_wGrad_wEta_lkd \
#     = calc_cond_lkd(use_grad, kernel_type, wellcond_mtd)
    
# use_grad = False
# log10_condK_woGrad_wEta_all, lkd_woGrad_wEta_all, loc_max_woGrad_wEta_lkd \
#     = calc_cond_lkd(use_grad, kernel_type, wellcond_mtd)

# # Plot only without nugget
# fig, axes = plt.subplots(1, 2, figsize=(5,2.5), sharey=True)

# axes[0].set_ylabel(r'$\gamma_2$', size=fs_label, rotation=0)
# axes[0].set_title(r'log($\kappa(\mathrm{K})$)',          size = fs_title)
# axes[1].set_title(r'log($\kappa(\mathrm{K}_{\nabla})$)', size = fs_title)

# for i in range(2):
#     axes[i].set_xlabel(r'$\gamma_1$', size=fs_label)

# setup_plot(axes[0], set_labels = False)
# setup_plot(axes[1], set_labels = False)

# plot_cond_number(fig, axes[0], log10_condK_woGrad_woEta_all, loc_max_woGrad_wEta_lkd, add_color_bar = False)
# color_axis = plot_cond_number(fig, axes[1], log10_condK_wGrad_woEta_all, loc_max_wGrad_wEta_lkd, add_color_bar = False)

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
# fig.colorbar(color_axis, cax=cbar_ax)

# # Plot only without nugget
# fig, axes = plt.subplots(2, 2, figsize=(6, 5), sharey=True, sharex=True)

# axes[0,0].set_title(r'log($\kappa(\mathrm{K} + \eta \mathrm{I})$)',          size = fs_title, pad = pad_title)
# axes[0,1].set_title(r'log($\kappa(\mathrm{K}_{\nabla} + \eta \mathrm{I})$)', size = fs_title, pad = pad_title)

# for i in range(2):
#     axes[i,0].set_ylabel(r'$\gamma_2$', size=fs_label, rotation=0)
#     axes[1,i].set_xlabel(r'$\gamma_1$', size=fs_label)
    
#     for j in range(2):
#         setup_plot(axes[i,j], set_labels = False)



# plot_cond_number(fig, axes[0,0], log10_condK_woGrad_woEta_all, loc_max_woGrad_wEta_lkd, add_color_bar = False)
# color_axis = plot_cond_number(fig, axes[0,1], log10_condK_wGrad_woEta_all, loc_max_wGrad_wEta_lkd, add_color_bar = False)

# plot_cond_number(fig, axes[1,0], log10_condK_woGrad_wEta_all, loc_max_woGrad_wEta_lkd, add_color_bar = False)
# plot_cond_number(fig, axes[1,1], log10_condK_wGrad_wEta_all, loc_max_wGrad_wEta_lkd, add_color_bar = False)

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
# fig.colorbar(color_axis, cax=cbar_ax)
