#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:23:21 2023

@author: andremarchildon
"""

import numpy as np 
from os import path, makedirs
import matplotlib.pyplot as plt
from scipy.spatial import distance
from smt.sampling_methods import LHS

from gpgradpy.src import GaussianProcess  
 
''' Set options '''

n_eval      = 10
cond_max    = 1e10 # Maximum condition number

# Plotting options
n_gamma     = 500
range_gamma = np.array([1e-3, 1e3])

save_fig    = True
folder_all  = path.join('figures', 'nugget')

label_fs    = 15
tick_fs     = 15
markersize  = 12

cmap_cond   = 'jet' # 'viridis' 'jet', 'rainbow', 'turbo'
cmap_lkd    = 'viridis' #'viridis' # 'Greens', 'BuGn'

varK = 0.1
dim = 2 # This function is must use dim = 2
use_grad     = True 
kernel_type  = 'SqExp'
wellcond_mtd = 'precon'    

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

''' Calculate certain nuggets '''

eta_w_trace  = n_eval * (dim + 1) / (cond_max - 1)
eta_Gaussian = (1 + (n_eval - 1) * 0.5*(1 + np.sqrt(1 + 4*dim)) * np.exp(-(1+2*dim - np.sqrt(1 + 4*dim))/(4*dim))) / (cond_max - 1)

''' Preliminary calculations '''



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

std_fval_vec = np.zeros(obj_eval.shape)
std_grad_vec = np.zeros(grad_eval.shape)

''' Create the GP and evaluate the surrogate'''

if use_grad:        
    str_wellcond_mtd = wellcond_mtd
    
    base_file_name = f'Kern_{kernel_type}_mtd_{str_wellcond_mtd}_grad_T_n{n_eval}.png'
else:
    base_file_name = f'Kern_{kernel_type}_grad_F_n{n_eval}.png'
    assert wellcond_mtd is None, 'If use_grad is False, then wellcond_mtd must be None'
    
GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)

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

lkd_all     = np.full((n_gamma), np.nan)

eta_Gcircle_all = np.full((n_gamma), np.nan)
eta_eig_all     = np.full((n_gamma), np.nan)
eta_w_cond_l2   = np.full((n_gamma), np.nan)
eta_w_cond_l1   = np.full((n_gamma), np.nan)

Kcor_is_diagdom = np.zeros((n_gamma), dtype=bool)
condKcor = np.full((n_gamma), np.nan)

ndata =  n_eval * (dim + 1)

for i in range(n_gamma):
    theta    = np.array([theta_vec[i], theta_vec[i]])
    hp_vals  = GP.make_hp_class(varK = varK, theta = theta, kernel = hp_kernel)
    lkd_info = GP.calc_lkd_all(hp_vals, calc_lkd = True, calc_cond = True, calc_grad = False)[0]
    
    lkd_all[i]   = lkd_info.ln_lkd
    
    if GP.Rtensor_init is not None:
        Kcor    = GP.calc_all_K_w_chofac(GP.Rtensor_init, hp_vals)[1]
        eigall  = np.linalg.eigvalsh(Kcor)
        
        if np.max(np.sum(np.abs(Kcor), axis=1)) <= 1:
            Kcor_is_diagdom[i] = True 
        else:
            Kcor_is_diagdom[i] = False
        
        eig_max_est          = np.max(np.sum(np.abs(Kcor), axis=1))
        eta_Gcircle_all[i] = eig_max_est / (cond_max - 1)
        
        eta_eig_all[i]     = np.max((0, (np.max(eigall) - np.min(eigall) * cond_max) / (cond_max - 1)))
        
        condK_l1    = np.linalg.cond(Kcor, p=1)
        condK_l2    = np.linalg.cond(Kcor, p=2)
        
        condKcor[i] = condK_l2
        
        eigmax_est  = ndata * condK_l2 / (condK_l2 + ndata - 1)
        eta_w_cond_l2[i] = (eigmax_est / (cond_max - 1)) * (1 - cond_max / condK_l2)
        
        eigmax_est  = ndata * np.sqrt(ndata) * condK_l1 / (condK_l1 * np.sqrt(ndata) + ndata - 1)
        eta_w_cond_l1[i] = np.max((0, (eigmax_est / (cond_max - 1)) * (1 - cond_max / (np.sqrt(ndata) * condK_l1))))
   
    
# Check diago dominance
gamma_Kcor_diagdom = gamma_vec[Kcor_is_diagdom]
cond2large = gamma_vec[condKcor > cond_max]

idx_lkd_max = np.argmax(lkd_all)

    
''' Plot '''

def setup_plots():
    
    fig, ax = plt.subplots(figsize=(6,3))
    ax.grid()
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\gamma_1 = \gamma_2$', size=label_fs)
    
    
    return fig, ax

# # Plot the nuggets
# fig, ax = setup_plots()

# ax.plot(gamma_vec, np.ones(n_gamma) * eta_w_trace, '-k', label='$\eta$ calculated with trace of K')
# ax.plot(gamma_vec, eta_Gcircle_all, '-b', label='Gcircle')
# ax.plot(gamma_vec, eta_eig_all, '-r', label='Minimum required $\eta$')
# # ax.plot(gamma_vec, eta_w_cond_l2, '-y', label='CondK l2')
# ax.plot(gamma_vec, eta_w_cond_l1, '-m', label='CondK l1')

# ax.set_yscale('log')
# ax.set_ylabel(r'$\eta$', size=label_fs, rotation=0)
# fig.tight_layout()
# ax.legend()
        
# Plot the scalled nuggets
fig, ax = setup_plots()

ax.plot(gamma_vec, eta_w_cond_l1 / eta_w_trace, '-r', label=r'$\eta$ from the $\ell_2$ condition number')
ax.plot(gamma_vec, np.ones(n_gamma) * eta_Gaussian / eta_w_trace, '-b', label=r'Constant $\eta$: Gershgorin circle theorem')
ax.plot(gamma_vec, eta_Gcircle_all / eta_w_trace, '-m', label=r'Variable $\eta$: Gershgorin circle theorem')
# ax.plot(gamma_vec, eta_w_cond_l2, '-y', label='CondK l2')
ax.plot(gamma_vec, eta_eig_all / eta_w_trace, '-g', label='Minimum required $\eta$')

ax.set_yscale('linear')
ax.set_ylabel(r'$\frac{\eta}{\eta_{\mathrm{tr}}}$', size=label_fs+3, rotation=0, labelpad = 10)
fig.tight_layout()

ylim = ax.get_ylim()
ax.plot([gamma_vec[idx_lkd_max], gamma_vec[idx_lkd_max]], ylim, 'c--')

ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.46, 0.5, 0.5))

if save_fig:
    file_name = 'nugget_1d.png'
    full_path = path.join(folder_all, file_name)
    fig.savefig(full_path, dpi = 800, format='png')

# Plots the condition number
fig, ax = setup_plots()

ax.plot(gamma_vec, cond_max * np.ones(n_gamma), 'r--')
ax.plot(gamma_vec, condKcor, 'k-')


ax.set_yscale('log')
ax.set_ylabel(r'$\kappa(\dot{\mathrm{K}}_{\nabla})$', size=label_fs)
plt.tight_layout()

ylim = ax.get_ylim()
ax.plot([gamma_vec[idx_lkd_max], gamma_vec[idx_lkd_max]], ylim, 'c--')

if save_fig:
    file_name = 'condnum_1d.png'
    full_path = path.join(folder_all, file_name)
    fig.savefig(full_path, dpi = 800, format='png')



 