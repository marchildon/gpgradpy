#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 07:35:41 2023

@author: andremarchildon
"""

import numpy as np 
from os import path
from scipy.stats import norm
import matplotlib.pyplot as plt

from SurrPlt import SurrPlt
from gpgradpy.src import GaussianProcess  
    
''' Set options '''

# Not to be changed
dim = 1

# Other parameters
n_eval          = 4     # No. of data points where the function is evaluated
use_grad        = True
kernel_type     = 'SqExp' # 'SqExp',' 'Maff2', 'RatQu'

wellcond_mtd    = 'precon' # 'req_vmin' # 'req_vmin' # 'req_vmin' 'precon' None

# Set range for hyperparameters
n_theta         = 101
range_gamma     = np.array([1e-1, 1e1])
gamma_vec       = np.logspace(np.log10(range_gamma[0]), np.log10(range_gamma[1]), n_theta) 
    
# Plotting options
save_fig        = True
plt_gamma_val   = False
legend_loc      = 'upper center'
surr_ylim       = [-3.5, 3.5]
n_sig           = 2 # No. of standard deviations that are plotted 

list_color_mu   = ['r', 'g', 'b']
list_color_sig  = ['pink', 'palegreen', 'cyan']

fs_ticks        = 16
fs_axis         = 18
fs_legend       = 12
fs_text         = 22
markersize      = 10

folder_all = path.join('figures', '1d_surr')

''' Define function of interest '''

xmin = 2.5 
xmax = 7.5 

def calc_obj(x):
    x_1d = x[:,0]
    return np.atleast_1d(np.sin(x_1d) + np.sin(10*x_1d / 3.0))

def calc_grad(x):
    x_1d = x[:,0]
    grad = np.atleast_1d(np.cos(x_1d) + (10.0/3.0) * np.cos(10*x_1d / 3.0))
    
    return grad[:,None]

''' Acquisiiton function '''

def calc_acq_upper_conf(fmu, fsig, beta = 2):
    return fmu - beta * fsig
    
def calc_acq_exp_improvement(fmu, fsig, fval_best):
    z = (fval_best - fmu) / fsig
    return -((fval_best - fmu) * norm.cdf(z) + fsig * norm.pdf(z))

''' Set data points '''

n_exa       = 500
x_eval      = np.linspace(xmin, xmax, n_eval+2)[1:-1,None]
x_exa       = np.linspace(xmin, xmax, n_exa)[:,None]

''' Methods '''

def calc_lkd(GP, theta_vec):
    
    n_theta     = theta_vec.size
    
    hp_mu_vec   = np.zeros(n_theta)
    varK_vec    = np.zeros(n_theta)
    ln_lkd_all  = np.zeros(n_theta)

    for i in range(n_theta):
        hp_vals = GP.make_hp_class(None, np.atleast_1d(theta_vec[i]))
        
        lkd_info, b_chofac_good = GP.calc_lkd_all(hp_vals, calc_cond = True)
        
        hp_mu_vec[i]  = lkd_info.hp_beta[0]
        varK_vec[i]   = lkd_info.hp_varK
        ln_lkd_all[i] = lkd_info.ln_lkd
        
    ln_lkd_scl = (ln_lkd_all - np.min(ln_lkd_all)) / (np.max(ln_lkd_all) - np.min(ln_lkd_all))

    return ln_lkd_scl, hp_mu_vec, varK_vec

def eval_model(GP, theta, hp_mu, varK):
    
    hp_vals_th_best = GP.make_hp_class(np.atleast_1d(hp_mu), np.atleast_1d(theta), varK = varK)
    
    GP.hp_vals = hp_vals_th_best
    GP.setup_eval_model() 
    gp_mu, gp_sig, gp_mu_grad, gp_sig_grad = GP.eval_model(x_exa)[:4]
    
    return gp_mu, gp_sig

''' Calculations '''

ndata     = n_eval * (1 + dim) if use_grad else n_eval
obj_eval  = calc_obj(x_eval)
std_fval  = np.zeros(obj_eval.shape)

if use_grad:
    grad_eval = calc_grad(x_eval)
    std_grad  = np.zeros(grad_eval.shape)
else:
    grad_eval = None
    std_grad  = None

obj_exa   = calc_obj(x_exa)
grad_exa  = calc_grad(x_exa)

''' Create the GP and evaluate the likelihood '''

GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
GP.set_data(x_eval, obj_eval, std_fval, grad_eval, std_grad)
theta_vec = GP.gamma2theta(gamma_vec)

ln_lkd_scl, hp_mu_vec, varK_vec = calc_lkd(GP, theta_vec)

# Identify local maximum of the lkd function
grad_pos = np.append(ln_lkd_scl[:-1] > ln_lkd_scl[1:], False)
grad_neg = np.append(False, ln_lkd_scl[1:] > ln_lkd_scl[:-1],)

bvec_max = np.logical_and(grad_pos, grad_neg)

idx_vec = np.arange(n_theta)
vec_idx_lkd_max = idx_vec[bvec_max]

if vec_idx_lkd_max.size == 0:
    vec_idx_lkd_max = np.array([int(n_theta/2)])

vec_idx_lkd_max = np.array([0, np.argmax(ln_lkd_scl), -1])
# vec_idx_lkd_max = np.array([62, 62, 62])

n_max = vec_idx_lkd_max.size
    
''' Plot the scaled likelihood '''

fig1, ax1 = plt.subplots()

ax1.set_xscale('log')
ax1.tick_params(axis='both', labelsize=fs_ticks)

ax1.set_xlabel(r'$\gamma$', fontsize = fs_axis)
ax1.set_ylabel(r'Scaled likelihood', fontsize = fs_axis)
ax1.plot(gamma_vec, ln_lkd_scl, 'k-')

for i in range(n_max):
    idx = vec_idx_lkd_max[i]
    ax1.plot(gamma_vec[idx], ln_lkd_scl[idx], list_color_mu[i] + 'o', markersize = markersize)

ax1.grid(True)
plt.tight_layout()

if save_fig:
    if use_grad:
        fig_name_lkd = 'Scl_ln_lkd_w_grad.png'
    else:
        fig_name_lkd = 'Scl_ln_lkd_wo_grad.png'
        
    full_path = path.join(folder_all, fig_name_lkd)
    fig1.savefig(full_path, dpi = 800, format='png')

''' Evaluate the model '''

gp_mu_all        = np.zeros((n_max, n_exa))
gp_sig_all       = np.zeros((n_max, n_exa))

gp_mu_grad_all   = np.zeros((n_max, n_exa, dim))
gp_sig_grad_all  = np.zeros((n_max, n_exa, dim))

gp_mu_grad_xeval_all   = np.zeros((n_max, n_eval, dim))
gp_sig_grad_xeval_all  = np.zeros((n_max, n_eval, dim))

for i in range(n_max):
    
    idx = vec_idx_lkd_max[i]
    hp_vals_th_best = GP.make_hp_class(np.atleast_1d(hp_mu_vec[idx]), np.atleast_1d(theta_vec[idx]), varK = varK_vec[idx])
    GP.hp_vals = hp_vals_th_best
    
    GP.setup_eval_model() 
    
    gp_mu_all[i,:], gp_sig_all[i,:], gp_mu_grad_all[i,:,:], gp_sig_grad_all[i,:,:] = GP.eval_model(x_exa, calc_grad = True)[:4]

''' Plot the model '''

if use_grad:
    base_str_surr_plt = 'Surr_w_grad_'
else:
    base_str_surr_plt = 'Surr_wo_grad_'

surr_plt = SurrPlt() 

for i in range(n_max):

    fig, ax = plt.subplots()

    surr_plt.plot_surr(ax, x_exa, obj_exa, x_eval, obj_eval, gp_mu_all[i,:], gp_sig_all[i,:], n_sig, 
                       color_mean = list_color_mu[i], 
                       color_std  = list_color_sig[i], 
                       legend_loc = legend_loc)
    
    # ax.grid()
    ax.tick_params(axis='both', labelsize=fs_ticks)
    ax.set_xlabel(r'$x$', fontsize = fs_axis)
    ax.set_ylabel(r'$f(x)$', fontsize = fs_axis, rotation=0)
    
    gamma = gamma_vec[vec_idx_lkd_max[i]]
    
    if plt_gamma_val:
        ax.text(xmax - 0.05, surr_ylim[1] - 0.7, rf'$\gamma = {gamma:.1f}$', fontsize = fs_text, ha = 'right')
    
    ax.set_ylim(surr_ylim)
    plt.tight_layout()
    
    if save_fig:
        full_path = path.join(folder_all, base_str_surr_plt + f'_{gamma:.1e}.png')
        fig.savefig(full_path, dpi = 800, format='png')

