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

''' Define the function of interest '''

# xmin = 2.5
# xmax = 7.5

xmin = 2
xmax = 8

def calc_obj(x):
    x_1d = x[:,0]
    return np.atleast_1d(np.sin(x_1d) + np.sin(10*x_1d / 3.0))

def calc_grad(x):
    x_1d = x[:,0]
    grad = np.atleast_1d(np.cos(x_1d) + (10.0/3.0) * np.cos(10*x_1d / 3.0))
    
    return grad[:,None]

''' Acquisiiton functions '''

def calc_acq_upper_conf(fmu, fsig, beta = 2):
    return fmu - beta * fsig
    
def calc_acq_exp_improvement(fmu, fsig, fval_best):
    z = (fval_best - fmu) / fsig
    return -((fval_best - fmu) * norm.cdf(z) + fsig * norm.pdf(z))
    
''' Set options '''

use_grad     = True
kernel_type  = 'SqExp' # 'SqExp',' 'Maff2', 'RatQu'
wellcond_mtd = 'precon' # 'req_vmin' # 'req_vmin' # 'req_vmin' 'precon' None
legend_loc   = 'out' # Not to be changed

# No. of function evaluations
n_eval  = 4
n_exa   = 500

# x_eval  = np.linspace(xmin, xmax, n_eval+2)[1:-1,None]
x_eval = np.array([3.5, 4.5, 5.5, 6.5])[:,None]

x_exa   = np.linspace(xmin, xmax, n_exa)[:,None]

# Set the hyperparameters for the surrogate 
# model_mean = np.array([0])
theta_vec  = np.array([1.5099758602010078])
# varK       = 1.0

# Plotting
n_sig   = 2  # For the uncertainty of the surrogate
fs_axis = 18 # Font size for the labels


dpi        = 200
folder_all = 'figures'

ms_acq_min    = 12
save_fig      = True
color_vec_acq = ['g', 'b', 'm', 'r']

''' Evaluate the function of interest '''

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

''' Create the GP and evaluate the surrogate'''

n_eval, dim = x_eval.shape
GP = GaussianProcess(dim, use_grad, kernel_type, wellcond_mtd)
GP.set_data(x_eval, obj_eval, std_fval, grad_eval, std_grad)

hp_vals     = GP.make_hp_class(None, theta_vec)
lkd_info    = GP.calc_lkd_all(hp_vals, calc_cond = True)[0]

hp_mu       = np.array([lkd_info.hp_beta[0]])
hp_varK     = lkd_info.hp_varK

# hp_mu = np.array([-1.4])

hp_vals_th_best = GP.make_hp_class(hp_mu, theta_vec, varK = hp_varK)
GP.hp_vals      = hp_vals_th_best

GP.setup_eval_model() 

gp_mu_all, gp_sig_all = GP.eval_model(x_exa, calc_grad = True)[:2]

''' Calc the acquisition function with model '''

n_acq = 2

str_acq_vec = [''] * n_acq
acq_val_all = np.zeros((n_exa, n_acq))

# Aquisition function: mean of surrogate
str_acq_vec[0]   = r'$\mu_f$'
acq_val_all[:,0] = calc_acq_upper_conf(gp_mu_all, gp_sig_all, beta = 0)

# Aquisition function: upper confidence
beta = 2
str_acq_vec[1]   = rf'$\mu_f - {beta} \sigma_f$'
acq_val_all[:,1] = calc_acq_upper_conf(gp_mu_all, gp_sig_all, beta = beta)

# # Acquisition function: expected improvement
# fval_best        = np.min(obj_eval)
# str_acq_vec[2]   = r'-$\int_{-\infty}^{f^*} \mathcal{N}(z;\mu_f(x), \sigma_f(x)) dz$'
# acq_val_all[:,2] = calc_acq_exp_improvement(gp_mu_all, gp_sig_all, fval_best)

min_acq_val = np.min(acq_val_all, axis=0)
max_acq_val = np.max(acq_val_all, axis=0)

acq_val_scl_all = (acq_val_all - min_acq_val[None,:]) / (max_acq_val[None,:] - min_acq_val[None,:])


''' Plot '''

surr_plt = SurrPlt() 

fig, axes = plt.subplots(2, figsize=(8,4), sharex=True)

plt.rcParams.update({'text.usetex': True,
                      'text.latex.preamble': r'\usepackage{amsfonts}'})



ax = axes[0] 
ax.set_xlim([xmin, xmax])
ax.set_ylabel('Surrogate', fontsize = fs_axis)
surr_plt.plot_surr(ax, x_exa, obj_exa, x_eval, obj_eval, gp_mu_all, gp_sig_all, 
                   n_sig, legend_loc = legend_loc)

ax = axes[1] 
# ax.set_ylabel('Acquisition',  fontsize  = fs_axis)
surr_plt.plot_acq(ax, x_exa, acq_val_scl_all, str_acq_vec, 
                  legend_loc = legend_loc,
                  color_vec  = color_vec_acq,
                  b_scl_data = True
                  )

if save_fig:
    fullpath = path.join(folder_all, 'surr_w_acq_1d.png')
    fig.savefig(fullpath, dpi=dpi)
    
    