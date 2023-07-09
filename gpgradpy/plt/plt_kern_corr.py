#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 08:36:38 2022

@author: andremarchildon
"""

import numpy as np
from os import path, makedirs
import matplotlib.pyplot as plt

''' Set parameters '''

save_figs         = True

plt_sq_exp        = True 
plt_Ma5f2         = True
plt_rat_qd        = True
plot_together_fwf = True

# Hyperparameter for the rational quadratic kernel
alpha = 2

# Plotting options
nr    = 100
rmin  = 6
rvec  = np.linspace(-rmin, rmin, nr)
Xrvec, Yrvec = np.meshgrid(rvec, rvec)

legend_fs   = 12
label_fs    = 16
tick_fs     = 14

dpi     = 800
folders = path.join('figures', 'kern_corr')

plot_style = ['b-', 'g-', 'm-', 'c-', 'k-']

labels_corr_1d = [r'$\mathrm{corr} \left( f, f \right)$', 
                  r'$\mathrm{corr} \left( f, \frac{\partial f}{\partial x} \right)$', 
                  r'$\mathrm{corr} \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial x} \right)$']

# labels_corr_1d = [r'$k(\tilde{r})$', 
#                   r'$\frac{dk(\tilde{r})}{d\tilde{r}}$', 
#                   r'$-\frac{d^2 k(\tilde{r})}{d\tilde{r}^2}$']

if not path.exists(folders):
    makedirs(folders)

''' Calculate correlations '''

def calc_sq_exp_1d_corr(rvec):
    
    k_vec = np.exp(-0.5*rvec**2)
    
    sq_exp_corr_fwf = k_vec * 1
    sq_exp_corr_fwg = -rvec * k_vec
    sq_exp_corr_gwg = (1 - rvec**2) * k_vec
    
    return sq_exp_corr_fwf, sq_exp_corr_fwg, sq_exp_corr_gwg

def calc_sq_exp_2d_corr(Xrvec_in, Yrvec_in):
    return -Xrvec * Yrvec * np.exp(-0.5*(Xrvec**2 + Yrvec**2))
    
def calc_Ma5f2_1d_corr(rvec):

    rnorm = np.abs(rvec)
    A     = np.exp(-np.sqrt(3) * rnorm)
    
    mat5f2_corr_fwf = (1 + np.sqrt(3) * rnorm + rnorm**2) * A
    mat5f2_corr_fwg = -rvec * ( 1 + np.sqrt(3) * rnorm) * A
    mat5f2_corr_gwg = ((1 + np.sqrt(3) * rnorm) - 3 * rvec**2) * A

    return mat5f2_corr_fwf, mat5f2_corr_fwg, mat5f2_corr_gwg

def calc_rat_qd_corr(rvec, alpha):
    
    base = 1 + rvec**2 / (2*alpha)
    rat_qd_corr_fwf = base**(-alpha)
    rat_qd_corr_fwg = -rvec * base**(-alpha -1)
    rat_qd_corr_gwg = base**(-alpha -1) - rvec**2 * ((alpha + 1) / alpha) * base**(-alpha-2)
    
    return rat_qd_corr_fwf, rat_qd_corr_fwg, rat_qd_corr_gwg

# 1D correlation
sq_exp_corr_fwf, sq_exp_corr_fwg, sq_exp_corr_gwg = calc_sq_exp_1d_corr(rvec)
mat5f2_corr_fwf, mat5f2_corr_fwg, mat5f2_corr_gwg = calc_Ma5f2_1d_corr(rvec)
rat_qd_corr_fwf, rat_qd_corr_fwg, rat_qd_corr_gwg = calc_rat_qd_corr(rvec, alpha)

# 2D correlation
sq_exp_corr_gwg_2d = calc_sq_exp_2d_corr(Xrvec, Yrvec)

''' Plotting functions '''

def make_1d_plot():
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlabel(r'$\tilde{r} = \gamma(x - y)$', fontsize = label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.grid()
    
    return fig, ax

def plt_1d_corr(corr_fwf, corr_fwg, corr_gwg, str_kern):
    
    fig1, ax1 = make_1d_plot()
    
    ax1.plot(rvec, corr_fwf, plot_style[0], label = labels_corr_1d[0])
    ax1.plot(rvec, corr_fwg, plot_style[1], label = labels_corr_1d[1])
    ax1.plot(rvec, corr_gwg, plot_style[2], label = labels_corr_1d[2])
    
    ax1.legend(fontsize = legend_fs, loc = 'upper left')
    fig1.tight_layout()
    
    if save_figs:
        fullpath = path.join(folders, f'corr_1d_{str_kern}.png')
        plt.savefig(fullpath, dpi = dpi, format = 'png')

def plt_2d_corr(corr_gwg_2d, str_kern):
    levels = np.linspace(-0.4, 0.4, 9)
    
    fig2, ax2 = plt.subplots()
    
    ax2.set_aspect('equal', adjustable='box')
    ax2.tick_params(axis='both', which='major', labelsize=tick_fs)
    ax2.grid(True)
    
    ax2.set_xlabel(r'$\tilde{r}_1 = \gamma_1(x_1 - y_1)$', size=label_fs)
    ax2.set_ylabel(r'$\tilde{r}_2 = \gamma_2(x_2 - y_2)$', size=label_fs)
    
    cs = ax2.contourf(Xrvec, Yrvec, corr_gwg_2d, levels = levels)
    fig2.colorbar(cs)
    
    fig2.tight_layout()
    plt.show() 
    
    if save_figs:
        fullpath = path.join(folders, f'corr_2d_{str_kern}.png')
        plt.savefig(fullpath, dpi = dpi, format = 'png')

''' Plot '''

if plt_sq_exp:
    plt_1d_corr(sq_exp_corr_fwf, sq_exp_corr_fwg, sq_exp_corr_gwg, 'sq_exp')
    plt_2d_corr(sq_exp_corr_gwg_2d, 'sq_exp')

if plt_Ma5f2:
    plt_1d_corr(mat5f2_corr_fwf, mat5f2_corr_fwg, mat5f2_corr_gwg, 'mat5f2')
    
if plt_rat_qd:
    plt_1d_corr(rat_qd_corr_fwf, rat_qd_corr_fwg, rat_qd_corr_gwg, f'rat_qd_a{alpha}')

if plot_together_fwf:
    fig, ax = make_1d_plot()
    
    ax.plot(rvec, sq_exp_corr_fwf, plot_style[0], label='Gaussian')
    ax.plot(rvec, mat5f2_corr_fwf, plot_style[1], label=r'Matérn $\frac{5}{2}$')  
    ax.plot(rvec, rat_qd_corr_fwf, plot_style[2], label=rf'Rat quad, $\alpha = {alpha}$')
    
    ax.legend(fontsize = legend_fs, loc = 'upper left')
    fig.tight_layout()
    
    if save_figs:
        fullpath = path.join(folders, 'corr_all_kernel_fwf.png')
        plt.savefig(fullpath, dpi = dpi, format = 'png')
