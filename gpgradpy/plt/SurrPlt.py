#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:06:00 2023

@author: andremarchildon
"""

import numpy as np 
import matplotlib.pyplot as plt

class SurrPlt:
    
    fs_ticks   = 16
    fs_axis    = 18
    fs_legend  = 12
    fs_text    = 22
    markersize = 10
    
    # def add_surr_std(self, ax, x_model, gp_mu, gp_sig, n_sig, color_mean, color_std):
        
    #     # Plot the line for min and max uncertainty for +/- n_sig
    #     model_lb = gp_mu - n_sig * gp_sig
    #     model_ub = gp_mu + n_sig * gp_sig
        
    #     ax.plot(x_model, model_lb, '-', color='grey', lw=1, label='_nolabel_')[0] 
    #     ax.plot(x_model, model_ub, '-', color='grey', lw=1, label='_nolabel_')[0] 
        
    #     ax.plot(x_model, gp_mu, '-', color = color_mean, label = r'$\mu_f$')
    #     ax.fill_between(x_model[:,0], model_lb, model_ub, facecolor=color_std, alpha=0.5, interpolate=True, label=f'$\pm {n_sig} \sigma_f$')
    
    def plot_surr(self, ax, x_model, obj_exa, xeval, obj_xeval, gp_mu, gp_sig, n_sig, 
                  color_mean    = 'g',     color_std      = 'palegreen', 
                  legend_loc    = 'best',  ncol_legend    = 2,           
                  fs_legend     = None,    grid_axis      = 'both',
                  xeval_ms      = 8,       fmin_ms        = 14, 
                  b_plt_fmin    = False, 
                  b_plt_exact   = True,    b_plt_eval     = True,        
                  b_plt_surr_mu = True,    b_plt_surr_sig = True):
        
        if fs_legend is None:
            fs_legend = self.fs_legend
        
        ax.grid(axis=grid_axis)
        ax.tick_params(axis='both', labelsize=self.fs_ticks)
        
        if b_plt_surr_sig:
            # Plot the line for min and max uncertainty for +/- n_sig
            model_lb = gp_mu - n_sig * gp_sig
            model_ub = gp_mu + n_sig * gp_sig
            
            ax.plot(x_model, model_lb, '-', color='grey', lw=1, label='_nolabel_')[0] 
            ax.plot(x_model, model_ub, '-', color='grey', lw=1, label='_nolabel_')[0] 
        
        # Plot the exa sol and evaluation points
        if b_plt_exact:
            ax.plot(x_model, obj_exa, 'k-', label=r'$f(x)$')
            
        if b_plt_fmin:
            idx_fmin = np.argmin(obj_exa)
            ax.plot(x_model[idx_fmin], obj_exa[idx_fmin], 'r*', ms = fmin_ms, label=r'min $f(x)$')
            
        if b_plt_surr_mu:
            ax.plot(x_model, gp_mu, '-', color = color_mean, label = r'$\mu_f$')
        
        if b_plt_surr_sig:
            ax.fill_between(x_model[:,0], model_lb, model_ub, facecolor=color_std, alpha=0.5, interpolate=True, label=f'$\pm {n_sig} \sigma_f$')
        
        if b_plt_eval:
            ax.plot(xeval, obj_xeval, 's', color = color_mean, ms = xeval_ms, label=r'$x_{\mathrm{eval}}$')
        
        if b_plt_fmin:
            # Plot again to be on top of mu and sig
            ax.plot(x_model[idx_fmin], obj_exa[idx_fmin], 'r*', ms = fmin_ms, label=r'_nolabel_')
        
        if legend_loc == 'out':
            ax.legend(fontsize = fs_legend, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc = legend_loc, ncol = ncol_legend, fontsize = fs_legend)
            
    def plot_acq(self, ax, x_vec, acq_val_all, str_acq_vec, 
                 b_scl_data = False,           b_plt_min_acq = True, 
                 color_vec  = ['r', 'g', 'b'], legend_loc    = 'upper right', 
                 ms_acq_min = 14,              grid_axis     = 'y'):
        
        n_acq = acq_val_all.shape[1]
        
        if b_scl_data:
            acq_val_2plt = np.zeros(acq_val_all.shape)
            
            for i in range(n_acq):
                min_data = np.min(acq_val_all[:,i])
                max_data = np.max(acq_val_all[:,i])
                acq_val_2plt[:,i] = (acq_val_all[:,i] - min_data) / np.max((1e-8, (max_data - min_data)))
        else:
            acq_val_2plt = acq_val_all
        
        ax.grid(axis=grid_axis)
        ax.tick_params(axis = 'both', labelsize = self.fs_ticks)
        
        ax.set_xlabel(r'$x$', fontsize = self.fs_axis)
        ax.set_ylabel(r'Acquisition', fontsize = self.fs_axis)

        for i in range(n_acq):
            ax.plot(x_vec, acq_val_2plt[:,i], '-', color = color_vec[i], label = str_acq_vec[i])
            
            if b_plt_min_acq:
                label = 'min(' + str_acq_vec[i] + ')'
                idx_argmin = np.argmin(acq_val_2plt[:,i])
                ax.plot(x_vec[idx_argmin,0], acq_val_2plt[idx_argmin,i], color_vec[i] + '*', 
                        label = label, ms = ms_acq_min)
            
        if legend_loc == 'out':
            ax.legend(fontsize = self.fs_legend, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(fontsize = self.fs_legend, loc='upper left')
            
        plt.tight_layout()
    