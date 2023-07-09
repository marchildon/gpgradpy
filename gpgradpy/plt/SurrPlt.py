#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:06:00 2023

@author: andremarchildon
"""

import numpy as np 

class SurrPlt:
    
    fs_ticks   = 16
    fs_axis    = 18
    fs_legend  = 12
    fs_text    = 22
    markersize = 10
    
    def plot_surr(self, ax, x_model, obj_exa, xeval, obj_xeval, gp_mu, gp_sig, n_sig, 
                  color_mean = 'g', color_std = 'palegreen', legend_loc = 'upper right'):
        
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=self.fs_ticks)
        
        # Plot the line for min and max uncertainty for +/- n_sig
        model_lb = gp_mu - n_sig * gp_sig
        model_ub = gp_mu + n_sig * gp_sig
        
        ax.plot(x_model, model_lb, '-', color='grey', lw=1, label='_nolabel_')[0] 
        ax.plot(x_model, model_ub, '-', color='grey', lw=1, label='_nolabel_')[0] 
        
        # Plot the exa sol and evaluation points
        ax.plot(x_model, obj_exa, 'k-', label='Exact')
        ax.plot(xeval, obj_xeval, 's', color = color_mean, label=r'$x_{\mathrm{eval}}$')
        
        # Plot The mean and uncertainty of the surrogates
        ax.plot(x_model, gp_mu, '-', color = color_mean, label = 'Surr mean')
        ax.fill_between(x_model[:,0], model_lb, model_ub, facecolor=color_std, alpha=0.5, interpolate=True, label=f'$\pm {n_sig} \sigma$')
        
        if legend_loc == 'out':
            ax.legend(fontsize = self.fs_legend, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(loc = legend_loc, fontsize = self.fs_legend)
            
    def plot_acq(self, ax, x_vec, acq_val_all, str_acq_vec, 
                 color_vec = ['r', 'g', 'b'], legend_loc = 'upper right', b_scl_data = False):
        
        n_acq = acq_val_all.shape[1]
        
        if b_scl_data:
            acq_val_2plt = np.zeros(acq_val_all.shape)
            
            for i in range(n_acq):
                min_data = np.min(acq_val_all[:,i])
                max_data = np.max(acq_val_all[:,i])
                acq_val_2plt[:,i] = (acq_val_all[:,i] - min_data) / np.max((1e-8, (max_data - min_data)))
        else:
            acq_val_2plt = acq_val_all
        
        ax.grid(True)
        ax.tick_params(axis = 'both', labelsize = self.fs_ticks)

        for i in range(n_acq):
            ax.plot(x_vec, acq_val_2plt[:,i], '-', color = color_vec[i], label = str_acq_vec[i])
            
        ax.set_xlabel(r'$x$', fontsize = self.fs_axis)
        
        if legend_loc == 'out':
            ax.legend(fontsize = self.fs_legend, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend(fontsize = self.fs_legend, loc='upper left')
    