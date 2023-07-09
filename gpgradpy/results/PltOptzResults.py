#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 09:39:12 2023

@author: andremarchildon
"""

import numpy as np 
from os import path
import matplotlib.pyplot as plt

class PltOptzResults:
    
    fs_ticks   = 16
    fs_axis    = 18
    fs_legend  = 12
    fs_text    = 22
    markersize = 10
    
    @staticmethod
    def load_npz_data(case_folder, file_vec, n_iter_max = 1000, n_x0_init = 20, load_noise_free_data = True):
        
        if n_iter_max is None:
            b_clip_iter = True 
            n_iter_max  = 1000 
        else:
            b_clip_iter = False
        
        n_files     = len(file_vec)
        merit_all   = np.full((n_files, n_x0_init, n_iter_max), np.nan)
        opt_all     = np.full((n_files, n_x0_init, n_iter_max), np.nan)
        fsb_all     = np.full((n_files, n_x0_init, n_iter_max), np.nan)

        n_x0_max       = 0
        true_iter_max  = 0

        for i_file in range(n_files):
            fullpath = path.join(case_folder, file_vec[i_file])
            
            if '.npz' not in file_vec[i_file]:
                raise Exception(f'Unknown format for {file_vec[i_file]}')
            
            npzfile = np.load(fullpath)
            
            merit            = npzfile['merit_all']
            n_x0_i, n_iter_i = merit.shape
            n_iter_i         = np.min((n_iter_i, n_iter_max))
            
            if load_noise_free_data:
                merit_all[i_file, :n_x0_i, :n_iter_i] = npzfile['merit_wo_noise_all'][:,:n_iter_i]
                opt_all[i_file, :n_x0_i, :n_iter_i]   = npzfile['opt_wo_noise_all'][:,:n_iter_i]
                fsb_all[i_file, :n_x0_i, :n_iter_i]   = npzfile['fsb_wo_noise_all'][:,:n_iter_i]
            else:
                merit_all[i_file, :n_x0_i, :n_iter_i] = merit[:,:n_iter_i]
                opt_all[i_file, :n_x0_i, :n_iter_i]   = npzfile['opt_all'][:,:n_iter_i]
                fsb_all[i_file, :n_x0_i, :n_iter_i]   = npzfile['fsb_all'][:,:n_iter_i]
            
            n_x0_max      = np.max((n_x0_max, n_x0_i))
            true_iter_max = np.max((true_iter_max, n_iter_i))
            
        if b_clip_iter:
            merit_all = merit_all[:, :n_x0_max, :true_iter_max]
            opt_all   = opt_all[:, :n_x0_max, :true_iter_max]
            fsb_all   = fsb_all[:, :n_x0_max, :true_iter_max]
        else:
            merit_all = merit_all[:, :n_x0_max, :]
            opt_all   = opt_all[:, :n_x0_max, :]
            
        return merit_all, opt_all, fsb_all
    
    @staticmethod
    def best_at_all_iter(data_vec, method='min'):
        
        assert data_vec.ndim == 1, 'Data must be in a 1d array'
        
        n_data = data_vec.size 
        
        best_vec    = np.full(n_data, np.nan) 
        best_vec[0] = data_vec[0]
        
        # Check if there are nan values
        idx_has_nan = np.isnan(data_vec)
        
        if np.any(idx_has_nan):
            if np.sum(idx_has_nan) >= n_data:
                return best_vec
            
            iter_vec = np.arange(n_data)
            n_max    = np.max(iter_vec[np.logical_not(idx_has_nan)]) + 1
            n_max    = np.min((n_data, n_max))
        else:
            n_max    = n_data

        # Find the cummulative min or max        
        if method == 'min':
            for i in range(1,n_max):
                best_vec[i] = np.nanmin((best_vec[i-1], data_vec[i]))
        elif method == 'max':
            for i in range(1,n_max):
                best_vec[i] = np.nanmax((best_vec[i-1], data_vec[i]))
        else:
            raise Exception('Method must be either min or max')
            
        return best_vec
    
    @staticmethod
    def make_base_conv_fig(log_yaxis = True, ylabel   = None,
                           fs_label  = 14,   fs_ticks = 12):
        
        fig, ax = plt.subplots()
        
        if log_yaxis:
            ax.set_yscale('log')
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize = fs_label)
        
        ax.grid(True)
        
        ax.set_xlabel('Iteration', fontsize = fs_label)
        plt.tick_params(axis='both', which = 'major', labelsize = fs_ticks)
        
        return fig, ax
    
    @staticmethod
    def plt_conv_nx0(log_yaxis, ylabel, optz_data_in, 
                     label_vec       = None, 
                     fig_name2save   = None,
                     color_vec       = ['b', 'g', 'm', 'c', 'y'], 
                     xlim            = None,
                     ylim            = None,
                     # marker_vec    = ['o', 'd', 's', 'v', '^', '<', '>', 'P'],
                     markersize      = 3,
                     legend_loc      = None, 
                     fs_legend       = 14, 
                     linewidth       = 0.5, 
                     b_plt_area      = False, 
                     str_data_minmax = 'min'):

        ''' Calculate the best min at each iteration ''' 
        
        if optz_data_in.ndim == 2:
            n_cases = 1
            n_x0, n_iter = optz_data_in.shape 
            optz_data = optz_data_in[None,:,:]
        elif optz_data_in.ndim == 3:
            n_cases, n_x0, n_iter = optz_data_in.shape 
            optz_data = optz_data_in
        else:
            raise Exception(f'optz_data must have 2 or 3 dim but is has shape:{optz_data_in.shape}')
        
        best_optz_data_all = np.zeros(optz_data.shape)
        
        for i_case in range(n_cases):
            for i_x0 in range(n_x0):
                best_optz_data_all[i_case, i_x0, :] = PltOptzResults.best_at_all_iter(optz_data[i_case, i_x0, :], str_data_minmax)
            
        # Shave of lb_all and ub_all is [n_cases, n_iter]
        lb_all = np.min(best_optz_data_all, axis=1)
        ub_all = np.max(best_optz_data_all, axis=1)
        
        ''' Plot '''
        
        fig, ax = PltOptzResults.make_base_conv_fig(log_yaxis, ylabel)
        iter_vec = np.arange(n_iter)
        
        if label_vec is None:
            label_vec = [None] * n_cases
        
        for i_case in range(n_cases):
            color = color_vec[i_case]
            
            if b_plt_area:
                ax.fill_between(iter_vec, lb_all[i_case,:], ub_all[i_case,:], facecolor = color, 
                                alpha = 0.5, interpolate=True, label = label_vec[i_case])
            
            label = None
            for i_x0 in range(n_x0):
                label = label_vec[i_case] if i_x0 == 0 else None
                ax.plot(iter_vec, best_optz_data_all[i_case, i_x0, :], '-', color = color_vec[i_case],
                        marker = '.', markersize = markersize, 
                        linewidth = linewidth, label = label)
            
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if xlim is not None:
            ax.set_xlim(xlim)
            
        if label_vec is not None:
            ax.legend(fontsize = fs_legend, loc = legend_loc)
            
        fig.tight_layout()
        
        if fig_name2save is not None:
            fig.savefig(fig_name2save, dpi = 800, format = 'png')
    
    