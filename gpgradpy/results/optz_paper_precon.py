#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:30:32 2023

@author: andremarchildon
"""

import numpy as np 
from os import path

from PltOptzResults import PltOptzResults

'''
This file plots the optimization results for the paper "A Solution to the 
Ill-Conditioning of Gradient- Enhanced Covariance Matrices for Gaussian Processes"
'''

markersize  = 3
color_vec   = ['m', 'b', 'g']

b_plt_merit = False
b_plt_opt   = True
b_plt_fsb   = False

ylim_obj    = [1e-28, 1e6]
ylim_opt    = [1e-16, 1e6]

def plot_obj_opt(case_folder, file_vec, label_vec, basefigname, n_iter_max = None, legend_loc = 'upper right'):
    
    merit_all, opt_all, fsb_all = PltOptzResults.load_npz_data(case_folder, file_vec, n_iter_max, load_noise_free_data = True)
    
    xlim      = None
    log_yaxis = True
    
    max_fsb = np.nanmax(fsb_all)
    if (np.isnan(max_fsb) == False) and (max_fsb > 0):
        str_obj = 'Merit'
        str_ext = '_Merit.png'
    else:
        str_obj = 'Objective'
        str_ext = '_Obj.png'

    if b_plt_merit:
        fig_name2save = path.join(case_folder, basefigname + str_ext)
        PltOptzResults.plt_conv_nx0(log_yaxis, str_obj,  merit_all, label_vec, 
                              fig_name2save, color_vec, xlim, ylim_obj,
                              markersize = markersize, 
                              legend_loc = legend_loc)

    if b_plt_opt:
        fig_name2save = path.join(case_folder, basefigname + '_Opt.png')
        PltOptzResults.plt_conv_nx0(log_yaxis, 'Optimality', opt_all,   label_vec, 
                             fig_name2save, color_vec, xlim, ylim_opt, 
                             markersize = markersize, 
                             legend_loc = legend_loc)

# Set case to plot
base_folder    = 'data_paper_precon'
list_test_case = ['Rosen_a10_d2', 'Rosen_a10_d5', 'Rosen_a10_d10', 'Rosen_a10_d15']

''' Indicate files and other case specific parameters '''

file_vec    = ['Baye_Kern_SE_n500_Grad_T_None_all.npz', 
               'Baye_Kern_SE_n500_Grad_T_Vreq_all.npz', 
               'Baye_Kern_SE_n500_Grad_T_Precon_all.npz']

label_vec   = ['Baseline', 'Rescaling', 'Precondition']

n_cases = len(list_test_case)

for i_case in range(n_cases):
    str_test_case = list_test_case[i_case]
    case_folder   = path.join(base_folder, str_test_case)
    basefigname   = str_test_case

    if str_test_case == 'Rosen_a10_d2':
        n_iter_max  = 160
        plot_obj_opt(case_folder, file_vec, label_vec, basefigname, n_iter_max, legend_loc = 'upper right')
    elif str_test_case == 'Rosen_a10_d5':
        n_iter_max  = 160
        plot_obj_opt(case_folder, file_vec, label_vec, basefigname, n_iter_max, legend_loc = 'upper right')
    elif str_test_case == 'Rosen_a10_d10':
        n_iter_max  = 160
        plot_obj_opt(case_folder, file_vec, label_vec, basefigname, n_iter_max, legend_loc = 'upper right')
    elif str_test_case == 'Rosen_a10_d15':
        n_iter_max  = 160
        plot_obj_opt(case_folder, file_vec, label_vec, basefigname, n_iter_max, legend_loc = 'lower left')
    else:
        raise Exception(f'Unknown test case str_test_case = {str_test_case}')
