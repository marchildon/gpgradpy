#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:48:36 2021

@author: andremarchildon
"""

import numpy as np
from tabulate import tabulate

class GpInfo:
    
    '''
    Print info about the Gaussian process either to the screen or to a text file
    '''
    
    _gp_floatfmt = '.4e'

    def get_txt_info_surr(self, print2screen = None, save2file = None):

        if print2screen is None: print2screen = self.print_txt_data
        if save2file    is None: save2file    = self.save_data_txt

        # Get all the text data
        data2print  = '\nData from the Gaussian process \n ----------------------- \n\n'
        data2print += self.info_surr_options(False)
        data2print += self.info_surr_all_data(False)

        if print2screen:
            print(data2print)

        if save2file:
            self.save_txt_data(data2print, self.path_data_surr_txt, self.path_data_surr_old_txt)

        return data2print

    def info_surr_all_data(self, print2screen=True):
        
        data2print  = self.info_surr_summary(False)
        data2print += self.info_surr_theta(False)
        data2print += self.info_surr_optz(False)
        
        return data2print

    def info_surr_options(self, print2screen=True):

        data2print = ('\nSurr options \n' + '-'*60 + '\n\n' +
                      'Printing and saving data' +
                      f'\n save_data_npz        = {self.save_data_npz}' +
                      f'\n save_data_txt        = {self.save_data_txt}' +
                      #
                      '\n\nKernel options' +
                      f'\n kernel_type          = {self.kernel_type}' +
                      #
                      '\n\nSub-optimization options' +
                      f'\n optz_log_hp_theta    = {self.optz_log_hp_theta}' +
                      f'\n optz_log_hp_var      = {self.optz_log_hp_var}'  +
                      f'\n optz_log_hp_kernel   = {self.optz_log_hp_kernel}' +
                      f'\n hp_optz_method       = {self.hp_optz_method}'  +
                      f'\n n_surr_optz_start    = {self.n_surr_optz_start}'  +
                      f'\n n_surr_optz_iter     = {self.n_surr_optz_iter}'  +
                      f'\n hp_optz_obj_tol      = {self.hp_optz_obj_tol}'  +
                      f'\n hp_optz_xtol         = {self.hp_optz_xtol}'  +
                      #
                      '\n\nWell-conditioning options' +
                      f'\n wellcond_mtd         = {self.wellcond_mtd}'  + 
                      f'\n set_eta_mtd_dflt     = {self.set_eta_mtd_dflt}'  + 
                      f'\n cond_max             = {self.cond_max:.2e}'  +
                      f'\n cond_max_target      = {self.cond_max_target:.2e}'  +
                      f'\n cond_max_abs         = {self.cond_max_abs:.2e}'  +
                      f'\n condnum_norm         = {self.condnum_norm}'  +
                      f'\n min_nugget_dflt      = {self.min_nugget_dflt}'  +
                      f'\n vreq_frac            = {self.vreq_frac}'  + 
                      #
                      '\n\nMethods for optimizing the hyperparameters' +
                      f'\n n_eval_hp_const      = {self.n_eval_hp_const}' +
                      f'\n mean_fun_type        = {self.mean_fun_type}' +
                      #
                      f'\n b_has_noisy_data     = {self.b_has_noisy_data}' +
                      f'\n b_optz_var_fval      = {self.b_optz_var_fval}' +
                      f'\n b_optz_var_fgrad     = {self.b_optz_var_fgrad}' +
                      #
                      '\n\nRange for the hyperparameters' +
                      f'\n range_theta          = {self.range_theta[0]:.1e} - {self.range_theta[1]:.1e}' +
                      f'\n range_hp_kernel      = {self.range_hp_kernel[0]:.1e} - {self.range_hp_kernel[1]:.1e}' +
                      f'\n range_var_fval       = {self.range_var_fval[0]:.1e} - {self.range_var_fval[1]:.1e}' +
                      f'\n range_var_fgrad      = {self.range_var_fgrad[0]:.1e} - {self.range_var_fgrad[1]:.1e}' +
                      '\n\n')

        if print2screen:
            print(data2print)

        return data2print

    def info_surr_summary(self, print2screen=True):

        max_n_theta_print = 3 # If dim > 3 then theta min, max and mean are printed

        ''' Get the required data '''

        n_iter  = self.hp_varK_all.size
        n_data  = 4 + self.b_optz_var_fval + self.b_optz_var_fgrad \
            + np.min((max_n_theta_print, self.dim)) \
            
        idx_vec = np.arange(n_iter)

        ''' Organize the data '''

        headers   = [''] * n_data
        data      = np.zeros((n_iter, n_data))

        data[:,0] = idx_vec
        data[:,1] = self.hp_kernel_all
        data[:,2] = self.hp_beta_all[:,0] if (self.n_beta_coeff >= 1) else np.nan
        data[:,3] = self.hp_varK_all
        idx       = 4

        headers[:4] = [' ', 'Kernel', 'Mu', 'sig2K']

        if self.b_optz_var_fval:
            data[:,idx]   = self.hp_var_fval_all
            headers[idx] += 'sig fun'
            idx += 1

        if self.b_optz_var_fgrad:
            data[:,idx]   = self.hp_var_fgrad_all
            headers[idx] += 'sig grad'
            idx += 1
            
        if self.dim <= max_n_theta_print: 
            for i in range(self.dim):
                data[:,idx]    = self.hp_theta_all[:,i]
                headers[idx]  += 'theta ' + str(i)
                idx           += 1
        else:
            theta_min          = np.min(self.hp_theta_all,  axis=1)
            theta_max          = np.max(self.hp_theta_all,  axis=1)
            theta_mean         = np.mean(self.hp_theta_all, axis=1)

            data[:,idx]        = theta_min
            data[:,idx+1]      = theta_max
            data[:,idx+2]      = theta_mean
            headers[idx:idx+3] = ['theta min', 'theta max', 'theta mean']
            idx += 3

        ''' Print and return the data '''

        data2print = ('Surrogate hyperparameters  \n'
                      + tabulate((data), headers=headers, tablefmt='orgtbl', 
                                 floatfmt = self._gp_floatfmt)
                      + '\n\n')

        if print2screen:
            print(data2print)

        return data2print

    def info_surr_theta(self, print2screen=True):

        n_iter     = self.hp_theta_all.shape[0]
        idx_vec    = np.arange(n_iter)
        n_col      = self.dim + 1

        data       = np.zeros((n_iter, n_col))
        data[:,0]  = idx_vec
        data[:,1:] = self.hp_theta_all
        
        headers    = [' '] * n_col
        
        cnt = 1

        for i in range(self.dim):
            headers[i+cnt] = 'th' + str(i)

        data2print = ('Surrogate hyperparameter theta \n'
                      + tabulate((data), headers=headers, tablefmt='orgtbl', 
                                 floatfmt = self._gp_floatfmt)
                      + '\n\n')

        if print2screen:
            print(data2print)

        return data2print

    def info_surr_optz(self, print2screen=True):

        n_iter      = self.Kcov_cond_all.size
        idx_vec     = np.arange(n_iter)
        headers     = [' ', 'Time', 'Success (%)', 'Mean i', 'Max i', 'Con good %', 'Cond', 'At cond max?']
        data        = np.array([idx_vec, self.time_hp_optz_all,
                            100*self.hp_optz_success, self.hp_optz_iter_mean, self.hp_optz_iter_max,
                            100*self.hp_optz_con_good, self.Kcov_cond_all, self.Kcov_cond_at_max_all]).T

        data2print  = ('Surrogate optimization \n'
                       + tabulate((data), headers=headers, tablefmt='orgtbl')
                       + '\n\n')
        if print2screen:
            print(data2print)

        return data2print
    