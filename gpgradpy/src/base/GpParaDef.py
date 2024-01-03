#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 21:16:09 2021

@author: andremarchildon
"""

import os
import numpy as np

class GpParaDef:
    
    '''
    Initialize, update, save and load data for the Gaussian process
    '''
    
    _save_data = False

    def init_optz_surr(self, n_optz_max):
        
        self._save_data = True
        self.n_optz_max = n_optz_max

        # Hyperparameters for the GP
        self.hp_beta_all            = np.full((n_optz_max, self.n_beta_coeff), np.nan)
        self.hp_varK_all            = np.full(n_optz_max, np.nan)

        self.hp_var_fval_all        = np.full(n_optz_max, np.nan)
        self.hp_var_fgrad_all       = np.full(n_optz_max, np.nan)

        self.hp_kernel_all          = np.full(n_optz_max, np.nan)
        self.hp_theta_all           = np.full((n_optz_max, self.dim), np.nan)

        # Condition number
        self.min_nugget_all         = np.full(n_optz_max, np.nan)
        self.Kcov_cond_all          = np.full(n_optz_max, np.nan)
        self.Kcov_cond_at_max_all   = np.full(n_optz_max, False, dtype=bool)

        # Rescaling and nugget to keep the corr-mat well-conditioned
        self.eta_Kbase_all          = np.full(n_optz_max, np.nan)
        self.eta_Kgrad_all          = np.full(n_optz_max, np.nan) 
        self.vmin_init_all          = np.full(n_optz_max, np.nan)
        self.vmin_req_grad_all      = np.full(n_optz_max, np.nan)
        self.xvec_rescaling_all     = np.full((n_optz_max, self.dim), np.nan)

        # Optimization info
        self.hp_optz_success        = np.full(n_optz_max, np.nan)
        self.hp_optz_iter_mean      = np.full(n_optz_max, np.nan)
        self.hp_optz_iter_max       = np.full(n_optz_max, np.nan)
        self.hp_optz_con_good       = np.full(n_optz_max, np.nan)
        
        # Condition number info for the hp optz 
        self.optz_n_cho_fail_all    = np.full(n_optz_max, np.nan)
        self.optz_n_cond2big_all    = np.full(n_optz_max, np.nan)
        self.optz_max_init_cond_all = np.full(n_optz_max, np.nan)
        
        # Time to optimize the hyper-parameters
        self.time_pick_hp0_all      = np.full(n_optz_max, np.nan)
        self.time_hp_optz_all       = np.full(n_optz_max, np.nan)
        self.time_chofac_all        = np.full(n_optz_max, np.nan)
        
        # Misc
        self.var_fval               = np.full(n_optz_max, np.nan)
        self.varK_var_fval          = np.full(n_optz_max, np.nan)

    def finish_optz_surr(self, n_optz_final):
        
        assert self._save_data, 'If the method init_optz_surr has not been called, then finish_optz_surr cannot be used'

        idx = n_optz_final

        # Hyper-parameters
        self.hp_beta_all            = self.hp_beta_all[:idx,:]
        self.hp_varK_all            = self.hp_varK_all[:idx]

        self.hp_var_fval_all        = self.hp_var_fval_all[:idx]
        self.hp_var_fgrad_all       = self.hp_var_fgrad_all[:idx]

        self.hp_theta_all           = self.hp_theta_all[:idx, :]
        self.hp_kernel_all          = self.hp_kernel_all[:idx]

        # Condition number
        self.min_nugget_all         = self.min_nugget_all[:idx]
        self.Kcov_cond_all          = self.Kcov_cond_all[:idx]
        self.Kcov_cond_at_max_all   = self.Kcov_cond_at_max_all[:idx]

        # Rescaling and nugget to keep the corr-mat well-conditioned
        self.eta_Kbase_all          = self.eta_Kbase_all[:idx]
        self.eta_Kgrad_all          = self.eta_Kgrad_all[:idx]
        self.vmin_init_all          = self.vmin_init_all[:idx]
        self.vmin_req_grad_all      = self.vmin_req_grad_all[:idx]
        self.xvec_rescaling_all     = self.xvec_rescaling_all[:idx]

        # Optimization info
        self.hp_optz_success        = self.hp_optz_success[:idx]
        self.hp_optz_iter_mean      = self.hp_optz_iter_mean[:idx]
        self.hp_optz_iter_max       = self.hp_optz_iter_max[:idx]
        self.hp_optz_con_good       = self.hp_optz_con_good[:idx]

        # Condition number info for the hp optz 
        self.optz_n_cho_fail_all    = self.optz_n_cho_fail_all[:idx]
        self.optz_n_cond2big_all    = self.optz_n_cond2big_all[:idx]
        self.optz_max_init_cond_all = self.optz_max_init_cond_all[:idx]

        # Time
        self.time_pick_hp0_all      = self.time_pick_hp0_all[:idx]
        self.time_hp_optz_all       = self.time_hp_optz_all[:idx]
        self.time_chofac_all        = self.time_chofac_all[:idx]
        
        # Misc
        self.var_fval               = self.var_fval[:idx]
        self.varK_var_fval          = self.varK_var_fval[:idx]

    def load_data_surr(self, all_data=None):
        
        assert self._save_data, 'If the method init_optz_surr has not been called, then load_data_surr cannot be used'
        
        if all_data is None: 
            file2load = self.path_data_acq_npz
            
            if not os.path.isfile(file2load): return 
            
            all_data = np.load(file2load)

        name = self.surr_name
        idx  = all_data[name + 'hp_beta_all'].size

        # The hyper-parameters at each iteration
        self.hp_beta_all[:idx,:]          = all_data[name + 'hp_beta_all']
        self.hp_varK_all[:idx]            = all_data[name + 'hp_varK_all']

        self.hp_var_fval_all[:idx]        = all_data[name + 'hp_var_fval_all']
        self.hp_var_fgrad_all[:idx]       = all_data[name + 'hp_var_fgrad_all']

        self.hp_theta_all[:idx,:]         = all_data[name + 'hp_theta_all']
        self.hp_kernel_all[:idx]          = all_data[name + 'hp_kernel_all']

        # Condition number
        self.min_nugget_all[:idx]         = all_data[name + 'min_nugget_all']
        self.Kcov_cond_all[:idx]          = all_data[name + 'Kcov_cond_all']
        self.Kcov_cond_at_max_all[:idx]   = all_data[name + 'Kcov_cond_at_max_all']

        # Rescaling and nugget to keep the corr-mat well-conditioned
        self.eta_Kbase_all[:idx]          = all_data[name + 'eta_Kbase_all']
        self.eta_Kgrad_all[:idx]          = all_data[name + 'eta_Kgrad_all']
        self.vmin_init_all[:idx]          = all_data[name + 'vmin_init_all']
        self.vmin_req_grad_all[:idx]      = all_data[name + 'vmin_req_grad_all']
        self.xvec_rescaling_all[:idx]     = all_data[name + 'xvec_rescaling_all']

        # Optimization info
        self.hp_optz_success[:idx]        = all_data[name + 'hp_optz_success']
        self.hp_optz_iter_mean[:idx]      = all_data[name + 'hp_optz_iter_mean']
        self.hp_optz_iter_max[:idx]       = all_data[name + 'hp_optz_iter_max']
        self.hp_optz_con_good[:idx]       = all_data[name + 'hp_optz_con_good']

        # Condition number info for the hp optz 
        self.optz_n_cho_fail_all[:idx]    = all_data[name + 'optz_n_cho_fail_all']
        self.optz_n_cond2big_all[:idx]    = all_data[name + 'optz_n_cond2big_all']
        self.optz_max_init_cond_all[:idx] = all_data[name + 'optz_max_init_cond_all']

        # Time
        self.time_pick_hp0_all[:idx]      = all_data[name + 'time_pick_hp0_all']
        self.time_hp_optz_all[:idx]       = all_data[name + 'time_hp_optz_all']
        self.time_chofac_all[:idx]        = all_data[name + 'time_chofac_all']
        
        # Misc
        self.var_fval[:idx]               = all_data[name + 'var_fval']
        self.varK_var_fval[:idx]          = all_data[name + 'varK_var_fval']

    def export_data_surr(self, save2file=True, file2save=None, file2save_old=None):
        
        assert self._save_data, 'If the method init_optz_surr has not been called, then export_data_surr cannot be used'
        
        name = self.surr_name
        
        data2save = {name + 'hp_beta_all'            : self.hp_beta_all,
                     name + 'hp_varK_all'            : self.hp_varK_all,
                     name + 'hp_var_fval_all'        : self.hp_var_fval_all,
                     name + 'hp_var_fgrad_all'       : self.hp_var_fgrad_all,
                     name + 'hp_theta_all'           : self.hp_theta_all,
                     name + 'hp_kernel_all'          : self.hp_kernel_all,
                     #
                     name + 'min_nugget_all'         : self.min_nugget_all,
                     name + 'Kcov_cond_all'          : self.Kcov_cond_all,
                     name + 'Kcov_cond_at_max_all'   : self.Kcov_cond_at_max_all,
                     #
                     name + 'eta_Kbase_all'          : self.eta_Kbase_all,
                     name + 'eta_Kgrad_all'          : self.eta_Kgrad_all,
                     name + 'vmin_init_all'          : self.vmin_init_all,
                     name + 'vmin_req_grad_all'      : self.vmin_req_grad_all,
                     name + 'xvec_rescaling_all'     : self.xvec_rescaling_all,
                     #
                     name + 'hp_optz_success'        : self.hp_optz_success,
                     name + 'hp_optz_iter_mean'      : self.hp_optz_iter_mean,
                     name + 'hp_optz_iter_max'       : self.hp_optz_iter_max,
                     name + 'hp_optz_con_good'       : self.hp_optz_con_good,
                     #
                     name + 'optz_n_cho_fail_all'    : self.optz_n_cho_fail_all,
                     name + 'optz_n_cond2big_all'    : self.optz_n_cond2big_all,
                     name + 'optz_max_init_cond_all' : self.optz_max_init_cond_all,
                     #
                     name + 'time_pick_hp0_all'      : self.time_pick_hp0_all, 
                     name + 'time_hp_optz_all'       : self.time_hp_optz_all, 
                     name + 'time_chofac_all'        : self.time_chofac_all,
                     #
                     name + 'var_fval'               : self.var_fval, 
                     name + 'varK_var_fval'          : self.varK_var_fval
                     }

        if save2file:
            if file2save     is None: file2save     = self.path_data_surr_npz
            if file2save_old is None: file2save_old = self.path_data_surr_old_npz
            
            self.save_npz_data(data2save, file2save, file2save_old)
                
        return data2save

    def store_new_para_surr(self, i_optz, hp_vals, 
                            surr_optz_info = None,   cond_val    = np.nan, 
                            time_hp_optz   = np.nan, time_chofac = np.nan, time_pick_hp0 = np.nan):
        
        self.hp_vals = hp_vals
        
        if self._save_data is False:
            return

        idx = i_optz
        
        self.time_hp_optz_all[idx]      = time_hp_optz
        self.time_chofac_all[idx]       = time_chofac
        self.time_pick_hp0_all[idx]     = time_pick_hp0

        self.hp_beta_all[idx,:]         = hp_vals.beta
        self.hp_theta_all[idx,:]        = hp_vals.theta
        self.hp_kernel_all[idx]         = hp_vals.kernel
        
        self.hp_varK_all[idx]           = hp_vals.varK
        self.hp_var_fval_all[idx]       = hp_vals.var_fval
        self.hp_var_fgrad_all[idx]      = hp_vals.var_fgrad

        # Condition number
        if self.use_grad:
            self.min_nugget_all[idx]    = self._eta_Kgrad
        else:
            self.min_nugget_all[idx]    = self._eta_Kbase
            
        self.Kcov_cond_all[idx]         = cond_val
        self.Kcov_cond_at_max_all[idx]  = cond_val >= (0.99 * self.cond_max)
        
        # Rescaling and nugget to keep the corr-mat well-conditioned
        self.eta_Kbase_all[idx]         = self._eta_Kbase
        self.eta_Kgrad_all[idx]         = self._eta_Kgrad
        self.vmin_init_all[idx]         = self._vmin_init
        self.vmin_req_grad_all[idx]     = self._vmin_req_grad
        
        if self.b_use_data_scl:
            self.xvec_rescaling_all[idx] = self.DataScl.xvec_scale

        # Optimization info 
        if surr_optz_info is not None:

            if 'hp_optz_success'   in surr_optz_info:
                self.hp_optz_success[idx]    = surr_optz_info['hp_optz_success']

            if 'hp_optz_iter_mean' in surr_optz_info:
                self.hp_optz_iter_mean[idx]  = surr_optz_info['hp_optz_iter_mean']

            if 'hp_optz_iter_max'  in surr_optz_info:
                self.hp_optz_iter_max[idx]   = surr_optz_info['hp_optz_iter_max']
                
            if 'hp_optz_con_good'  in surr_optz_info:
                self.hp_optz_con_good[idx]   = surr_optz_info['hp_optz_con_good']
                
            self.optz_n_cho_fail_all[idx]    = surr_optz_info['optz_n_cho_fail']
            self.optz_n_cond2big_all[idx]    = surr_optz_info['optz_n_cond2big']
            self.optz_max_init_cond_all[idx] = surr_optz_info['optz_max_init_cond']
            
        # Misc
        self.var_fval[idx]      = np.var(self._fval_in)
        self.varK_var_fval[idx] = self.hp_varK_all[idx] / self.var_fval[idx]
