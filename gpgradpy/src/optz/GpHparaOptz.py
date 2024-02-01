#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:48:33 2021

@author: andremarchildon
"""

import time 
import numpy as np
from dataclasses import dataclass

from . import OptzLkd
from . import GpHparaCon
from . import GpHparaGrad

@dataclass(frozen=True)
class HparaOptzInfo:
    n_hp:           int      = None
    has_theta:      bool     = False 
    idx_theta:      np.array = None
    has_kernel:     bool     = False 
    idx_kernel:     np.array = None
    has_varK:       bool     = False 
    idx_varK:       np.array = None 
    has_var_fval:   bool     = False
    idx_var_fval:   np.array = None
    has_var_fgrad:  bool     = False 
    idx_var_fgrad:  np.array = None
    bvec_log_optz:  np.array = None

class GpHparaOptz(OptzLkd, GpHparaCon, GpHparaGrad):
    
    HparaOptzInfo = HparaOptzInfo
    
    def set_hp_optz_info(self, has_theta, 
                         has_kernel   = False, has_varK      = False, 
                         has_var_fval = False, has_var_fgrad = False):
        '''
        Parameters
        ----------
        has_theta : bool
            True if theta is a hyperparameter to be optimized numerically.
        has_kernel : bool, optional
            True if the kernel has a hyperparameter that is to be optimized 
            numerically. The default is False.
        has_varK : bool, optional
            True if varK is a hyperparameter to be optimized numerically. 
            The hyperparameter varK is optimized numerically when there is 
            noise either in the value or the gradient of the function of interest.
            The default is False.
        has_var_fval : bool, optional
            True if the var_fval, ie the variance of the normally distributed 
            noise for the value of the function of interest, is a 
            hyperparameter to be optimized numerically. The default is False.
        has_var_fgrad : bool, optional
            True if the var_fval, ie the variance of the normally distributed 
            noise for the gradient of the function of interest, is a 
            hyperparameter to be optimized numerically. The default is False.

        Returns
        -------
        hp_optz_info : data class of type HparaOptzInfo
            Info on the numerical optimization of the hyperparameters.
        '''
        
        n_hp2optz     = has_theta * self.dim + has_kernel + has_varK + has_var_fval + has_var_fgrad
        bvec_log_optz = np.zeros(n_hp2optz, dtype=bool)
            
        cnt = 0    
        
        if has_theta:
            cnt2        = cnt + self.dim
            idx_theta   = np.arange(cnt, cnt2, dtype=int)
            cnt        += self.dim
            
            if self.optz_log_hp_theta: 
                bvec_log_optz[idx_theta] = 1
        else:
            idx_theta   = np.array([], dtype=int)
        
        if has_kernel:
            assert self.kernel_has_hp, 'Kernel must have hyperaparamters if b_optz_hp_kernel is set to True'
            idx_kernel = np.array([cnt])
            cnt += 1 # Can be generalized if kernel has more hyperparameters
            
            if self.optz_log_hp_kernel: 
                bvec_log_optz[idx_kernel] = 1
        else: 
            idx_kernel = np.array([], dtype=int)
        
        if has_varK:
            idx_varK = cnt
            cnt += 1
            
            if self.optz_log_hp_var: 
                bvec_log_optz[idx_varK] = 1
        else:
            idx_varK = np.array([], dtype=int)
        
        if has_var_fval:
            assert self.known_eps_fval is False, 'var_fval should not be a hyperparameter if known_eps_fval is True'
            idx_var_fval  = cnt
            cnt += 1
            
            if self.optz_log_hp_var: 
                bvec_log_optz[idx_var_fval] = 1
        else:
            idx_var_fval  = np.array([], dtype=int)
            
        if has_var_fgrad:
            assert self.known_eps_fgrad is False, 'var_fgrad should not be a hyperparameter if known_eps_fgrad is True'
            idx_var_fgrad = cnt
            cnt += 1
            
            if self.optz_log_hp_var: 
                bvec_log_optz[idx_var_fgrad] = 1
        else:
            idx_var_fgrad = np.array([], dtype=int)

        hp_optz_info \
            = self.HparaOptzInfo(n_hp          = n_hp2optz, 
                                 has_theta     = has_theta,     idx_theta     = idx_theta, 
                                 has_kernel    = has_kernel,    idx_kernel    = idx_kernel, 
                                 has_varK      = has_varK,      idx_varK      = idx_varK, 
                                 has_var_fval  = has_var_fval,  idx_var_fval  = idx_var_fval, 
                                 has_var_fgrad = has_var_fgrad, idx_var_fgrad = idx_var_fgrad, 
                                 bvec_log_optz = bvec_log_optz)
        
        return hp_optz_info
    
    def optz_hp(self, i_optz):
        '''
        1) Gets the starting points for the optimization
        2) Performs the numerical optimization 
        3) Store the new hyperparameter 

        Parameters
        ----------
        i_optz : int
            Iteration for the optimization of the hyperparameters.
        '''
        
        if self.n_eval <= self.hp_const_n_eval:
            hp_vals        = self.get_init_hp_vals()
            surr_optz_info = None
            Rtensor        = self.get_scl_x_w_dist()[1]
            cond_val       = self.calc_all_K_w_chofac(Rtensor, hp_vals, calc_chofac = False, calc_cond = True)[4]
            time_hp_optz   = time_chofac = time_pick_hp0 = 0
        else:
            self._time_chofac = 0
            
            if ('rescale' in self.wellcond_mtd) and (self.cond_vreq_max_iter > 1):
                time_pick_hp0 = 0 
                
                start_time = time.time() 
                hp_vec, cond_val, surr_optz_info = self.optz_hp_max_lkd_mtd_rescale(i_optz)
                time_hp_optz = time.time() - start_time
            else:
                if self.lkd_optz_start_mtd == 'lhs':
                    # Select points from a lhs
                    start_time = time.time() 
                    hp_x0, optz_bound = self.get_hp_optz_x0(self.hp_info_optz_lkd, self.optz_n_x0)
                    time_pick_hp0 = time.time() - start_time
                    
                    # Perform optiization starting at all points in hp_x0
                    start_time = time.time() 
                    hp_vec, cond_val, surr_optz_info = self.optz_hp_max_lkd(hp_x0, optz_bound)
                    time_hp_optz = time.time() - start_time
                    
                elif self.lkd_optz_start_mtd == 'hp_best':
                    # Create several hp_val and select 1 to start the optz 
                    start_time = time.time() 
                    hp_best_init, optz_bound = self.get_hp_best_init(i_optz)
                    time_pick_hp0 = time.time() - start_time
                    
                    # Perform optimization starting at hp_best_init
                    start_time = time.time() 
                    hp_vec, cond_val, surr_optz_info = self.optz_hp_max_lkd(hp_best_init, optz_bound)
                    time_hp_optz = time.time() - start_time
                else:
                    raise Exception(f'Unknown option lkd_optz_start_mtd = {self.lkd_optz_start_mtd}')
                    
            time_chofac = self._time_chofac
            
            hp_vals = self.hp_vec2dataclass(self.hp_info_optz_lkd, hp_vec)
            
            # Optimize closed form hyperparameter and add them to hp_vals
            hp_vals = self.optz_closed_form_hp(hp_vals)
                
        self.store_new_para_surr(i_optz, hp_vals, surr_optz_info, cond_val, 
                                 time_hp_optz, time_chofac, time_pick_hp0)

    def setup_hp_idx4optz(self):
        
        # Set options for optimization with maximum absolute likelihood
        b_optz_theta    = True
        b_optz_hp_kern  = self.b_optz_hp_kernel and self.kernel_has_hp
        b_optz_varK     = self.b_has_noisy_data # varK has a closed form solution if there is no noise
        
        self.hp_info_optz_lkd \
            = self.set_hp_optz_info(b_optz_theta, b_optz_hp_kern, b_optz_varK, 
                                    self.b_optz_var_fval, self.b_optz_var_fgrad)

    def get_init_hp_vals(self):
        
        theta = self.hp_theta_init * np.ones(self.dim)
        fval_scl, std_fval_scl, fgrad_scl, std_fgrad_scl = self.get_scl_eval_data()
        
        beta = np.zeros(self.n_beta_coeff)
        
        if self.n_beta_coeff > 0:
            beta[0] = np.mean(fval_scl)

        hp_var_fval = None if self.known_eps_fval else self.hp_var_fval_init
        
        if (self.use_grad is False) or self.known_eps_fgrad:
            hp_var_fgrad = None 
        else:
            hp_var_fgrad = self.hp_var_fgrad_init
            
        hp_vals = self.make_hp_class(beta, theta, self.hp_kernel_default, 
                                     self.hp_varK_init, hp_var_fval, hp_var_fgrad)
        
        return hp_vals
            
    def optz_closed_form_hp(self, hp_vals):
        
        lkd_info, b_chofac_good = self.calc_lkd_all(hp_vals, calc_lkd = False, 
                                                    calc_cond = False, calc_grad = False)
            
        hp_vals.beta = lkd_info.hp_beta 
        
        if self.b_has_noisy_data is False:
            hp_vals.varK = lkd_info.hp_varK
            
        return hp_vals
    