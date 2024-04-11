#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:38:30 2024

@author: andremarchildon
"""

import time 
import numpy as np
from scipy.optimize import Bounds
from smt.sampling_methods import LHS

class GpHparaX0:
    
    def select_hp_optz_x0(self, i_optz, hp_optz_info):
        
        start_time = time.time() # Time to select points from LHS
        
        ''' Select hp_x0 with LHS ''' 
        
        if self.lkd_optz_start_mtd == 'lhs':
            n_x0 = self.optz_n_x0
        elif self.lkd_optz_start_mtd == 'hp_best':
            n_x0 = self.lkd_hp_best_n_eval
        else:
            raise Exception(f'Unknown lkd_optz_start_mtd: {self.lkd_optz_start_mtd}')
            
        hp_x0, optz_bound = self.get_hp_x0_lhs_median(i_optz, hp_optz_info, n_x0)
        
        ''' Evaluate the likelihood at all starting points '''
        
        if self.lkd_optz_start_mtd == 'hp_best':
            ln_lkd_all = np.full(n_x0, np.nan)
            cond_all   = np.full(n_x0, np.nan)
            
            calc_cond = False if self.wellcond_mtd == 'precon' else True
            
            for i in range(n_x0):
                hp_vals = self.hp_vec2dataclass(self.hp_info_optz_lkd, hp_x0[i,:])
                lkd_info, b_chofac_good = self.calc_lkd_all(hp_vals, calc_cond = calc_cond)
                    
                if b_chofac_good:
                    ln_lkd_all[i] = lkd_info.ln_lkd 
                    cond_all[i]   = lkd_info.cond
            
            if calc_cond:
                bvec_cond_no_good = cond_all > (1.2*self.cond_max)
                
                # Esure theere is at least one valid initial solution
                if np.sum(bvec_cond_no_good) == bvec_cond_no_good.size:
                    idx = np.nanargmin(cond_all) 
                    bvec_cond_no_good[idx] = False
                
                ln_lkd_all[bvec_cond_no_good] = np.nan
            
            # Use hyperparameter with highest marginal log likelihood to start optz
            idx_max = np.nanargmax(ln_lkd_all)
            hp_x0   = hp_x0[idx_max, :][None,:]
        
        ''' Return hp_x0 '''
        
        time_pick_hp0 = time.time() - start_time    
        
        return hp_x0, optz_bound, time_pick_hp0
        
    def get_hp_x0_lhs_median(self, i_optz, hp_optz_info, n_x0):
        
        ''' Preliminary '''
        
        idx_min    = int(np.max((0, i_optz - self.hp_median_n_idx)))
        idx_max    = i_optz
        lhs_factor = self.hp_lhs_bound_factor
        box_factor = self.hp_box_bound_factor
        
        ''' Get lb and ub for LHS and box constraints '''
        
        n_hp = hp_optz_info.n_hp
        para_med_all = np.full(n_hp, np.nan)
        
        lhs_lb       = np.full(n_hp, np.nan)
        lhs_ub       = np.full(n_hp, np.nan)
        
        box_lb       = np.full(n_hp, np.nan)
        box_ub       = np.full(n_hp, np.nan)
        
        if hp_optz_info.has_theta:
            para_med = np.median(self.hp_theta_all[idx_min:idx_max, :], axis=0)
            para_med = np.maximum(para_med, self.hp_theta_range[0])
            para_med = np.minimum(para_med, self.hp_theta_range[1])
            
            para_med_all[hp_optz_info.idx_theta] = para_med
            
            lhs_lb[hp_optz_info.idx_theta] = np.maximum(para_med / lhs_factor, self.hp_theta_range[0])
            lhs_ub[hp_optz_info.idx_theta] = np.minimum(para_med * lhs_factor, self.hp_theta_range[1])
            
            box_lb[hp_optz_info.idx_theta] = np.maximum(para_med / box_factor, self.hp_theta_range[0])
            box_ub[hp_optz_info.idx_theta] = np.minimum(para_med * box_factor, self.hp_theta_range[1])
        
        if hp_optz_info.has_kernel:
            para_med = np.median(self.hp_kernel_all[idx_min:idx_max])
            para_med = np.max((para_med, self.hp_kernel_range[0]))
            para_med = np.min((para_med, self.hp_kernel_range[1]))
            
            para_med_all[hp_optz_info.idx_kernel] = para_med
            
            lhs_lb[hp_optz_info.idx_kernel] = np.max((para_med / lhs_factor, self.hp_kernel_range[0]))
            lhs_ub[hp_optz_info.idx_kernel] = np.min((para_med * lhs_factor, self.hp_kernel_range[1]))
            
            box_lb[hp_optz_info.idx_kernel] = np.max((para_med / box_factor, self.hp_kernel_range[0]))
            box_ub[hp_optz_info.idx_kernel] = np.min((para_med * box_factor, self.hp_kernel_range[1]))
        
        if hp_optz_info.has_varK:
            para_med = np.median(self.hp_varK_all[idx_min:idx_max])
            para_med = np.max((para_med, self.hp_varK_range[0]))
            para_med = np.min((para_med, self.hp_varK_range[1]))
            
            para_med_all[hp_optz_info.idx_varK] = para_med
            
            lhs_lb[hp_optz_info.idx_varK] = np.max((para_med / lhs_factor, self.hp_varK_range[0]))
            lhs_ub[hp_optz_info.idx_varK] = np.min((para_med * lhs_factor, self.hp_varK_range[1]))
            
            box_lb[hp_optz_info.idx_varK] = np.max((para_med / box_factor, self.hp_varK_range[0]))
            box_ub[hp_optz_info.idx_varK] = np.min((para_med * box_factor, self.hp_varK_range[1]))
            
        if hp_optz_info.has_var_fval:
            para_med = np.max((self.hp_var_fval_range[0], np.median(self.hp_var_fval_all[idx_min:idx_max])))
            para_med = np.max((para_med, self.hp_var_fval_range[0]))
            para_med = np.min((para_med, self.hp_var_fval_range[1]))
            
            para_med_all[hp_optz_info.idx_var_fval] = para_med
            
            lhs_lb[hp_optz_info.idx_var_fval] = np.max((para_med / lhs_factor, self.hp_var_fval_range[0]))
            lhs_ub[hp_optz_info.idx_var_fval] = np.min((para_med * lhs_factor, self.hp_var_fval_range[1]))
            
            box_lb[hp_optz_info.idx_var_fval] = np.max((para_med / box_factor, self.hp_var_fval_range[0]))
            box_ub[hp_optz_info.idx_var_fval] = np.min((para_med * box_factor, self.hp_var_fval_range[1]))
            
        if hp_optz_info.has_var_fgrad:
            para_med = np.max((self.hp_var_fgrad_range[0], np.median(self.hp_var_fgrad_all[idx_min:idx_max])))
            para_med = np.max((para_med, self.hp_var_fgrad_range[0]))
            para_med = np.min((para_med, self.hp_var_fgrad_range[1]))
            
            para_med_all[hp_optz_info.idx_var_fgrad] = para_med
            
            lhs_lb[hp_optz_info.idx_var_fgrad] = np.max((para_med / lhs_factor, self.hp_var_fgrad_range[0]))
            lhs_ub[hp_optz_info.idx_var_fgrad] = np.min((para_med * lhs_factor, self.hp_var_fgrad_range[1]))
            
            box_lb[hp_optz_info.idx_var_fgrad] = np.max((para_med / box_factor, self.hp_var_fgrad_range[0]))
            box_ub[hp_optz_info.idx_var_fgrad] = np.min((para_med * box_factor, self.hp_var_fgrad_range[1]))
        
        ''' Create LHS and box constraints '''
        
        # Some hyperparameters are optimized with their log values
        bvec = hp_optz_info.bvec_log_optz
        
        lhs_lb[bvec] = np.log10(lhs_lb[bvec])
        lhs_ub[bvec] = np.log10(lhs_ub[bvec])
        
        box_lb[bvec] = np.log10(box_lb[bvec])
        box_ub[bvec] = np.log10(box_ub[bvec])
        
        if np.any(lhs_lb > lhs_ub):
            print(f'lhs_lb = {lhs_lb}')
            print(f'lhs_ub = {lhs_ub}')
            raise Exception('Invalid bounds for lhs')
            
        if np.any(box_lb > box_ub):
            print(f'box_lb = {box_lb}')
            print(f'box_ub = {box_ub}')
            raise Exception('Invalid bounds for box')
        
        hp_optz_bounds = Bounds(box_lb, box_ub, keep_feasible=True)
        
        if lhs_lb.size == 1:
            # Vector of length n_x0 without nodes at the boundaries
            hp_x0       = np.linspace(lhs_lb[0], lhs_ub[0], n_x0+2)[1:-1,None]
        else:
            para_limit  = np.array([lhs_lb, lhs_ub]).T
            sampling    = LHS(xlimits=para_limit, random_state = 1)
            hp_x0       = sampling(n_x0) 
        
        return hp_x0, hp_optz_bounds

    # def get_hp_optz_x0_theta_diag(self, hp_optz_info, n_x0, magn_rand = 2.0, 
    #                               cstr_w_old_hp = False, i_optz = None):
    #     '''
    #     Select n_x0 initial points for the optimization of the hyperparameters
        
    #     Parameters
    #     ----------
    #     hp_optz_info : HparaOptzInfo from the file GpHparaOptz
    #         Holds info on what hyperparameters to optimize numerically.
    #     n_x0 : int
    #         Number of initial data points to get for the numerical optimization 
    #         of the hyperparameters.
    #     magn_rand : float, optional
    #         All hyperparameters theta are initiated to have the same value. 
    #         To have some variance in the initial solution use magn_rand > 0. 
    #         The default is 2.
    #     cstr_w_old_hp : bool, optional
    #         If set to True the initial value of the hyperparameter are 
    #         constrained to be around a range of the previously optimized 
    #         hyperparameter. The default is False.
    #     i_optz : int, optional
    #         Iteration for the optimization of the hyperparameters. 
    #         Must be provided if cstr_w_old_hp is True. The default is None.

    #     Returns
    #     -------
    #     hp_x0 : 2D numpy array of floats
    #         Each row is the initial solution for the optimization of the hyperparameters.
    #     hp_optz_bounds : scipy.optimize.Bounds
    #         Bound constraints for hp_x0
    #     '''
        
    #     def make_latin_hypercube(lb_vec, ub_vec):
            
    #         if lb_vec.size == 1:
    #             # Vector of length n_x0 without nodes at the boundaries
    #             hp_x0       = np.linspace(lb_vec[0], ub_vec[0], n_x0+2)[1:-1,None]
    #         else:
    #             para_limit  = np.array([lb_vec, ub_vec]).T
    #             sampling    = LHS(xlimits=para_limit, random_state = 1)
    #             hp_x0       = sampling(n_x0) 
            
    #         return hp_x0
            
    #     para_min, para_max, hp_optz_bounds \
    #         = self.get_hp_bounds(hp_optz_info, cstr_w_old_hp = cstr_w_old_hp, 
    #                               i_optz = i_optz)
        
    #     if hp_optz_info.has_theta:
    #         idx_theta  = hp_optz_info.idx_theta
    #         min_idx_th = np.min(idx_theta)
            
    #         # Same initial solution is used for all values of theta
    #         # Only have one value of theta as a free parameter
    #         bvec               = np.ones(hp_optz_info.n_hp, dtype=bool)
    #         bvec[idx_theta]    = False
    #         bvec[min_idx_th]   = True
        
    #         para_min_mod       = para_min[bvec]
    #         para_max_mod       = para_max[bvec]
    #         hp_x0_init         = make_latin_hypercube(para_min_mod, para_max_mod)
            
    #         hp_x0              = np.zeros((n_x0, hp_optz_info.n_hp))
    #         hp_x0[:,bvec]      = hp_x0_init
            
    #         ''' Add randomness to theta around the diagonal th_1 = ... = th_d '''
            
    #         th_mean = hp_x0_init[:,min_idx_th]
    #         rng     = np.random.default_rng(seed=42)
    #         th_all  = th_mean[:,None] + rng.normal(0, magn_rand, (n_x0, self.dim))
            
    #         # Enfource bounds for theta
    #         th_min = para_min[min_idx_th]
    #         th_max = para_max[min_idx_th]
    #         th_all[th_all < th_min] = th_min
    #         th_all[th_all > th_max] = th_max
            
    #         hp_x0[:,idx_theta] = th_all
    #     else:
    #         hp_x0 = make_latin_hypercube(para_min, para_max, n_x0)
        
    #     return hp_x0, hp_optz_bounds        
    
    # def get_hp_best_init(self, i_optz):
    #     '''
    #     Evaluate the marginal log-likelihood at self.lkd_hp_best_n_eval different 
    #     hyperparameter values and identify the one that provides the maximium 
    #     likelihood. This is then the point that is returned and where the 
    #     optimization of the hyperparameters is initiated.
        
    #     Parameters
    #     ----------
    #     i_optz : int
    #         Iteration for the optimization of the hyperparameters. 

    #     Returns
    #     -------
    #     hp_best : 1D numpy array of floats
    #         The starting point for the optimization of the hyperparameters.
    #     hp_optz_bounds : scipy.optimize.Bounds
    #         Bound constraints for hp_best
    #     '''
        
    #     calc_cond = False if self.wellcond_mtd == 'precon' else True
        
    #     n_cases = self.lkd_hp_best_n_eval
    #     hp_x0, optz_bound = self.get_hp_optz_x0_theta_diag(self.hp_info_optz_lkd, n_cases, 
    #                                                        cstr_w_old_hp = True, i_optz = i_optz)
        
    #     ln_lkd_all = np.full(n_cases, np.nan)
    #     cond_all   = np.full(n_cases, np.nan)
        
    #     for i in range(n_cases):
    #         hp_vals = self.hp_vec2dataclass(self.hp_info_optz_lkd, hp_x0[i,:])
    #         lkd_info, b_chofac_good = self.calc_lkd_all(hp_vals, calc_cond = calc_cond)
                
    #         if b_chofac_good:
    #             ln_lkd_all[i] = lkd_info.ln_lkd 
    #             cond_all[i]   = lkd_info.cond
        
    #     if calc_cond:
    #         bvec_cond_no_good = cond_all > (1.2*self.cond_max)
            
    #         # Esure theere is at least one valid initial solution
    #         if np.sum(bvec_cond_no_good) == bvec_cond_no_good.size:
    #             idx = np.nanargmin(cond_all) 
    #             bvec_cond_no_good[idx] = False
            
    #         ln_lkd_all[bvec_cond_no_good] = np.nan
        
    #     # Use hyperparameter with highest marginal log likelihood to start optz
    #     idx_max = np.nanargmax(ln_lkd_all)
    #     hp_best = hp_x0[idx_max, :][None,:]
        
    #     return hp_best, optz_bound