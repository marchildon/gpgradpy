#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:02:23 2022

@author: andremarchildon
"""

import numpy as np
from scipy.optimize import minimize

from . import CalcLkd

class OptzLkd(CalcLkd):
    
    def calc_store_likelihood(self, hp_vec, always_calc_cond = False):
        '''
        -Evaluate the following:
            the marginal log-likelihood (lkd)
            the condition number if either self.b_use_cond_cstr or always_calc_cond are True, 
            the gradients of the likelihood and condition number (if needed)
        -Save the last calculations and only calculate if hp_vec does not 
        match self._last_hp_vec

        Parameters
        ----------
        hp_vec : 1d numpy array
            Vector with hyperparameters that are being optimized numerically.
        always_calc_cond : bool, optional
            Set to True to ensure the condition number is calculated. 
            The default is False.

        Returns
        -------
        ln_lkd_val : float
            marginal log-likelihood.
        ln_lkd_grad : 1d numpy array
            DESCRIPTION.
        cond_val : float
            condition number of the covariance matrix.
        cond_grad : 1d numpy array
            gradient of cond_val.
        '''
        
        hp_vec = np.atleast_1d(hp_vec).ravel() 
        
        # Check if the likelihood and its gradient have already been calculated
        if not(np.array_equal(hp_vec, self._last_hp_vec)):
            
            hp_vals   = self.hp_vec2dataclass(self.hp_info_optz_lkd, hp_vec)
            calc_cond = self.b_use_cond_cstr or always_calc_cond
            
            lkd_info, b_chofac_good \
                = self.calc_lkd_all(hp_vals, calc_lkd = True, 
                                    calc_cond = calc_cond, calc_grad = True)
            
            cond_val    = lkd_info.cond
            cond_grad   = lkd_info.cond_grad
            
            if b_chofac_good:
                ln_lkd_val  = lkd_info.ln_lkd 
                ln_lkd_grad = lkd_info.ln_lkd_grad
                
                # Adjust gradients for terms being optimized with respect to their log values
                bvec            = self.hp_info_optz_lkd.bvec_log_optz
                log_hp          = hp_vec[bvec]
                transformation  = 10**log_hp * np.log(10)
                
                ln_lkd_grad[bvec] *= transformation
                
                if self.b_use_cond_cstr:
                    cond_grad[bvec] *= transformation
            else:
                # If the Cholesky decomposition fails use the condition number as the obj
                ln_lkd_val  = -cond_val
                ln_lkd_grad = -cond_grad
                
            # Store values
            self._lkd_val   = ln_lkd_val
            self._lkd_grad  = ln_lkd_grad
            
            self._cond_val  = cond_val
            self._cond_grad = cond_grad
            
        return self._lkd_val, self._lkd_grad, self._cond_val, self._cond_grad
    
    ''' Optmization objective and gradient '''
    
    def return_optz_val(self, hp_vec):
        
        hp_vec         = np.atleast_1d(hp_vec).ravel() 
        likelihood_val = self.calc_store_likelihood(hp_vec)[0]
        return -likelihood_val # Negative value is returned since we are using a minimization algorithm

    def return_optz_grad(self, hp_vec):
        
        hp_vec          = np.atleast_1d(hp_vec).ravel() 
        likelihood_grad = self.calc_store_likelihood(hp_vec)[1]
        return -likelihood_grad # Negative value is returned since we are using a minimization algorithm
    
    ''' Nonelinear condition number constraint '''
    
    def return_cond_val(self, hp_vec):
        
        hp_vec  = np.atleast_1d(hp_vec).ravel() 
        cond_val = self.calc_store_likelihood(hp_vec)[2]
        return cond_val

    def return_cond_grad(self, hp_vec):
        
        hp_vec    = np.atleast_1d(hp_vec).ravel() 
        cond_grad = self.calc_store_likelihood(hp_vec)[3]
        return cond_grad

    def optz_hp_max_lkd_mtd_rescale(self, i_optz):
        '''
        Parameters
        ----------
        i_optz : int
            Index for the optimization of the Bayesian optimizer.

         Returns
         -------
         best_hp : 1d numpy array
             Optimized hyperparameters with the largest marginal-log 
             likelihhod satisfying the constraints.
         cond_val : float
             Condition number of the original or preconditioned covariance 
             matrix evaluated with best_hp.
         surr_optz_info : dict
             Info on the optimization, see below for entries.
        '''
        
        assert 'rescale' in self.wellcond_mtd, f'Method should only be called for the rescale methods, but it is wellcond_mtd = {self.wellcond_mtd}'
        
        ''' Do optimization '''
        
        hp_x0, optz_bound = self.get_hp_optz_x0(self.hp_info_optz_lkd, self.optz_n_x0)
        
        best_hp, cond_val, surr_optz_info = self.optz_hp_max_lkd(hp_x0, optz_bound)
        
        max_iter      = self.cond_vreq_max_iter
        theta_all     = np.full((max_iter, self.dim), np.nan)
        dist2line_all = np.full(max_iter, np.nan)
        scale_vec_all = np.full((max_iter, self.dim), np.nan)
        
        if self.n_eval > 1:
            cnt      = 0
            not_done = True
            
            theta_new_xo = best_hp[self.hp_info_optz_lkd.idx_theta]
            
            while not_done:
                x_scl      = self.DataScl.x_scl
                xvec_scale = self.DataScl.xvec_scale
                
                theta_new_xo, est_dist2sol, xvec_scale_new = self.rescaling_data_w_theta_sol(x_scl, xvec_scale, theta_new_xo)
                
                theta_all[cnt,:]     = theta_new_xo
                dist2line_all[cnt]   = est_dist2sol
                scale_vec_all[cnt,:] = xvec_scale_new
                
                if (cnt == (max_iter - 1)) or (est_dist2sol < self.cond_vreq_iter_tol):
                    not_done = False
                else:
                    x0_hp_vec = np.copy(best_hp)
                    x0_hp_vec[self.hp_info_optz_lkd.idx_theta] = theta_new_xo
                    
                    best_hp,cond_val = self.optz_hp_max_lkd(x0_hp_vec, optz_bound)[:2]
                
                cnt += 1
                
            idx_min_dist    = np.nanargmin(dist2line_all)
            theta_final     = theta_all[idx_min_dist, :]
            scale_vec_final = scale_vec_all[idx_min_dist, :]
            
            self.DataScl.set_xscale_data(xvec_scale_in = scale_vec_final)
            
            best_hp_final = np.copy(best_hp)
            best_hp_final[self.hp_info_optz_lkd.idx_theta] = theta_final
        else:
            best_hp_final = best_hp
        
        return best_hp_final, cond_val, surr_optz_info

    def optz_hp_max_lkd(self, hp_x0_all, optz_bound):
        '''
        Primary optimizer for the numerical optimization of the hyperparameters 
        to maximize the marginal log-likelihood, with possible constrain on 
        the condition number of the covariance matrix.
        
        Parameters
        ----------
        hp_x0_all : 2d numpy array 
            Each row is the starting point for the numerical optimization.
        optz_bound : scipy.optimize.Bounds 
            Box constraints for the optimization.

        Returns
        -------
        best_hp : 1d numpy array
            Optimized hyperparameters with the largest marginal-log 
            likelihhod satisfying the constraints.
        cond_val : float
            Condition number of the original or preconditioned covariance 
            matrix evaluated with best_hp.
        surr_optz_info : dict
            Info on the optimization, see below for entries.
        '''
        
        # For this optz problem 'SLSQP' work better and is faster than 'trust-constr'
        if self.optz_mtd == 'SLSQP':
            optz_opt = {'ftol'      : self.optz_tol_obj,
                        'eps'       : self.optz_tol_x,
                        'maxiter'   : self.optz_iter_max,
                        'disp'      : False}
        
        elif self.optz_mtd == 'trust-constr':
            optz_opt = {'initial_tr_radius' : 0.1,
                        'xtol'              : self.optz_tol_x,
                        'gtol'              : self.optz_tol_obj,
                        'maxiter'           : self.optz_iter_max,
                        'disp'              : False}
        
        ''' Setup arrays to track optz '''
        
        if hp_x0_all.ndim == 1:
            hp_x0_all = hp_x0_all[None,:]
        
        n_optz = hp_x0_all.shape[0]

        all_optz_success   = np.full(n_optz, False, dtype=bool)
        all_total_fun_iter = np.full(n_optz, np.nan)
        all_con_good       = np.full(n_optz, False, dtype=bool)

        optz_obj_all       = np.full(n_optz, np.nan)
        optz_sol_all       = np.full((n_optz, self.hp_info_optz_lkd.n_hp), np.nan)
        optz_cond_all      = np.full(n_optz, np.nan)
        
        # Track data related to the condition number
        optz_n_cho_fail    = 0 # No. of instances when the Cholesky factorization fails
        optz_n_cond2big    = 0
        optz_max_init_cond = np.nan
        
        # Set the nonlinear constraints, if any
        nlc_method = self.condnum_nlc if self.b_use_cond_cstr else []

        ''' Run optimizer '''
        
        for i in range(n_optz):

            x0_i = hp_x0_all[i, :]
            
            ''' Calculate parameters related to the condition number '''
            
            if self.b_use_cond_cstr:
                lkd_val, _, cond_val, = self.calc_store_likelihood(x0_i)[:3]
                optz_max_init_cond    = np.max((optz_max_init_cond, cond_val))
                if np.isnan(lkd_val):        optz_n_cho_fail += 1
                if cond_val > self.cond_max: optz_n_cond2big += 1
            
            ''' Optimization '''

            self._last_hp_vec = np.full((1, x0_i.size), np.nan)

            res = minimize(self.return_optz_val, x0_i,
                           method      = self.optz_mtd,
                           jac         = self.return_optz_grad,
                           bounds      = optz_bound,
                           constraints = nlc_method,
                           options     = optz_opt)
            
            # Store the result of the minimization
            optz_sol_all[i,:]     = res.x
            optz_obj_all[i]       = res.fun
            all_optz_success[i]   = res.success
            all_total_fun_iter[i] = res.nit
            
            if self.b_use_cond_cstr:
                cond_val          = self.return_cond_val(res.x)
                optz_cond_all[i]  = cond_val
                all_con_good[i]   = cond_val < 1.01 * self.cond_max
            else:
                all_con_good[i]   = True
                
            if res.success:
                if all_con_good[i] is False:
                    print('Surr hpara optz: Con FAIL, Optimizer: GOOD')
            else:
                if all_con_good[i]:
                    print(f'Surr hpara optz: Con GOOD, Optimizer: {res.message}')
                else:
                    print(f'Surr hpara optz: Con FAIL, Optimizer: {res.message}')
                
        ''' Identify the best solution '''
            
        if any(all_con_good):
            optz_obj_con_good = optz_obj_all[all_con_good] 
            optz_sol_con_good = optz_sol_all[all_con_good, :] 
        else:
            print('*** No solutions satisfy the constraints for the GP hyperparameter optimization ***')
            print(f'Cond = {optz_cond_all}')
            optz_obj_con_good = optz_obj_all
            optz_sol_con_good = optz_sol_all

        idx_min = np.nanargmin(optz_obj_con_good)
        best_hp = optz_sol_con_good[idx_min, :]

        ''' Set the new hyper parameters and store their values '''

        hp_optz_success   = np.mean(all_optz_success)
        hp_optz_iter_mean = np.mean(all_total_fun_iter)
        hp_optz_iter_max  = np.max(all_total_fun_iter)
        acq_optz_con_good = np.mean(all_con_good)

        surr_optz_info  = {'hp_optz_success'    : hp_optz_success,
                           'hp_optz_iter_mean'  : hp_optz_iter_mean,
                           'hp_optz_iter_max'   : hp_optz_iter_max, 
                           'hp_optz_con_good'   : acq_optz_con_good, 
                           'optz_n_cho_fail'    : optz_n_cho_fail, 
                           'optz_n_cond2big'    : optz_n_cond2big, 
                           'optz_max_init_cond' : optz_max_init_cond}
        
        ''' Calculate the final condition number '''
        
        x_scl, Rtensor = self.get_scl_x_w_dist()
        best_hp_vals = self.hp_vec2dataclass(self.hp_info_optz_lkd, best_hp)
        
        if self.b_has_noisy_data:
            cond_val = self.calc_all_K_w_chofac(Rtensor, best_hp_vals, calc_chofac = False, calc_cond = True)[4]
        else:
            cond_val = self.calc_Kern_w_chofac(Rtensor, best_hp_vals, calc_chofac = False, calc_cond = True)[4]
        
        return best_hp, cond_val, surr_optz_info
