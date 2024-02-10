#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 10:52:48 2021

@author: andremarchildon
"""

from dataclasses import dataclass

import numpy as np
from smt.sampling_methods import LHS


@dataclass
class HparaOptzVal:
    beta:       np.array = None # Coefficients for the mean function
    theta:      np.array = None # Hyperparameters related to the characteristic length for the kernel
    kernel:     float    = None # Extra hyperparameter for the kernel
    varK:       float    = None # Variance of the stationary residual error
    var_fval:   float    = None # Variance of the noise on the value of the function of interest
    var_fgrad:  float    = None # Variance of the noise on the gradient of the function of interest
    
class GpHpara:
    
    # Keeps track of the last point where the acq value, gradient and Hessian 
    # have been calculated. This ensures these parameters do not need to be 
    # calculated again
    _last_hp_vec = None

    def make_hp_class(self, beta = None, theta = None, kernel = None, 
                      varK = None, var_fval = None, var_fgrad = None):
        
        return HparaOptzVal(beta, theta, kernel, varK, var_fval, var_fgrad)
    
    def set_hp_from_idx(self, i_optz):
        
        if np.isnan(self.hp_var_fval_all[i_optz]):
            self.known_eps_fval = True
            hp_var_fval = None
        else:
            self.known_eps_fval = False
            hp_var_fval = self.hp_var_fval_all[i_optz]
            
        if np.isnan(self.hp_var_fgrad_all[i_optz]):
            self.known_eps_fgrad = True
            hp_var_fgrad = None
        else:
            self.known_eps_fgrad = False
            hp_var_fgrad = self.hp_var_fgrad_all[i_optz]
        
        hp_vals = self.make_hp_class(
            self.hp_beta_all[i_optz,:], self.hp_theta_all[i_optz,:], 
            self.hp_kernel_all[i_optz], self.hp_varK_all[i_optz], 
            hp_var_fval,                hp_var_fgrad)
        
        self.hp_vals = hp_vals
        
        
        
    def hp_vec2dataclass(self, hp_optz_info, hp_vec):
        '''
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from the file GpHparaOptz
            Holds info on what hyperparameters to optimize numerically.
        hp_vec : 1D numpy array
            Array with values of the hyperparameters being optimized numerically.

        Returns
        -------
        hp_vals : data class of type HparaOptzVal
            Holds the hyperparameters in a data class.
        '''
        
        hp_vec_mod       = np.copy(hp_vec)
        bvec             = hp_optz_info.bvec_log_optz
        hp_vec_mod[bvec] = 10**(hp_vec_mod[bvec])
        
        if hp_optz_info.has_theta:
            theta = hp_vec_mod[hp_optz_info.idx_theta]
        else:
            theta = None
        
        if hp_optz_info.has_kernel:
            hp_kernel = hp_vec_mod[hp_optz_info.idx_kernel]
        else:
            hp_kernel = None
        
        if hp_optz_info.has_varK:
            assert self.b_has_noisy_data
            varK = hp_vec_mod[hp_optz_info.idx_varK]
        else:
            varK = None
            
        if hp_optz_info.has_var_fval:
            assert self.b_optz_var_fval
            var_fval = hp_vec_mod[hp_optz_info.idx_var_fval]
        else:
            var_fval = None

        if hp_optz_info.has_var_fgrad:
            assert self.b_optz_var_fgrad
            var_fgrad = hp_vec_mod[hp_optz_info.idx_var_fgrad]
        else:
            var_fgrad = None
        
        return self.make_hp_class(None, theta, hp_kernel, varK, var_fval, var_fgrad)

    def set_custom_hp(self, beta = None, theta = None, kernel = None, varK = None, 
                      var_fval = None, var_fgrad = None):
        
        ''' 
        See definitions of inputs from the data class HparaOptzVal
        '''
        
        if varK is not None:
            assert varK > 0, f'varK must be positive but it is {varK}'
        
        hp_vals      = self.make_hp_class(beta, theta, kernel, varK, var_fval, var_fgrad)
        self.hp_vals = hp_vals
    
    ''' Bounds and initial starting points '''

    def get_hp_optz_x0(self, hp_optz_info, n_points, magn_rand = 2.0, 
                       cstr_w_old_hp = False, i_optz = None):
        '''
        Select n_points initial points for the optimization of the hyperparameters
        
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from the file GpHparaOptz
            Holds info on what hyperparameters to optimize numerically.
        n_points : int
            Number of initial data points to get for the numerical optimization 
            of the hyperparameters.
        magn_rand : float, optional
            All hyperparameters theta are initiated to have the same value. 
            To have some variance in the initial solution use magn_rand > 0. 
            The default is 2.
        cstr_w_old_hp : bool, optional
            If set to True the initial value of the hyperparameter are 
            constrained to be around a range of the previously optimized 
            hyperparameter. The default is False.
        i_optz : int, optional
            Iteration for the optimization of the hyperparameters. 
            Must be provided if cstr_w_old_hp is True. The default is None.

        Returns
        -------
        hp_x0 : 2D numpy array of floats
            Each row is the initial solution for the optimization of the hyperparameters.
        hp_optz_bounds : scipy.optimize.Bounds
            Bound constraints for hp_x0
        '''
        
        def make_latin_hypercube(lb_vec, ub_vec):
            
            if lb_vec.size == 1:
                # Vector of length n_points without nodes at the boundaries
                hp_x0       = np.linspace(lb_vec[0], ub_vec[0], n_points+2)[1:-1,None]
            else:
                para_limit  = np.array([lb_vec, ub_vec]).T
                sampling    = LHS(xlimits=para_limit, random_state = 1)
                hp_x0       = sampling(n_points) 
            
            return hp_x0
            
        para_min, para_max, hp_optz_bounds \
            = self.get_hp_bounds(hp_optz_info, cstr_w_old_hp = cstr_w_old_hp, 
                                 i_optz = i_optz)
        
        if hp_optz_info.has_theta:
            idx_theta  = hp_optz_info.idx_theta
            min_idx_th = np.min(idx_theta)
            
            # Same initial solution is used for all values of theta
            # Only have one value of theta as a free parameter
            bvec               = np.ones(hp_optz_info.n_hp, dtype=bool)
            bvec[idx_theta]    = False
            bvec[min_idx_th]   = True
        
            para_min_mod       = para_min[bvec]
            para_max_mod       = para_max[bvec]
            hp_x0_init         = make_latin_hypercube(para_min_mod, para_max_mod)
            
            hp_x0              = np.zeros((n_points, hp_optz_info.n_hp))
            hp_x0[:,bvec]      = hp_x0_init
            
            ''' Add randomness to theta around the diagonal th_1 = ... = th_d '''
            
            th_mean = hp_x0_init[:,min_idx_th]
            rng     = np.random.default_rng(seed=42)
            th_all  = th_mean[:,None] + rng.normal(0, magn_rand, (n_points, self.dim))
            
            # Enfource bounds for th
            th_min = para_min[min_idx_th]
            th_max = para_max[min_idx_th]
            th_all[th_all < th_min] = th_min
            th_all[th_all > th_max] = th_max
            
            hp_x0[:,idx_theta] = th_all
        else:
            hp_x0 = make_latin_hypercube(para_min, para_max, n_points)
        
        return hp_x0, hp_optz_bounds        
    
    def get_hp_best_init(self, i_optz):
        '''
        Evaluate the marginal log-likelihood at self.lkd_hp_best_n_eval different 
        hyperparameter values and identify the one that provides the maximium 
        likelihood. This is then the point that is returned and where the 
        optimization of the hyperparameters is initiated.
        
        Parameters
        ----------
        i_optz : int
            Iteration for the optimization of the hyperparameters. 

        Returns
        -------
        hp_best : 1D numpy array of floats
            The starting point for the optimization of the hyperparameters.
        hp_optz_bounds : scipy.optimize.Bounds
            Bound constraints for hp_best
        '''
        
        calc_cond = False if self.wellcond_mtd == 'precon' else True
        
        n_cases = self.lkd_hp_best_n_eval
        hp_x0, optz_bound = self.get_hp_optz_x0(self.hp_info_optz_lkd, n_cases, 
                                                cstr_w_old_hp = True, i_optz = i_optz)
        
        ln_lkd_all = np.full(n_cases, np.nan)
        cond_all   = np.full(n_cases, np.nan)
        
        for i in range(n_cases):
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
        hp_best = hp_x0[idx_max, :][None,:]
        
        return hp_best, optz_bound
