#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:21:22 2022

@author: andremarchildon
"""

import numpy as np 

from gpgradpy.src.base import CommonFun

'''
The methods in this file are used to ensure the gradient-enhanced correlation 
matrix is well-conditioned 
'''

class GpWellCondVreq:
    
    ''' 
    This class is used for the rescaling method. See the following paper:
        A Non-intrusive Solution to the Ill-Conditioning Problem of the 
        Gradient-Enhanced Gaussian Covariance Matrix for Gaussian Processes
    '''

    def calc_mtd_rescale_origin_vreq(self, n_eval, dim = None):
        
        if dim is None:
            dim = self.dim 
        
        # Calculate req vmin for gradient-enhanced GP
        if n_eval == 1:
            vmin_req_grad = 1
        else:
            dist_star     = 2 * np.sqrt(dim)
            sqrt_term     = np.sqrt(4 + 2 * np.exp(2) * np.log((n_eval - 1) * (1 + dist_star) / 2))
            vmin_req_grad = (2 + sqrt_term) / np.exp(1)
            vmin_req_grad = np.minimum(vmin_req_grad, dist_star)
            
        return vmin_req_grad

    def rescaling_data_w_theta_sol(self, X_scl_v1, xvec_scale_v1, hp_theta, 
                                   tol_min_dist_x = 1e-15):
        
        ''' Calculate new scaling '''
        
        n_eval = X_scl_v1.shape[0] 
        assert n_eval > 1, 'This method should only be called if n_eval > 1'
        
        if self.optz_log_hp_theta:
            theta_sol     = 10**hp_theta
            log_theta_sol = hp_theta
        else:
            theta_sol     = hp_theta
            log_theta_sol = np.log10(hp_theta)
            
        vreq = self.calc_mtd_rescale_origin_vreq(n_eval, self.dim)
        
        theta_star_v1   = 10**np.mean(log_theta_sol)
        
        xvec_scale_v2   = np.sqrt(theta_sol / theta_star_v1)
        X_scl_v2        = X_scl_v1 * xvec_scale_v2[None,:]
        
        # Calculate and apply correction to have vmin(X) == vreq
        min_dist_v2     = np.max((CommonFun.calc_dist_min(X_scl_v2), tol_min_dist_x))
        correction      = vreq / min_dist_v2
        xvec_scale_new  = xvec_scale_v1 * xvec_scale_v2 * correction
        
        ''' Provide estimate of new solution '''
        
        dist2th_star      = np.dot(log_theta_sol, log_theta_sol) - np.dot(log_theta_sol, np.ones(self.dim))**2 / self.dim 
        theta_star_v2_est = np.ones(self.dim) * theta_star_v1 / correction**2
        
        theta_out = np.log10(theta_star_v2_est) if self.optz_log_hp_theta else theta_star_v2_est
        
        return theta_out, dist2th_star, xvec_scale_new
    
    def calc_nugget_Kfull_vreq(self, n_eval, vmin = None):
        
        if vmin is None:
            vmin = self.calc_mtd_rescale_origin_vreq(n_eval)
        
        cond_max = self.cond_max_target
        
        # Calculate req vmin for gradient-enhanced GP
        if n_eval == 1:
            eta_Kgrad = n_eval / (cond_max - 1)
        else:
            assert vmin >= np.sqrt(2), 'This method requires that vmin = {vmin} >= sqrt(2)'
            
            v_frac      = 2 * np.sqrt(self.dim) / vmin
            u_eigmax    = 1 + (n_eval - 1) * v_frac * np.exp(1/v_frac - 1)
            eta_Kgrad   = u_eigmax / (cond_max - 1)          
            eta_Kbase   = self.calc_nugget_Kbase(n_eval, cond_max)
            
            assert v_frac >= 0.99, f'This term should be greater or equal to 1, v_frac = {v_frac}'
            assert eta_Kgrad >= 0.99 * eta_Kbase, f'We expect that eta_Kgrad > eta_Kbase but eta_Kgrad = {eta_Kgrad}, eta_Kbase = {eta_Kbase}'
        
        return eta_Kgrad

class GpWellCond(GpWellCondVreq):
    
    ''' 
    For more details on the preconditioning method see the following paper:
        A Solution to the Ill-Conditioning of Gradient- Enhanced Covariance 
        Matrices for Gaussian Processes
    '''
    
    def calc_nugget_Kbase(self, n_eval, cond_max = None):
        
        if cond_max is None:
            cond_max = self.cond_max_target
        
        return n_eval / (cond_max - 1)
    
    def calc_nugget(self, n_eval):
        
        if self.cond_eta_set_mtd == 'dflt_eta':
            eta_Kbase = eta_Kgrad = self.cond_eta_dflt
        else:
            eta_Kbase = self.calc_nugget_Kbase(n_eval)
            
            if self.use_grad:
                if n_eval == 1:
                    eta_Kgrad = eta_Kbase
                elif self.wellcond_mtd == 'precon':
                    dim = self.dim
                    
                    if self.kernel_type == 'SqExp' or self.kernel_type == 'RatQu' :
                        ub_sum_off_diag = 0.5*(n_eval - 1) * (1 + np.sqrt(1 + 4*dim)) \
                            * np.exp(-(1 + 2*dim - np.sqrt(1 + 4*dim)) / (4*dim))
                    elif self.kernel_type == 'Ma5f2':
                        alpha           = (np.sqrt(3*dim) - 1 + np.sqrt(15*dim + 2*np.sqrt(3*dim) +1)) / (2*(3*dim + np.sqrt(3*dim)))
                        ub_sum_off_diag = (n_eval - 1) * (1 + (dim + np.sqrt(3*dim)) * alpha + dim*(1 + np.sqrt(3*dim))*alpha**2) * np.exp(-np.sqrt(3*dim) * alpha)
                    else:
                        raise Exception(f'Unknown kernel of {self.kernel_type}, needs to be added')
                            
                    eta_Kgrad = (1 + ub_sum_off_diag) / (self.cond_max_target - 1)
                    
                elif 'rescale' in self.wellcond_mtd:
                    eta_Kgrad = self.calc_nugget_Kfull_vreq(n_eval)
                else:
                    if self.cond_eta_set_mtd == 'Kbase_eta':
                        eta_Kgrad = eta_Kbase
                    elif self.cond_eta_set_mtd == 'Kbase_eta_w_dim':
                        eta_Kgrad = eta_Kbase * (self.dim + 1)
                    elif self.cond_eta_set_mtd == 'dflt_eta':
                        eta_Kgrad = self.cond_eta_dflt
                    else:
                        raise Exception(f'Uknown method for cond_eta_set_mtd = {self.cond_eta_set_mtd}')
            else:
                eta_Kgrad = np.nan
            
        return eta_Kbase, eta_Kgrad
    