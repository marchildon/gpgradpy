#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:04:13 2022

@author: andremarchildon
"""

import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import Bounds


class GpHparaCon:
    
    def get_hp_bounds(self, hp_optz_info, cstr_w_old_hp = False, i_optz = None, range_frac = 0.2):
        '''
        Parameters
        ----------
        hp_optz_info : HparaOptzInfo from file GpHparaOptz
            Info on the numerical optimization of the hyperparameters.
        cstr_w_old_hp : bool, optional
            Set to True to constrain the hyperparameters to be within a certain 
            range of the previous hyperparameters. The default is False.
        i_optz : int, optional
            Index for the optimization iteration. The default is None.
        range_frac : float, optional
            Fraction of the total range of values the hyperparameter can take. 
            Only used if cstr_w_old_hp is True. The default is 0.2.

        Returns
        -------
        para_min : 1D numpy array
            Minimum values the hyperparameters can take.
        para_max : 1D numpy array
            Maximum values the hyperparameters can take.
        hp_optz_bounds : scipy.optimize.Bounds
            Bounds for the optimization of the hyperparameters
        '''
        
        def calc_hp_range(range_hp, hp_old_in, use_log10 = True):
            
            hp_old = np.max((hp_old_in, range_hp[0]))
            hp_old = np.min((hp_old,    range_hp[1]))
            
            dom_hp_len  = range_frac * (range_hp[1] - range_hp[0])
            hp_min      = np.max((range_hp[0], hp_old - dom_hp_len))
            hp_max      = np.min((range_hp[1], hp_old + dom_hp_len))
            
            if use_log10:
                return 10**hp_min, 10**hp_max
            else:
                return hp_min, hp_max
        
        if i_optz == 0:
            cstr_w_old_hp = False
        
        if cstr_w_old_hp:
            assert 0 < range_frac < 1, f'Must have 0 < range_frac < 1 but range_frac = {range_frac}'
            idx = i_optz - 1
        
        ''' Set parameters limits '''

        para_min = np.full(hp_optz_info.n_hp, np.nan)
        para_max = np.full(hp_optz_info.n_hp, np.nan)
        
        if hp_optz_info.has_theta:
            if cstr_w_old_hp:
                mean_log_theta = np.mean(np.log10(self.hp_theta_all[idx, :]))
                hp_min, hp_max = calc_hp_range(np.log10(self.hp_theta_range), mean_log_theta)
            else:
                hp_min = self.hp_theta_range[0]
                hp_max = self.hp_theta_range[1]
            
            para_min[hp_optz_info.idx_theta] = hp_min
            para_max[hp_optz_info.idx_theta] = hp_max
        
        if hp_optz_info.has_kernel:
            para_min[hp_optz_info.idx_kernel] = self.hp_kernel_range[0]
            para_max[hp_optz_info.idx_kernel] = self.hp_kernel_range[1]
        
        if hp_optz_info.has_varK:
            if cstr_w_old_hp and (i_optz > (self.hp_const_n_eval + 1)):
                hp_min, hp_max = calc_hp_range(np.log10(self.hp_varK_range), np.log10(self.hp_varK_all[idx]))
            else:
                hp_min = self.hp_varK_range[0]
                hp_max = self.hp_varK_range[1]
                
            para_min[hp_optz_info.idx_varK] = hp_min
            para_max[hp_optz_info.idx_varK] = hp_max
        
        if hp_optz_info.has_var_fval:
            if cstr_w_old_hp and (i_optz > (self.hp_const_n_eval + 1)):
                hp_min, hp_max = calc_hp_range(np.log10(self.hp_var_fval_range), np.log10(self.hp_var_fval_all[idx]))
            else:
                hp_min = self.hp_var_fval_range[0]
                hp_max = self.hp_var_fval_range[1]
            
            para_min[hp_optz_info.idx_var_fval] = hp_min
            para_max[hp_optz_info.idx_var_fval] = hp_max
            
        if hp_optz_info.has_var_fgrad:
            if cstr_w_old_hp:
                hp_min, hp_max = calc_hp_range(np.log10(self.hp_var_fgrad_range), np.log10(self.hp_var_fgrad_all[idx]))
            else:
                hp_min = self.hp_var_fgrad_range[0]
                hp_max = self.hp_var_fgrad_range[1]
            
            para_min[hp_optz_info.idx_var_fgrad] = hp_min
            para_max[hp_optz_info.idx_var_fgrad] = hp_max
            
        assert np.sum(np.isnan(para_min)) == 0, f'There are nan entries in para_min = {para_min}'
        assert np.sum(np.isnan(para_max)) == 0, f'There are nan entries in para_max = {para_max}'
        
        if np.any(para_min >= para_max):
            print('Surr hpara bounds are not feasible')
            print(f'para_min = {para_min}')
            print(f'para_max = {para_max}')
            print(f'para_max - para_min = {para_max - para_min}')
            
        bvec           = hp_optz_info.bvec_log_optz
        para_min[bvec] = np.log10(para_min[bvec])
        para_max[bvec] = np.log10(para_max[bvec])

        hp_optz_bounds = Bounds(para_min, para_max, keep_feasible=True)

        return para_min, para_max, hp_optz_bounds

    def calc_cond_w_grad(self, Kmat_w_eta, Kmat_chofac, Kmat_grad_hp = None, cond_norm = None):
        '''
        Parameters
        ----------
        Kmat_w_eta : 2D numpy array
            Either the regularized kernel or covariance matrix.
        Kmat_chofac : scipy.linalg.cho_factor
            Cholesky factorization of Kmat_w_eta.
        Kmat_grad_hp : 3D numpy array, optional
            Derivative of Kmat_w_eta with respect to the hyperparameters. 
            The default is None.
        cond_norm : int or float, optional
            Indicates the type of norm to use to calculate the condition number. 
            The default is None.
        '''
        
        if cond_norm is None:
            cond_norm = self.cond_norm
        
        if cond_norm == 2:
            return self.calc_cond_L2_w_grad(Kmat_w_eta, Kmat_grad_hp)
        elif cond_norm == 'fro':
            return self.calc_cond_fronorm_w_grad(Kmat_w_eta, Kmat_chofac, Kmat_grad_hp)
        else:
            raise Exception(f'cond_norm must be either 2 or "fro" but it is {cond_norm}')
        
    def calc_cond_L2_w_grad(self, Kmat_w_eta, Kmat_grad_hp = None):
        # assert Kmat_w_eta[0,0] > 1, 'Kmat_w_eta should have diagonal entries greater than 1 since it contains the noise'

        n_data = Kmat_w_eta.shape[0]
        cond = np.linalg.cond(Kmat_w_eta, p=2)
        
        if Kmat_grad_hp is None:
            cond_grad = None
        else:
            assert self.wellcond_mtd != 'precon', 'Not setup to calculate the gradient of the condition number if wellcond_mtd = "precon" '
            
            nder = Kmat_grad_hp.shape[0]
            assert n_data == Kmat_grad_hp.shape[1] == Kmat_grad_hp.shape[2], 'Shape of Kmat_grad_hp is incompatible'
    
            eigval_all, eigvec_all = np.linalg.eig(Kmat_w_eta)
            
            idx_max  = np.argmax(eigval_all)
            v_max    = eigvec_all[:,idx_max]
            
            idx_mmin = np.argmin(eigval_all)
            eig_min  = eigval_all[idx_mmin]
            v_min    = eigvec_all[:,idx_mmin]
            
            eig_min_mod  = np.max((eig_min, 1e-16))
            
            v_outer_diff = np.outer(v_max, v_max) - cond * np.outer(v_min, v_min)
            cond_grad    = np.zeros(nder)
    
            for i in range(nder):
                cond_grad[i] = np.sum(v_outer_diff * Kmat_grad_hp[i,:,:]) / eig_min_mod
    
            # Code below is not used since scipy.linalg.eigh has a bug and occationally fails
    
            # try:
            #     eig_min, v_min = linalg.eigh(Kmat_w_eta, eigvals=[0,0])
            # except:
            #     print('** Error with linalg.eigh for eig min **')
            #     cond_grad = np.zeros(nder)
            #     return cond, cond_grad
    
            # try:
            #     eig_max, v_max = linalg.eigh(Kmat_w_eta, eigvals=[n_data-1,n_data-1])
            # except:
            #     if eig_max.size == 0:
            #         try:
            #             print('** Error with scipy linalg.eigh **')
            #             eigval_all, eigvec_all = np.linalg.eig(Kmat_w_eta)
            #             idx_max = np.argmax(eigval_all)
            #             eig_max = eigval_all[idx_max]
            #             v_max   = eigvec_all[:,idx_max]
            #         except:
            #             print('** Error with scipy linalg.eigh and np.linalg.eig **')
            #             cond_grad = np.zeros(nder)
            #             return cond, cond_grad
            #     else:
            #         print('** Error with linalg.eigh for eig max **')
            #         cond_grad = np.full(nder, np.nan)
            #         return cond, cond_grad
    
            # eig_min_mod  = np.max((eig_min[0], 1e-16))
            
            # try:
            #     v_outer_diff = np.outer(v_max, v_max) - cond * np.outer(v_min, v_min)
            #     cond_grad    = np.zeros(nder)
        
            #     for i in range(nder):
            #         cond_grad[i] = np.sum(v_outer_diff * Kmat_grad_hp[i,:,:]) / eig_min_mod
            # except:
            #     print('Error calculating cond_grad')

        return cond, cond_grad
    
    def calc_cond_fronorm_w_grad(self, Kmat_w_eta, Kmat_chofac, Kmat_grad_hp = None):
        # assert Kmat_w_eta[0,0] > 1, 'Kmat_w_eta should have diagonal entries greater than 1 since it contains the noise'
        
        n     = Kmat_w_eta.shape[0]
        K_inv = cho_solve(Kmat_chofac, np.eye(n))
        
        fro_norm_K      = np.linalg.norm(Kmat_w_eta, 'fro')
        fro_norm_K_inv  = np.linalg.norm(K_inv, 'fro')
        
        cond = fro_norm_K * fro_norm_K_inv
        
        if Kmat_grad_hp is None:
            cond_grad = None
        else:
            assert self.wellcond_mtd != 'precon', 'Not setup to calculate the gradient of the condition number if wellcond_mtd = "precon" '
            
            K_inv          = cho_solve(Kmat_chofac, np.eye(n))
            Kinv_Kinv_Kinv = cho_solve(Kmat_chofac, K_inv)          # Kinv @ Kinv
            Kinv_Kinv_Kinv = cho_solve(Kmat_chofac, Kinv_Kinv_Kinv) # Kinv @ Kinv @ Kinv
            
            frac        = fro_norm_K_inv / fro_norm_K
            d_cond_dK   = frac * Kmat_w_eta - Kinv_Kinv_Kinv / frac
            cond_grad   = np.sum(d_cond_dK[None,:,:] * Kmat_grad_hp, axis=(1,2))
        
        return cond, cond_grad
    