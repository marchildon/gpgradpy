#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:20:38 2022

@author: andremarchildon
"""

import numpy as np
from scipy import linalg

class GpMeanFunConst:
    
    def eval_mean_fun_const(self, x2model, beta_vec, bvec_use_grad = None,
                            calc_grad = False, calc_hess = False):
        
        mean_fval  = beta_vec[0]
        mean_fgrad = np.zeros(self.dim)             if calc_grad else None
        mean_fhess = np.zeros((self.dim, self.dim)) if calc_hess else None
            
        return mean_fval, mean_fgrad, mean_fhess
    
    def eval_mean_fun_batch_const(self, x2model, beta_vec, bvec_use_grad = None,
                                  calc_grad = False):
        
        nx, dim    = x2model.shape
        mean_fval  = np.ones(nx) * beta_vec[0]
        mean_fgrad = np.zeros((nx, dim)) if calc_grad else None
            
        return mean_fval, mean_fgrad
    
    def calc_mean_fun_wrt_hpara_const(self, K_chofac, data_vec, K_grad_hp = None):
        
        # Calculate hyperparameter that maximizes marginal log-likelihood
        one_mod               = np.zeros(data_vec.size)
        one_mod[:self.n_eval] = 1
        
        invK_one      = linalg.cho_solve(K_chofac, one_mod)
        invK_data     = linalg.cho_solve(K_chofac, data_vec)
        
        denominator   = np.dot(one_mod, invK_one)
        beta_star     = np.dot(data_vec, invK_one) / denominator
        
        # Evaluate the model with beta_star
        model_val = one_mod * beta_star
        
        # Calculate gradients
        if K_grad_hp is None:
            beta_grad     = None
            model_grad_hp = None
        else:
            beta_grad     = np.einsum('i,kij,j->k', beta_star * invK_one - invK_data, K_grad_hp, invK_one / denominator)
            model_grad_hp = np.outer(one_mod, beta_grad)
        
        return model_val, model_grad_hp, beta_star, beta_grad

        
class GpMeanFunPoly:
    
    def eval_mean_fun_poly(self, x2model, beta_vec, bvec_use_grad = None, 
                           calc_grad = False, calc_hess = False):
        '''
        Parameters
        ----------
        x2model : 2D numpy array of size [n2model, dim]
            Points at which the mean function is to be evaluated.
        beta_vec : 1D numpy array
            Coefficients for the mean function.
        bvec_use_grad : 1D numpy array of bool of length n2model
            Indicates if the gradient at the i-th row of x2model is to be kept
        bvec_use_grad : 1D numpy array of bool of length n2model
            Indicates if the Hessian at the i-th row of x2model is to be kept
        calc_grad : bool, optional
            Set to True to calculate the gradient of the mean function. 
            The default is False.
        calc_hess : bool, optional
            Set to True to calculate the Hessian of the mean function. 
            The default is False.

        Returns
        -------
        mean_fval : 1D numpy array
            Value of the mean function.
        mean_fgrad : 2D numpy array
            Gradient of the mean function.
        mean_fhess : 3D numpy array
            Hessian of the mean function.
        '''
        
        assert beta_vec.size == self.n_beta_coeff, f'Size of beta_vec is {beta_vec.size} but n_basis = {self.n_beta_coeff}'
        
        vand_base, vand_grad, vand_hess = self.calc_vand(x2model, calc_grad, calc_hess)
        
        mean_fval = vand_base @ beta_vec
        
        if calc_grad: 
            if bvec_use_grad is not None:
                vand_grad = vand_grad[:,bvec_use_grad,:]
                
            mean_fgrad = np.einsum('ijk,k->ji', vand_grad, beta_vec)
            
            if calc_hess:
                mean_fhess = np.einsum('ijkl,l->kij', vand_hess, beta_vec)
            else:
                mean_fhess = None
        else:
            mean_fgrad = mean_fhess = None
            
        return mean_fval, mean_fgrad, mean_fhess
    
    def eval_mean_fun_batch_poly(self, x2model, beta_vec, bvec_use_grad = None,
                                 calc_grad = False):
        
        x_scl    = self.get_scl_x_w_dist()[0]
        vand_aug = self.calc_aug_vand(x_scl, bvec_use_grad)
        
        ''' Calculate the parameters and evaluate the model '''
        
        mean_fval = vand_aug @ beta_vec # Model value and its gradient wrt x
        
        if calc_grad:
            raise Exception('This is not available')
        else:
            mean_fgrad = None
        
        return mean_fval, mean_fgrad
    
    def calc_mean_fun_wrt_hpara_poly(self, K_chofac, data_vec, K_grad_hp = None):
        '''
        Parameters
        ----------
        x2model : 1D numpy array
            Points at which the mean function is to be evaluated.
        K_chofac : scipy.linalg.cho_factor
            Cholesky factorization of the kernel matrix.
        data_vec : 1D numpy array
            Array with appended function evaluations and gradients.
        K_grad_hp : 3D numpy array
            Gradients of the kernel matrix with respect to the hyperparameters.

        Returns
        -------
        model_val : float
            Value of the mean function at x2model.
        model_grad_hp : 1D numpy array
            Gradient of the mean function at x2model with respect to the 
            hyperparameters.
        beta : 1D numpy array
            Coefficients for the mean function.
        beta_grad : 2D numpy array
            Gradient of the coefficients of the mean function with respect to 
            the hyperparameters.
        '''
        
        ''' Preliminary '''
        
        x_scl       = self.get_scl_x_w_dist()[0]
        vand_aug    = self.calc_aug_vand(x_scl, self.bvec_use_grad)
        invK_vand   = linalg.cho_solve(K_chofac, vand_aug)
        term1       = np.linalg.solve(vand_aug.T @ invK_vand, invK_vand.T)
        
        # n_data, n_beta_coeff = vand_aug.shape
        
        ''' Calculate the parameters and evaluate the model '''
        
        beta      = term1 @ data_vec
        model_val = vand_aug @ beta # Model value and its gradient wrt x
        
        if K_grad_hp is None:
            beta_grad     = None
            model_grad_hp = None
        else:
            invK_data = linalg.cho_solve(K_chofac, data_vec)
            
            term2     = invK_vand @ beta - invK_data
            beta_grad = np.einsum('ij,kjl,l->ik', term1, K_grad_hp, term2)
        
            # Gradient of model_val with respect to the hyperparamerters
            model_grad_hp = vand_aug @ beta_grad 
        
        return model_val, model_grad_hp, beta, beta_grad
    
    def calc_vand(self, x2model, calc_grad = False, calc_hess = False):
        '''
        Parameters
        ----------
        x2model : 2D numpy array of size [n2model, dim]
            Points at which the Vandermonde matrix is to be evaluated.
        calc_grad : bool, optional
            Set to True to calculate the gradient of the Vandermonde matrix 
            with respect to x2model. The default is False.
        calc_hess : TYPE, optional
            Set to True to calculate the Hessian of the Vandermonde matrix 
            with respect to x2model. The default is False.

        Returns
        -------
        vand_base : 2D numpy array 
            Vandermonde matrix.
        vand_grad : 3D numpy array
            Derivative of the Vandermonde matrix.
        vand_hess : 4D numpy array
            Hessian of the Vandermonde matrix.
        '''
        
        n2model = x2model.shape[0]
        
        vand_base = np.full((n2model, self.n_beta_coeff), np.nan)
        
        if calc_grad:
            vand_grad = np.zeros((self.dim, n2model, self.n_beta_coeff))
            
            if calc_hess:
                vand_hess = np.zeros((self.dim, self.dim, n2model, self.n_beta_coeff))
            else:
                vand_hess = None
        else:
            vand_grad = vand_hess = None
        
        # 0-th order term 
        vand_base[:,0] = 1 
        
        if self.n_beta_coeff > 1:
            vand_base[:,1:(self.dim +1)] = x2model
            
            for i in range(self.dim):
                if calc_grad: vand_grad[i, :, i+1] = 1
        
        return vand_base, vand_grad, vand_hess
    
    def calc_aug_vand(self, x2model, bvec_use_grad = None):
        
        n2model, dim = x2model.shape
        
        assert dim == self.dim, 'Unexpected shape for x2model'
        
        if self.use_grad:
            vand_base, vand_grad = self.calc_vand(x2model, calc_grad = True)[:2]
            
            if bvec_use_grad is None:
                vand_grad_vec = vand_grad.reshape((n2model*dim, self.n_beta_coeff))
            else:
                n_grad = np.sum(bvec_use_grad)
                vand_grad_vec = vand_grad[:,self.bvec_use_grad,:].reshape((n_grad*dim, self.n_beta_coeff))
            
            vand_aug = np.vstack((vand_base, vand_grad_vec))
        else:
            vand_aug = self.calc_vand(x2model, calc_grad = False)[0]
        
        return vand_aug


class GpMeanFunQN:
    
    def update_mean_fun_qN(self, n_eval, b_progress, xvec_new, fval_new, fgrad_new, tol_den = 1e-8):
        
        print(f'In update_mean_fun_qN, b_progress = {b_progress}, Hess =')
        print(f'{self._qN_fhess}')
        
        if n_eval == 1:
            self._qN_xvec  = np.copy(xvec_new)
            self._qN_fval  = fval_new
            self._qN_fgrad = np.copy(fgrad_new)
            self._qN_fhess = np.eye(self.dim)
        
        elif b_progress:
            yvec        = fgrad_new - self._qN_fgrad
            svec        = xvec_new - self._qN_xvec
            common      = yvec - self._qN_fhess @ svec 
            denominator = common @ svec
            test_term   = tol_den * np.linalg.norm(svec) * np.linalg.norm(common)
            
            if denominator >= test_term:
                self._qN_xvec  = np.copy(xvec_new)
                self._qN_fval  = fval_new
                self._qN_fgrad = np.copy(fgrad_new)
                self._qN_fhess += np.outer(common, common / (common @ svec))
                
    def eval_mean_fun_qN(self, x2model, calc_grad = True, calc_hess = False):
        
        if x2model.ndim == 1:
            xdiff = x2model - self._qN_xvec
        else:
            xdiff = x2model[0,:] - self._qN_xvec
        
        hess_xdiff = self._qN_fhess @ xdiff
        mean_fval  = self._qN_fval + np.dot(self._qN_fgrad, xdiff) + 0.5* np.dot(xdiff, hess_xdiff)
        mean_fgrad = self._qN_fgrad + hess_xdiff if calc_grad else None
        mean_fhess = self._qN_fhess              if calc_hess else None
        
        return mean_fval, mean_fgrad, mean_fhess
        
    def eval_mean_fun_batch_qN(self, xeval, calc_grad = True):
        
        xdiff      = (xeval - self._qN_xvec[None,:]).T
        hess_xdiff = self._qN_fhess @ xdiff
        
        mean_fval  = self._qN_fval + np.dot(self._qN_fgrad, xdiff) + 0.5* np.diag(xdiff.T @ hess_xdiff)
        mean_fgrad = self._qN_fgrad[None,:] + hess_xdiff.T if calc_grad else None
        
        return mean_fval, mean_fgrad
    
    def calc_mean_fun_wrt_hpara_qN(self, K_chofac, data_vec, K_grad_hp):
        
        # Evaluate the mean function
        x_scl                 = self.get_scl_x_w_dist()[0]
        mean_fval, mean_fgrad = self.eval_mean_fun_batch_qN(x_scl)
        
        if self.use_grad:
            model_val = self.make_data_vec(mean_fval, mean_fgrad)
        else:
            model_val = mean_fval
        
        beta      = np.zeros(0)
        beta_grad = np.zeros((0, self.dim))
        
        if K_grad_hp is None:
            model_grad_hp = None
        else:
            n_hp          = K_grad_hp.shape[0]
            model_grad_hp = np.zeros((0,n_hp))
        
        return model_val, model_grad_hp, beta, beta_grad
    
class GpMeanFun(GpMeanFunConst, GpMeanFunPoly, GpMeanFunQN):
    
    def set_mean_fun_op(self, mean_fun_type = 'poly_ord_0'):
        
        self.mean_fun_type = mean_fun_type
        
        if mean_fun_type == 'poly_ord_0':
            self.n_beta_coeff    = 1 
            self.beta_var_npara  = self.n_beta_coeff
            self.optz_beta_w_lkd = True
        elif 'poly' in self.mean_fun_type:
            raise Exception('Method not tested')
        elif mean_fun_type == 'qN_SR1':
            self.n_beta_coeff    = 0 
            self.beta_var_npara  = self.n_beta_coeff
            self.optz_beta_w_lkd = False
            self._qN_fhess       = np.eye(self.dim)
        else:
            raise Exception(f'mean_fun_type = {mean_fun_type} not available')
    
    def update_mean_fun(self, n_eval, b_progress, xvec_new, fval_new, fgrad_new):
        
        if self.mean_fun_type == 'qN_SR1':
            self.update_mean_fun_qN(n_eval, b_progress, xvec_new, fval_new, fgrad_new)
    
    def eval_mean_fun(self, x2model, beta_vec = None, 
                      bvec_use_grad = None, calc_grad = False, calc_hess = False):
        
        if self.mean_fun_type == 'poly_ord_0':
            return self.eval_mean_fun_const(x2model,   beta_vec, bvec_use_grad, 
                                            calc_grad, calc_hess)
        elif 'poly' in self.mean_fun_type:
            return self.eval_mean_fun_poly(x2model,   beta_vec, bvec_use_grad, 
                                           calc_grad, calc_hess)
        elif self.mean_fun_type == 'qN_SR1':
            return self.eval_mean_fun_qN(x2model, calc_grad, calc_hess)
        else:
            raise Exception('Unknown method')
            
    def eval_mean_fun_batch(self, x2model, beta_vec = None, 
                            bvec_use_grad = None, calc_grad = False):
        
        if self.mean_fun_type == 'poly_ord_0':
            return self.eval_mean_fun_batch_const(x2model, beta_vec, 
                                                  bvec_use_grad, calc_grad)
        
        elif 'poly' in self.mean_fun_type:
            return self.eval_mean_fun_batch_poly(x2model, beta_vec, 
                                                 bvec_use_grad, calc_grad)
        elif self.mean_fun_type == 'qN_SR1':
            return self.eval_mean_fun_batch_qN(x2model, calc_grad)
        else:
            raise Exception('Unknown method')
            
    def calc_mean_fun_wrt_hpara(self, K_chofac, data_vec, K_grad_hp = None):
        
        if self.mean_fun_type == 'poly_ord_0':
            return self.calc_mean_fun_wrt_hpara_const(K_chofac, data_vec, K_grad_hp)
        elif 'poly' in self.mean_fun_type:
            return self.calc_mean_fun_wrt_hpara_poly(K_chofac, data_vec, K_grad_hp)
        elif self.mean_fun_type == 'qN_SR1':
            return self.calc_mean_fun_wrt_hpara_qN(K_chofac, data_vec, K_grad_hp)
        else:
            raise Exception('Unknown method')
