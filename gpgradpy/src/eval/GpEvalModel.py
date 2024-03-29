#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:44:56 2021

@author: andremarchildon
"""

import copy
import numpy as np
from scipy import linalg

from . import GpMeanFun

class GpEvalModel(GpMeanFun):

    def setup_eval_model(self, calc_cond = False):
        
        '''
        Prior to using this method the methods set_data() and set_hpara() need 
        to have been called. Once this method has been called the surrogate 
        can be evaluated with eval_model()
        '''
        
        self._hp_vals_model_setup = copy.copy(self.hp_vals)
        
        x_scl, Rtensor        = self.get_scl_x_w_dist()
        fval_scl, _, grad_scl = self.get_scl_eval_data()[:3]
        
        mean_fun_val, mean_fun_grad = self.eval_mean_fun(x_scl, self.hp_vals.beta, 
                                                         bvec_use_grad = self.bvec_use_grad,
                                                         calc_grad     = self.use_grad)[:2]
        
        ''' Calculate the inverse matrix product '''
        
        mean_fun_vec = self.make_data_vec(mean_fun_val, mean_fun_grad)
        data_vec     = self.make_data_vec(fval_scl, grad_scl)
        
        Kern, Kcor, KernEta, KernEta_chofac, condK, etaK \
            = self.calc_all_K_w_chofac(Rtensor, self.hp_vals, 
                                       b_normlz_w_varK = True, calc_cond = calc_cond)[:6]
        
        f_diff = data_vec - mean_fun_vec
        
        ''' Store data '''
        
        self.data_vec       = data_vec
        self.Kern           = Kern
        self.KernEta        = KernEta
        self.KernEta_chofac = KernEta_chofac
        self.condK          = condK
        self.etaK_eval      = etaK
        
        if KernEta_chofac is None:
            self.invKernEta_fdiff = None
        else:
            self.invKernEta_fdiff = linalg.cho_solve(KernEta_chofac, f_diff)
        
    def eval_model(self, x2model_in, calc_grad = False, calc_hess = False, squeeze_nx = False):
        '''
        Parameters
        ----------
        x2model_in : 2d numpy array of size [nx, dim], where nx is the number of points in the parameter space
            Each row is where the surrogate will be evaluated
        calc_grad : bool, optional
            If True the grad of the surrogate is evaluated. The default is False.
        calc_hess : bool, optional
            If True the Hessian of the surrogate is evaluated. The default is False.
        squeeze_nx : bool, optional
            If True then nx must be 1 and the returned arrays have their 
            first dimension collapsed. The default is False.

        Returns
        -------
        mu : 1d numpy array of length nx (or float if squeeze_nx is True)
            Mean of the surrogate evaluated at the rows of x2model_in.
        sig : 1d numpy array of length nx (or float if squeeze_nx is True)
            Standard deviation of the surrogate evaluated at the rows of x2model_in.
        dmudx : 2d numpy array of size [nx, dim]
            Gradient of the mean of the surrogate evaluated at the rows of x2model_in.
        dsigdx : 2d numpy array of size [nx, dim]
            Gradient of the standard deviation of the surrogate evaluated at the rows of x2model_in.
        d2mudx2 : 3d numpy array of size [nx, dim, dim]
            Hessian of the mean of the surrogate evaluated at the rows of x2model_in.
        d2sigdx2 : 3d numpy array of size [nx, dim, dim]
            Hessian of the standard deviation of the surrogate evaluated at the rows of x2model_in.
        '''
        
        ''' Check inputs '''
        
        assert self.KernEta_chofac is not None, 'To evaluate the surr the Cholesky decomposition is required'
        
        if x2model_in.ndim == 1:
            x2model = x2model_in[None,:]
        elif x2model_in.ndim == 2:
            x2model = x2model_in
        else:
            raise Exception(f'x2model_in should be a 2d array but it has shape {x2model_in.shape}')
        
        nx = x2model.shape[0]
        
        if squeeze_nx:
            assert nx == 1, 'If squeeze_nx is True, then x_acq must only have one point'
        
        check_hp_vals = self.hp_vals == self._hp_vals_model_setup
        
        if check_hp_vals is False:
            print('hp_vals has changed since setup_eval_model() was called')
            print(f'hp_vals_model_setup = \n{self._hp_vals_model_setup}')
            print(f'hp_vals = \n{self.hp_vals}')
            raise Exception('Cannot change hp_vals between calling setup_eval_model() and eval_model()')

        theta     = self.hp_vals.theta 
        hp_kernel = self.hp_vals.kernel
        hp_beta   = self.hp_vals.beta

        '''Preliminaries '''
        
        sigK = np.sqrt(self.hp_vals.varK)
        
        if self.b_use_data_scl:
            x2model = self.DataScl.x_init_2_scl(x2model)
            
        x_eval_scl = self.get_scl_x_w_dist()[0]
        
        if calc_hess: 
            assert calc_grad, 'To return the hessian calc_grad must also be set to True'

        ny      = self.n_eval
        nx      = x2model.shape[0]
        one_nx  = np.ones(nx)

        Rtensor = self.calc_Rtensor(x_eval_scl, x2model, 1)
        if self.use_grad or calc_grad:
            Kgrad_yx = self.calc_KernGrad(Rtensor, theta, hp_kernel, 
                                          bvec_use_grad1 = self.bvec_use_grad)
            
            if self.use_grad:
                Kyx      = Kgrad_yx[:, :nx]
                dKxy_dx  = Kgrad_yx[:, nx:].T
            else:
                Kyx       = Kgrad_yx[:ny, :nx]
                dKyx_dx   = Kgrad_yx[:ny, nx:]
                dKxy_dx   = dKyx_dx.T
        else:
            Kyx = self.calc_KernBase(Rtensor, theta, hp_kernel)
        
        Kxy = Kyx.T
        
        if calc_hess:
            Rtensor   = self.calc_Rtensor(x2model, x_eval_scl, 1)
            d2Kxy_dx2 = self.calc_Kern_hess_x(Rtensor, theta, hp_kernel) # * sigK
            
        Kxy_invKernEta = linalg.cho_solve(self.KernEta_chofac, Kyx).T

        ''' Solve for the mean and std of the model '''
        
        base_model_val, base_model_grad, base_model_hess \
            = self.eval_mean_fun(x2model, hp_beta, 
                                 calc_grad = calc_grad, calc_hess = calc_hess)
        
        sig2_wo_sigK = one_nx - np.diag(Kxy @ Kxy_invKernEta.T)
        assert np.min(sig2_wo_sigK) >= 0, f'The variance of the surr should be non-negative but min(sig2_wo_sigK) = {np.min(sig2_wo_sigK)}'
        
        sig2_wo_sigK[sig2_wo_sigK<0] = 0 # Clip very small negative values 
        sig = np.sqrt(sig2_wo_sigK) * sigK

        mu = base_model_val + Kyx.T @ self.invKernEta_fdiff

        if calc_grad:
            dmudx    = self.calc_dmudx(dKxy_dx) + base_model_grad
            dsigdx   = self.calc_dsigdx(sig, Kxy_invKernEta, dKxy_dx, sigK)
        else:
            dmudx    = dsigdx = None
            
        if calc_hess:
            d2mudx2  = self.calc_d2mudx2(d2Kxy_dx2) + base_model_hess
            d2sigdx2 = self.calc_d2sigdx2(sig, dsigdx, Kxy_invKernEta, dKxy_dx, d2Kxy_dx2, sigK)
        else:
            d2mudx2  = d2sigdx2 = None
        
        if self.b_use_data_scl:
            (mu, sig, dmudx, dsigdx, d2mudx2, d2sigdx2) \
                = self.data_scl_2_init(mu, sig, dmudx, dsigdx, d2mudx2, d2sigdx2)
        
        if squeeze_nx:
            mu  = mu[0]
            sig = sig[0]
            
            if calc_grad:
                dmudx  = dmudx[0,:]
                dsigdx = dsigdx[0,:]
                
            if calc_hess:
                d2mudx2  = d2mudx2[0,:,:]
                d2sigdx2 = d2sigdx2[0,:,:]
                
        return mu, sig, dmudx, dsigdx, d2mudx2, d2sigdx2

    def eval_model_var(self, x2model_in, calc_grad = False, calc_hess = False, squeeze_nx = False):
        '''
        Parameters
        ----------
        x2model_in : 2d numpy array of size [nx, dim], where nx is the number of points in the parameter space
            Each row is where the surrogate will be evaluated
        calc_grad : bool, optional
            If True the grad of the surrogate is evaluated. The default is False.
        calc_hess : bool, optional
            If True the Hessian of the surrogate is evaluated. The default is False.
        squeeze_nx : bool, optional
            If True then nx must be 1 and the returned arrays have their 
            first dimension collapsed. The default is False.

        Returns
        -------
        sig2 : 1d numpy array of length nx (or float if squeeze_nx is True)
            Standard deviation of the surrogate evaluated at the rows of x2model_in.
        dsig2dx : 2d numpy array of size [nx, dim]
            Gradient of the standard deviation of the surrogate evaluated at the rows of x2model_in.
        d2sig2dx2 : 3d numpy array of size [nx, dim, dim]
            Hessian of the standard deviation of the surrogate evaluated at the rows of x2model_in.
        '''
        
        ''' Check inputs '''
        
        if x2model_in.ndim == 1:
            x2model = x2model_in[None,:]
        elif x2model_in.ndim == 2:
            x2model = x2model_in
        else:
            raise Exception(f'x2model_in should be a 2d array but it has shape {x2model_in.shape}')
        
        nx = x2model.shape[0]
        
        if squeeze_nx:
            assert nx == 1, 'If squeeze_nx is True, then x_acq must only have one point'
        
        check_hp_vals = self.hp_vals == self._hp_vals_model_setup
        
        if check_hp_vals is False:
            print('hp_vals has changed since setup_eval_model() was called')
            print(f'hp_vals_model_setup = \n{self._hp_vals_model_setup}')
            print(f'hp_vals = \n{self.hp_vals}')
            raise Exception('Cannot change hp_vals between calling setup_eval_model() and eval_model()')

        theta     = self.hp_vals.theta 
        hp_kernel = self.hp_vals.kernel
        hp_beta   = self.hp_vals.beta

        '''Preliminaries '''
        
        varK = self.hp_vals.varK
        
        if self.b_use_data_scl:
            # x2model = self.DataScl.x_init_2_scl(x2model)
            raise Exception('The method eval_model_var() is not setup for cases where data must be rescaled')
            
        x_eval_scl = self.get_scl_x_w_dist()[0]
        
        if calc_hess: 
            assert calc_grad, 'To return the hessian calc_grad must also be set to True'

        ny      = self.n_eval
        nx      = x2model.shape[0]
        one_nx  = np.ones(nx)

        Rtensor = self.calc_Rtensor(x_eval_scl, x2model, 1)
        if self.use_grad or calc_grad:
            Kgrad_yx = self.calc_KernGrad(Rtensor, theta, hp_kernel, 
                                          bvec_use_grad1 = self.bvec_use_grad)
            
            if self.use_grad:
                Kyx      = Kgrad_yx[:, :nx]
                dKxy_dx  = Kgrad_yx[:, nx:].T
            else:
                Kyx       = Kgrad_yx[:ny, :nx]
                dKyx_dx   = Kgrad_yx[:ny, nx:]
                dKxy_dx   = dKyx_dx.T
        else:
            Kyx = self.calc_KernBase(Rtensor, theta, hp_kernel)
        
        Kxy = Kyx.T
        
        if calc_hess:
            Rtensor   = self.calc_Rtensor(x2model, x_eval_scl, 1)
            d2Kxy_dx2 = self.calc_Kern_hess_x(Rtensor, theta, hp_kernel) # * sigK
            
        Kxy_invKernEta = linalg.cho_solve(self.KernEta_chofac, Kyx).T

        ''' Solve for the mean and std of the model '''
        
        base_model_val, base_model_grad, base_model_hess \
            = self.eval_mean_fun(x2model, hp_beta, 
                                 calc_grad = calc_grad, calc_hess = calc_hess)
        
        sig2 = varK * (one_nx - np.diag(Kxy @ Kxy_invKernEta.T))
        assert np.min(sig2) >= 0, f'The variance of the surr should be non-negative but min(sig2) = {np.min(sig2)}'

        if calc_grad:
            dsig2dx = self.calc_dsig2dx(sig2, Kxy_invKernEta, dKxy_dx, varK)
        else:
            dsig2dx = None
            
        if calc_hess:
            raise Exception('Must add method to calculate d2sig2dx2')
        else:
            d2sig2dx2 = None
        
        if self.b_use_data_scl:
            raise Exception('The method eval_model_var() is not setup for cases where data must be rescaled')
        
        if squeeze_nx:
            sig2 = sig2[0]
            if calc_grad: dsig2dx   = dsig2dx[0,:]
            if calc_hess: d2sig2dx2 = d2sig2dx2[0,:,:]
                
        return sig2, dsig2dx, d2sig2dx2

    def calc_dmudx(self, dKxy_dx):
        
        nx        = int(dKxy_dx.shape[0] / (self.dim))
        dmudx_vec = dKxy_dx @ self.invKernEta_fdiff
        dmudx     = np.reshape(dmudx_vec, [nx, self.dim], order='f')
        
        return dmudx
    
    def calc_dsig2dx(self, sig, Kxy_invKernEta, dKxy_dx, varK):
            
            # Preliminaries
            nx      = int(dKxy_dx.shape[0] / (self.dim))
            one_d   = np.ones((self.dim, 1))
            one_ny  = np.ones(self.invKernEta_fdiff.size)
            
            dsig2dx = -2*np.dot((dKxy_dx * np.kron(one_d, Kxy_invKernEta)), one_ny) * varK
            dsig2dx = dsig2dx.reshape((nx, self.dim), order='f')
    
            return dsig2dx
    
    def calc_dsigdx(self, sig, Kxy_invKernEta, dKxy_dx, sigK):
            
            # Preliminaries
            nx      = int(dKxy_dx.shape[0] / (self.dim))
            one_d   = np.ones((self.dim, 1))
            one_ny  = np.ones(self.invKernEta_fdiff.size)
            
            inv_sig = np.divide(1, sig, out=np.zeros_like(sig), where=sig!=0)
            term1   = -np.kron(one_d.T, inv_sig[:,None])
    
            term2   = np.dot((dKxy_dx * np.kron(one_d, Kxy_invKernEta)), one_ny) * sigK**2
            term2   = term2.reshape((nx, self.dim), order='f')
    
            dsigdx  = term1 * term2
            
            return dsigdx

    def calc_d2mudx2(self, d2Kxy_dx2):
        
        n_data = self.n_eval * (1 + self.dim) if self.use_grad else self.n_eval
        assert d2Kxy_dx2.shape == (self.dim, self.dim, n_data), 'calc_d2mudx2 can only be used on one point per call'
        
        d2mudx2 = np.matmul(d2Kxy_dx2, self.invKernEta_fdiff[None,:,None])[:,:,0]
        
        return d2mudx2[None,:,:]
    
    def calc_d2sigdx2(self, sig, dsigdx, Kxy_invKernEta, dKxy_dx, d2Kxy_dx2, sigK):
        
        # Preliminaries
        n_data = self.n_eval * (1 + self.dim) if self.use_grad else self.n_eval
        assert d2Kxy_dx2.shape == (self.dim, self.dim, n_data), 'calc_d2sigdx2 can only be used on one point per call'
        
        # Calculate d2sig2_dx2
        term1       = np.matmul(d2Kxy_dx2, Kxy_invKernEta.T[None,:,:]) 
        term2       = dKxy_dx @ linalg.cho_solve(self.KernEta_chofac, dKxy_dx.T)
        d2sig2_dx2  = -2*sigK**2 * (term1[:,:,0] + term2)
        
        # Calculate d2sig_dx2
        sig_mod             = 1.0 * sig
        sig_mod[sig_mod==0] = np.nan
        
        d2sig_dx2 = (1/(2*sig_mod)) * (d2sig2_dx2 - 2*np.outer(dsigdx, dsigdx))
        
        return d2sig_dx2[None,:,:]
        
    def calc_model_mean_w_data_init(self, x2model_init, KernEta_chofac, hp_beta, 
                                    xeval_init, fval_init, grad_init = None, 
                                    theta = None, calc_grad = True): 
        
        x2model_scl           = self.x_init_2_scl(x2model_init)
        xeval_scl             = self.x_init_2_scl(xeval_init)
        fval_scl, _, grad_scl = self.data_init_2_scl(fval_init, None, grad_init)[:3]
        
        return self.calc_model_mean_w_data_scl(x2model_scl, KernEta_chofac, hp_beta, 
                                               xeval_scl, fval_scl, grad_scl, theta, calc_grad)
        
    def calc_model_mean_w_data_scl(self, x2model_scl, KernEta_chofac, hp_beta,
                                   xeval_scl, fval_scl, grad_scl = None, 
                                   theta = None, calc_grad = True): 
        
        Kgrad_xy = self.calc_KernGrad(x2model_scl, xeval_scl, theta)
        
        ny = fval_scl.size
        nx = int(Kgrad_xy.shape[0] / (self.dim + 1))

        mean_fval, mean_fgrad = self.eval_mean_fun(xeval_scl, hp_beta, self.use_grad)[:2]
        mean_fun_vec          = self.make_data_vec(mean_fval, mean_fgrad)
        
        x2model_base_mu_val, x2model_base_mu_grad \
            = self.eval_mean_fun(x2model_scl, hp_beta, calc_grad)[:2]
        
        if self.use_grad:
            assert grad_scl is not None, 'grad_scl should not be None when use_grad is True'
            
            n_data = ny * (self.dim + 1)
            assert n_data == KernEta_chofac[0].shape[0], 'Dimensions of matrices do not match'
            
            Kxy_base    = Kgrad_xy[:nx, :]
            dKxy_dx     = Kgrad_xy[nx:, :]
            
            grad_vec    = np.reshape(grad_scl, grad_scl.size, order='f')
            data_vec    = np.hstack((fval_scl, grad_vec))
        else:
            assert ny == KernEta_chofac[0].shape[0], 'Dimensions of matrices do not match'
            
            Kxy_base    = Kgrad_xy[:nx, :ny]
            dKxy_dx     = Kgrad_xy[nx:, :ny]
            data_vec    = fval_scl
        
        f_diff   = data_vec - mean_fun_vec
        sol1     = linalg.cho_solve(KernEta_chofac, f_diff)
        model_mu = x2model_base_mu_val + np.dot(Kxy_base, sol1)
        
        if calc_grad:
            model_dmudx = x2model_base_mu_grad + np.reshape(dKxy_dx @ sol1, [nx, self.dim], order='f') 
        else:
            model_dmudx = None
            
        model_mu, _, model_dmudx \
            = self.data_scl_2_init(model_mu, None, model_dmudx)[:3]

        return model_mu, model_dmudx
    