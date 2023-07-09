#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:59 2023

@author: andremarchildon
"""

import numpy as np 
from numba import jit
from scipy.spatial.distance import cdist

class CommonFun:
    
    @staticmethod
    def calc_dist_min(Xmat):
        '''
        Parameters
        ----------
        Xmat : 2D numpy array of size [nx, dim]
            Each row is one point in the parameter space.
        Returns
        -------
        dist_min : float
            Min dist between data points.
        '''
        
        nx = Xmat.shape[0]
        
        if nx == 1:
            return np.nan
        else:
            x_dist = cdist(Xmat, Xmat, 'euclidean') + np.diag(np.full(nx, np.nan))
            return np.nanmin(x_dist)
    
    @staticmethod
    def calc_dist_max(Xmat):
        '''
        Parameters
        ----------
        Xmat : 2D numpy array of size [nx, dim]
            Each row is one point in the parameter space.
        Returns
        -------
        dist_max : float
            Max dist between data points.
        '''
        nx = Xmat.shape[0]
        
        if nx == 1:
            return np.nan
        else:
            x_dist = cdist(Xmat, Xmat, 'euclidean')
            return np.nanmax(x_dist)

    @staticmethod
    @jit(nopython=True)
    def calc_Rtensor(X, Y, exp=1):
        '''
        Parameters
        ----------
        X : np array of floats of size [n1, dim].
        Y : np array of floats of size [n2, dim].
        exp : integer
            The elementwise exponent
            Default is 1 since this simply returns the distance between X1 and X2
            
        Returns
        -------
        Rtensor: np array of floats of size [dim, n1, n2]
            Difference beween X and Y in each of the coordinates
        '''

        n1, dim  = X.shape
        n2, dim2 = Y.shape

        assert dim == dim2, 'The dimensions of the arrays do not match'

        Rtensor = np.zeros((dim, n1, n2))

        for d in range(dim):
            Rtensor[d, :,:] = X[:,np.array([d])] - Y[:,np.array([d])].T

        return Rtensor
    
    @staticmethod
    def test_grad_calc(x0_in, fh_fun, fh_grad, eps = 1e-8, print_calc = True, calc_cent_diff = True):
        '''
        Parameters
        ----------
        x0_in : 1D numpy array of floats of length dim.
            Point at which to check the gradient.
        fh_fun : function handle 
            Calculates the value of the function of interest.
        fh_grad : function handle 
            Calculates the gradient of the function of interest.
        eps : float, optional
            Finite difference step size. The default is 1e-8.
        print_calc : bool, optional
            Set to True to print result of the test. The default is True.
        calc_cent_diff : bool, optional
            Set to true to use centre finite difference. The default is True.

        Returns
        -------
        fd_grad : float
            Finite difference approximation to the gradient.
        grad_x0 : float
            Grad evaluation at x0_in from fh_grad.

        '''
        
        x0  = np.atleast_1d(x0_in).astype('float64')
        dim = x0.size 
        
        fun_x0  = np.atleast_1d(fh_fun(x0))
        grad_x0 = np.atleast_1d(fh_grad(x0))
        
        n_fun   = fun_x0.size
        fd_grad = np.zeros((n_fun, dim))
        
        for i in range(dim):
            x0_i      = 1 * x0
            x0_i[i]  += eps 
            fun_p_eps = np.atleast_1d(fh_fun(x0_i))
            
            if calc_cent_diff:
                x0_i      = 1 * x0
                x0_i[i]  -= eps 
                fun_m_eps = np.atleast_1d(fh_fun(x0_i))
                
                fd_grad[:,i] = (fun_p_eps - fun_m_eps) / (2*eps)
                
            else:
                fd_grad[:,i] = (fun_p_eps - fun_x0) / eps
        
        if grad_x0.ndim == 1:
            assert n_fun == 1, 'If n_fun > 1 then eval_grad_x0 must be a 2D array'
            fd_grad = fd_grad[0,:]
            
        diff_fd = grad_x0 - fd_grad
        
        if print_calc:
            print(f'FD test: eval x0: {fun_x0}, x0 = {x0}')
            print(f' FD:   {fd_grad}')
            print(f' Grad: {grad_x0}')
            print(f' Diff: {diff_fd}')
        
        return fd_grad, grad_x0

    @staticmethod
    def make_data_vec(fval, fgrad = None):
        '''
        Parameters
        ----------
        fval : 1D numpy array of length nx
            Function evaluation at nx points.
        fgrad : 2D numpy array of size [nx, dim], optional
            Gradient evaluations at nx points

        Returns
        -------
        data_vec : 1D numpy array
            Appended data of fval and fgrad in 1D format.
        '''
        
        if fgrad is None:
            data_vec = np.atleast_1d(fval)
        else:
            vec_fgrad = fgrad.reshape(fgrad.size, order='f')
            data_vec  = np.hstack((fval, vec_fgrad))
                
        return data_vec
    