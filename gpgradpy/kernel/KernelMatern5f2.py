#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:38:59 2022

@author: andremarchildon
"""

import numpy as np
from numba import jit

from . import KernelCommon

class KernelMatern5f2:

    matern_5f2_hp_kernel_default = None
    matern_5f2_range_hp_kernel   = [np.nan, np.nan]
    
    @staticmethod
    def matern_5f2_theta2gamma(theta):
        # Convert hyperparameter theta to gamma
        return np.sqrt((5.0/3.0) * theta)
    
    @staticmethod
    def matern_5f2_gamma2theta(gamma):
        # Convert hyperparameter gamma to theta
        return (3.0/5.0) * gamma**2

    @staticmethod
    def matern_5f2_Kern_precon(theta, n1, calc_grad = False, b_return_vec = False):
        # Calculate the precondition matrix
        
        gamma    = KernelMatern5f2.matern_5f2_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n1), np.kron(gamma, np.ones(n1))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 5.0 / (6.0 * gamma)
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n1, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon

    @staticmethod
    def matern_5f2_calc_nu_mat(theta, Rtensor_sq):
        return np.sqrt(np.tensordot(theta, Rtensor_sq, axes=1))
    
    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernBase(Rtensor, theta, hp_kernel, nu_mat = None):
        '''
        Parameters
        ----------
        Rtensor : 3d numpy array of floats
            Relative distance between two sets of data points in each directions
            Size is [dim,n1,n2], where 
                dim is the no. of dimensions
                n1 and n2 are the no. of data points in the 1st and 2nd data sets, respectively
        theta : 1d numpy array of positive floats
            Hyperparameters related to the characteristic lengths.
        hp_kernel : float or None
            Hyperparameter for the kernel, not used for this kernel.

        Returns
        -------
        KernBase : 2d numpy array of floats
            Gradient-free kernel matrix.
        '''
        
        dim, n1, n2 = Rtensor.shape

        ''' Calculate the Kernel '''
        
        if nu_mat is None:
            nu_mat = np.zeros((n1, n2))
            
            for i in range(dim):
                nu_mat += theta[i] * Rtensor[i,:,:]**2
            
            nu_mat = np.sqrt(nu_mat)
        
        sqrt5    = np.sqrt(5)
        KernBase = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * np.exp(-sqrt5 * nu_mat)
            
        return KernBase

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad : 2d numpy array of floats
            Gradient-enhanced kernel matrix.
        '''
        
        dim, n1, n2 = Rtensor.shape
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)

        Abase     = np.exp(-np.sqrt(5) * nu_mat)
        sqrt5     = np.sqrt(5)
        mat_corr_obj_grad = (5.0/3.0) * (1 + sqrt5 * nu_mat) * Abase
        
        KernGrad = np.zeros((n1*(dim+ 1), n2*(dim + 1)))
        KernGrad[:n1, :n2] = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * Abase
        
        for ir in range(dim):
            r1 = (ir + 1) * n1
            r2 = r1 + n1
            
            c1 = (ir + 1) * n2
            c2 = c1 + n2
            
            # Covariance obj with grad
            term = mat_corr_obj_grad * Rtensor[ir,:,:] * theta[ir]
            KernGrad[r1:r2, :n2] = -term
            KernGrad[:n1, c1:c2] =  term
            
            # Covariance grad with grad for ir == ic
            KernGrad[r1:r2,c1:c2] = mat_corr_obj_grad * theta[ir] - (25./3.) * theta[ir]**2 * Rtensor[ir,:,:]**2 * Abase
            
            # Covariance grad with grad for ir != ic
            for ic in range(ir+1,dim):
                term = -(25./3.) * theta[ir] * theta[ic] * Rtensor[ir,:,:] * Rtensor[ic,:,:] * Abase
                KernGrad[r1:r2,               n2*(ic+1):n2*(ic+2)] += term
                KernGrad[n1*(ic+1):n1*(ic+2), n2*(ir+1):n2*(ir+2)] += term
                
        return KernGrad
    
    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernBase_hess_x(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_hess_x : 3d numpy array of floats
            Derivative of KernBase wrt first set of nodal locations X
        '''
        
        dim, n1, n2 = Rtensor.shape
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        base   = np.exp(-np.sqrt(5) * nu_mat)
        A      = (5/3) * (1 + np.sqrt(5) * nu_mat) * base
        
        KernBase_hess_x = np.zeros((dim, n1*dim, n2))
        
        for k in range(dim):
            Rk_til = Rtensor[k,:,:] * theta[k]
            
            KernBase_hess_x[k, k*n1:(k+1)*n1, :] -= theta[k] * A 
            
            for i in range(dim):
                th_i   = theta[i]
                Ri_til = Rtensor[i,:,:] * th_i
            
                r1 = i * n1
                r2 = r1 + n1
                
                KernBase_hess_x[k, r1:r2, :] += (25/3) * Ri_til * Rk_til * base
                    
        return KernBase_hess_x  

    @staticmethod
    def matern_5f2_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_x : 3d numpy array of floats
            Derivative of KernGrad wrt first set of nodal locations X
        '''
        
        KernBase_hess_x = KernelMatern5f2.matern_5f2_calc_KernBase_hess_x(Rtensor, theta, hp_kernel)
        
        return KernelMatern5f2.matern_5f2_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase_hess_x)

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase_hess_x):
        ''' See matern_5f2_calc_KernGrad_grad_x() for documentation '''
        
        dim, n1, n2 = Rtensor.shape

        ''' Calculate the Kernel '''
        
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        
        KernGrad_grad_x          = np.zeros((dim, n1*dim, n2*(1+dim)))
        KernGrad_grad_x[:,:,:n2] = KernBase_hess_x

        base = np.exp(-np.sqrt(5) * nu_mat)
        
        # Take the inverse of nu_mat without causing error for values of zero
        # Use 1e-16 as min value, does not change final calculations since
        # nu_mat[i,j] is zero iff Rtensor[k,i,j] is zero for all 0<=k<=(dim-1) 
        inv_nu_mat = 1 / np.maximum(nu_mat, 1e-16)

        for k in range(dim):
            th_k   = theta[k]
            Rk_til = Rtensor[k,:,:] * th_k
            
            for i in range(dim):
                th_i     = theta[i]
                delta_ik = int(i == k)
                Ri_til   = Rtensor[i,:,:] * th_i
                
                r1 = i * n1
                r2 = r1 + n1
                
                for j in range(dim):
                    th_j     = theta[j]
                    delta_jk = int(j == k)
                    delta_ij = int(i == j)
                    Rj_til   = Rtensor[j,:,:] * th_j
                    
                    c1 = (j+1)*n2
                    c2 = c1 + n2 
                    
                    KernGrad_grad_x[k, r1:r2, c1:c2] \
                        = -(25/3) * (th_i * delta_ik * Rj_til + th_j * delta_jk * Ri_til + th_i * delta_ij * Rk_til 
                                     - np.sqrt(5) * inv_nu_mat * Ri_til * Rj_til * Rk_til) * base
        
        return KernGrad_grad_x  
    
    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernBase_grad_th(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_grad_th : 3d numpy array of floats
            Derivative of KernBase wrt theta
        '''

        dim, n1, n2 = Rtensor.shape
        assert n1 == n2, 'n1 and n2 must match'
        
        nu_mat = np.zeros((n1, n1))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        
        # Take the inverse of nu_mat without causing error for values of zero
        # Use 1e-16 as min value, does not change final calculations since
        # nu_mat[i,j] is zero iff Rtensor[k,i,j] is zero for all 0<=k<=(dim-1) 
        inv_nu_mat = 1 / (2*np.maximum(nu_mat, 1e-16))

        sqrt5      = np.sqrt(5)
        Rtensor_sq = Rtensor**2
        KernBase   = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * np.exp(-sqrt5 * nu_mat)

        KernBase_grad_th = np.zeros((dim, n1, n1))

        for d in range(dim):
            KernBase_grad_th[d,:,:] = -sqrt5 * Rtensor_sq[d,:,:] * inv_nu_mat * KernBase

        return KernBase_grad_th
    
    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''

        dim, n1, n2 = Rtensor.shape
        assert n1 == n2, 'n1 and n2 must match'
        
        Rtensor_sq = Rtensor**2
        nu_mat = np.zeros((n1, n1))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        
        # Take the inverse of nu_mat without causing error for values of zero
        # Use 1e-16 as min value, does not change final calculations since
        # nu_mat[i,j] is zero iff Rtensor[k,i,j] is zero for all 0<=k<=(dim-1) 
        inv_nu_mat = 1 / np.maximum(nu_mat, 1e-16)
        
        sqrt5   = np.sqrt(5)
        mat_exp = np.exp(-sqrt5 * nu_mat)
        mat_1   = (5./3.) * (1 + sqrt5 * nu_mat) * mat_exp

        KernGrad_grad_th = np.zeros((dim, n1*(dim+1), n1*(dim+1)))
        
        for d in range(dim):
            # Covariance obj with obj
            KernGrad_grad_th[d,:n1,:n1] = -(5/6) * Rtensor_sq[d,:,:] * (1 + sqrt5*nu_mat) * mat_exp
            
            d1 = (d + 1) * n1
            d2 = d1 + n1
            
            # Covariance obj with grad for d == ir
            term = mat_1 * Rtensor[d,:,:]
            KernGrad_grad_th[d, d1:d2, :n1] -= term
            KernGrad_grad_th[d, :n1, d1:d2] += term
            
            # Covariance grad with grad for case of d == ir == ic
            KernGrad_grad_th[d, d1:d2, d1:d2] += mat_1
            
            for ir in range(dim):
                r1 = (ir + 1) * n1
                r2 = r1 + n1
                
                # Covariance obj with grad
                term = -(25. / 6.) * theta[ir] * Rtensor[ir] * Rtensor_sq[d,:,:] * mat_exp
                KernGrad_grad_th[d, r1:r2, :n1] -= term
                KernGrad_grad_th[d, :n1, r1:r2] += term
                
                # Covariance grad with grad when ir == d and ic == d
                term = -((25./3.) * theta[ir]) * Rtensor[ir] * Rtensor[d] * mat_exp
                KernGrad_grad_th[d, d1:d2, r1:r2] += term
                KernGrad_grad_th[d, r1:r2, d1:d2] += term
                
                # Covariance grad with grad when ir == ic
                KernGrad_grad_th[d, r1:r2, r1:r2] -= ((25./6.) * theta[ir]) * Rtensor_sq[d,:,:] * mat_exp
                KernGrad_grad_th[d, r1:r2, r1:r2] += ((25 * sqrt5 / 6.) * theta[ir]**2) * Rtensor_sq[ir] * Rtensor_sq[d,:,:] * inv_nu_mat * mat_exp
                    
                for ic in range(ir+1,dim):
                    c1 = (ic + 1) * n1
                    c2 = c1 + n1
                    term = ((25 * sqrt5 / 6.) * theta[ir] * theta[ic]) * Rtensor[ir] * Rtensor[ic] * Rtensor_sq[d,:,:] * inv_nu_mat * mat_exp
                    KernGrad_grad_th[d, r1:r2, c1:c2] += term
                    KernGrad_grad_th[d, c1:c2, r1:r2] += term
                
        return KernGrad_grad_th
            
    @staticmethod
    def matern_5f2_calc_KernBase_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the Matern 5/2 kernel')
        
    @staticmethod
    def matern_5f2_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the Matern 5/2 kernel')
