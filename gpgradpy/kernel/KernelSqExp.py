#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:31:28 2021

@author: andremarchildon
"""

import numpy as np
from numba import jit

from . import KernelCommon

class KernelSqExp:

    sq_exp_hp_kernel_default = None
    sq_exp_range_hp_kernel   = [np.nan, np.nan]

    @staticmethod
    def sq_exp_theta2gamma(theta):
        # Convert hyperparameter theta to gamma
        return np.sqrt(2 * theta) 
    
    @staticmethod
    def sq_exp_gamma2theta(gamma):
        # Convert hyperparameter gamma to theta
        return 0.5 * gamma**2
    
    @staticmethod
    def sq_exp_Kern_precon(theta, n_eval, calc_grad = False, b_return_vec = False):
        
        # Calculate the precondition matrix
        gamma    = KernelSqExp.sq_exp_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n_eval), np.kron(gamma, np.ones(n_eval))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 1 / gamma
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n_eval, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernBase(Rtensor, theta, hp_kernel):
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
        exp_sum     = np.zeros((n1, n2))
        
        for i in range(dim):
            exp_sum -= theta[i] * Rtensor[i,:,:]**2
        
        KernBase = np.exp(exp_sum)

        return KernBase

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad : 2d numpy array of floats
            Gradient-enhanced kernel matrix.
        '''

        ''' Calculate the base Kernel '''
        
        dim, n1, n2 = Rtensor.shape      
        exp_sum     = np.zeros((n1, n2))
        Rtensor_sq  = Rtensor**2
        
        for i in range(dim):
            exp_sum -= theta[i] * Rtensor_sq[i,:,:]
        
        KernBase = np.exp(exp_sum)

        ''' Calculate the Kernel '''

        KernGrad = np.zeros((n1*(dim+ 1), n2*(dim + 1)))
        KernGrad[:n1, :n2] = KernBase

        for ir in range(dim):
            
            r1 = (ir + 1) * n1
            r2 = r1 + n1
            
            c1 = (ir + 1) * n2
            c2 = c1 + n2
            
            # Covariance obj with grad
            term = 2*theta[ir] * Rtensor[ir] * KernBase
            KernGrad[r1:r2, :n2] = -term
            KernGrad[:n1, c1:c2] =  term
            
            # Covariance grad with grad for ir == ic
            KernGrad[r1:r2,c1:c2] = (2 * theta[ir] -4*theta[ir]**2 * Rtensor[ir,:,:]**2) * KernBase
            
            # Covariance grad with grad for ir != ic
            for ic in range(ir+1, dim):
                term = -4 * theta[ir] * theta[ic] * (Rtensor[ir] * Rtensor[ic] * KernBase) 
                KernGrad[r1:r2,               n2*(ic+1):n2*(ic+2)] += term
                KernGrad[n1*(ic+1):n1*(ic+2), n2*(ir+1):n2*(ir+2)] += term

        return KernGrad

    @staticmethod
    def sq_exp_calc_KernBase_hess_x(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_hess_x : 3d numpy array of floats
            Derivative of KernBase wrt first set of nodal locations X
        '''
    
        KernBase = KernelSqExp.sq_exp_calc_KernBase(Rtensor, theta, hp_kernel)
    
        return KernelSqExp.sq_exp_calc_KernBase_hess_x_jit(Rtensor, theta, hp_kernel, KernBase)
    
    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernBase_hess_x_jit(Rtensor, theta, hp_kernel, KernBase):
        '''
        See sq_exp_calc_KernBase_hess_x() for documentation
        '''
        
        # Returns the derivative of Kbase wrt its first argument X
        # This is used to calculate the Hessian of the surrogate

        dim, n1, n2 = Rtensor.shape
        KernBase_hess_x = np.zeros((dim, n1*dim, n2))
        
        for k in range(dim):
            for i in range(dim):
                delta_ik = int(i == k)
            
                r1 = i * n1
                r2 = r1 + n1
                
                KernBase_hess_x[k, r1:r2, :] \
                    = (-2 * theta[i] * delta_ik + 4*theta[i] * theta[k] * Rtensor[i,:,:] * Rtensor[k,:,:]) * KernBase
                    
        return KernBase_hess_x  

    @staticmethod
    def sq_exp_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_x : 3d numpy array of floats
            Derivative of KernGrad wrt first set of nodal locations X
        '''
        
        KernBase        = KernelSqExp.sq_exp_calc_KernBase(Rtensor, theta, hp_kernel)
        KernBase_hess_x = KernelSqExp.sq_exp_calc_KernBase_hess_x_jit(Rtensor, theta, hp_kernel, KernBase)
        
        return KernelSqExp.sq_exp_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase, KernBase_hess_x)
        
    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase, KernBase_hess_x):
        
        dim, n1, n2 = Rtensor.shape
        
        KernGrad_grad_x          = np.zeros((dim, n1*dim, n2*(1+dim)))
        KernGrad_grad_x[:,:,:n2] = KernBase_hess_x
        
        for k in range(dim):
            th_k = theta[k]
            Tk   = Rtensor[k,:,:]
            
            for i in range(dim):
                th_i     = theta[i]
                delta_ik = int(i == k)
                Ti       = Rtensor[i,:,:]
                
                r1 = i * n1
                r2 = r1 + n1
                
                for j in range(dim):
                    th_j     = theta[j]
                    delta_jk = int(j == k)
                    delta_ij = int(i == j)
                    Tj       = Rtensor[j,:,:]
                    
                    c1 = (j+1)*n2
                    c2 = c1 + n2 
                    
                    KernGrad_grad_x[k, r1:r2, c1:c2] \
                        = (-4* th_i * th_j * (delta_ik * Tj + delta_jk * Ti)
                            - 4 * delta_ij * th_i * th_k * Tk
                            + 8 * th_i * th_j * th_k * Ti * Tj * Tk) * KernBase
                     
        return KernGrad_grad_x  

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernBase_grad_th(KernBase, Rtensor, theta, hp_kernel):  
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_grad_th : 3d numpy array of floats
            Derivative of KernBase wrt theta
        '''
        
        ''' Check input parameters '''
        
        assert np.max(np.diag(KernBase)) < (1.0 + 1e-8), 'The correlation matrix without an2 noise should be provided'

        ''' Calculate the gradient of the Kernel '''

        dim = theta.size
        KernBase_grad_th = np.zeros(Rtensor.shape)

        for i in range(dim):
            KernBase_grad_th[i,:,:] = -Rtensor[i,:,:]**2 * KernBase

        return KernBase_grad_th

    @staticmethod
    def sq_exp_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''
        
        KernGrad = KernelSqExp.sq_exp_calc_KernGrad(Rtensor, theta, hp_kernel)
        
        return KernelSqExp.sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad)

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad):
        '''
        See method sq_exp_calc_KernGrad_grad_th() for documentation
        '''

        [dim, n1, n2] = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes'

        KernBase         = KernGrad[:n1, :n1]
        dist_sq_all      = Rtensor**2
        KernGrad_grad_th = np.zeros((dim, n1*(dim+1), n1*(dim+1)))
        
        for d in range(dim):
            
            # Covariance obj with obj
            KernGrad_grad_th[d, :n1, :n1] = -dist_sq_all[d,:,:] * KernBase
            
            d1 = (d + 1) * n1
            d2 = d1 + n1
            
            # Covariance obj with grad for d == ir
            term = 2 * Rtensor[d,:,:] * KernBase
            KernGrad_grad_th[d, d1:d2, :n1] -= term
            KernGrad_grad_th[d, :n1, d1:d2] += term
            
            # Covariance grad with grad for case of d == ir == ic
            KernGrad_grad_th[d, d1:d2, d1:d2] += 2 * KernBase
            
            for ir in range(dim):
                r1 = (ir + 1) * n1
                r2 = r1 + n1 
                
                # Covariance obj with grad
                KernGrad_grad_th[d, r1:r2, :n1] -= dist_sq_all[d,:,:] * KernGrad[r1:r2, :n1] 
                KernGrad_grad_th[d, :n1, r1:r2] -= dist_sq_all[d,:,:] * KernGrad[:n1, r1:r2] 
                
                # Covariance grad with grad when ir == d and ic == d
                term = -4 * theta[ir] * Rtensor[d,:,:] * Rtensor[ir,:,:] * KernBase
                KernGrad_grad_th[d, d1:d2, r1:r2] += term
                KernGrad_grad_th[d, r1:r2, d1:d2] += term
                
                # Covariance grad with grad when ir == ic
                KernGrad_grad_th[d, r1:r2, r1:r2] -= dist_sq_all[d,:,:] * KernGrad[r1:r2, r1:r2]
                    
                # Covariance grad with grad when ir != ic
                for ic in range(ir+1,dim):
                    c1 = (ic + 1) * n1 
                    c2 = c1 + n1
                    term = -dist_sq_all[d,:,:] * KernGrad[r1:r2, c1:c2]
                    KernGrad_grad_th[d, r1:r2, c1:c2] += term
                    KernGrad_grad_th[d, c1:c2, r1:r2] += term
                
        return KernGrad_grad_th
    
    @staticmethod
    def sq_exp_calc_KernBase_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the squared exponential kernel')
    
    @staticmethod
    def sq_exp_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the squared exponential kernel')
  