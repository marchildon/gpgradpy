#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 13:31:28 2021

@author: andremarchildon
"""

import numpy as np
from numba import jit

from . import KernelCommon

class KernelSqExpBase:

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
            exp_sum -= theta[i] * Rtensor[i]**2
        
        KernBase = np.exp(exp_sum)

        return KernBase

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
            
                ri1 = i * n1
                ri2 = ri1 + n1
                
                KernBase_hess_x[k, ri1:ri2, :] \
                    = (-2 * theta[i] * delta_ik + 4*theta[i] * theta[k] * Rtensor[i,:,:] * Rtensor[k,:,:]) * KernBase
                    
        return KernBase_hess_x  
        
    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernBase_grad_th(Rtensor, theta, hp_kernel):  
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_grad_th : 3d numpy array of floats
            Derivative of KernBase wrt theta
        '''
        
        ''' Calculate KernBase '''
        
        dim, n1, n2 = Rtensor.shape
        exp_sum     = np.zeros((n1, n2))
        Rtensor_sq  = Rtensor**2
        
        for i in range(dim):
            exp_sum -= theta[i] * Rtensor_sq[i,:,:]
        
        KernBase = np.exp(exp_sum)

        ''' Calculate the gradient of the Kernel '''

        dim = theta.size
        KernBase_grad_th = np.zeros(Rtensor.shape)

        for i in range(dim):
            KernBase_grad_th[i,:,:] = -Rtensor_sq[i,:,:] * KernBase

        return KernBase_grad_th

    @staticmethod
    def sq_exp_calc_KernBase_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the squared exponential kernel')
    
class KernelSqExpGrad:

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

        for i in range(dim):
            
            ri1 = (i + 1) * n1
            ri2 = ri1 + n1
            
            c1a = (i + 1) * n2
            c2a = c1a + n2
            
            # Covariance obj with grad
            term = 2*theta[i] * Rtensor[i] * KernBase
            KernGrad[ri1:ri2, :n2] = -term
            KernGrad[:n1, c1a:c2a] =  term
            
            # Covariance grad with grad for i == j
            KernGrad[ri1:ri2,c1a:c2a] = (2 * theta[i] -4*theta[i]**2 * Rtensor[i,:,:]**2) * KernBase
            
            # Covariance grad with grad for i != j
            for j in range(i+1, dim):
                term = -4 * theta[i] * theta[j] * (Rtensor[i] * Rtensor[j] * KernBase) 
                KernGrad[ri1:ri2,               n2*(j+1):n2*(j+2)] += term
                KernGrad[n1*(j+1):n1*(j+2), n2*(i+1):n2*(i+2)] += term

        return KernGrad

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
        
        KernBase        = KernelSqExpBase.sq_exp_calc_KernBase(Rtensor, theta, hp_kernel)
        KernBase_hess_x = KernelSqExpBase.sq_exp_calc_KernBase_hess_x_jit(Rtensor, theta, hp_kernel, KernBase)
        
        return KernelSqExpGrad.sq_exp_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase, KernBase_hess_x)
        
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
                
                ri1 = i * n1
                ri2 = ri1 + n1
                
                for j in range(dim):
                    th_j     = theta[j]
                    delta_jk = int(j == k)
                    delta_ij = int(i == j)
                    Tj       = Rtensor[j,:,:]
                    
                    c1a = (j+1)*n2
                    c2a = c1a + n2 
                    
                    KernGrad_grad_x[k, ri1:ri2, c1a:c2a] \
                        = (-4* th_i * th_j * (delta_ik * Tj + delta_jk * Ti)
                            - 4 * delta_ij * th_i * th_k * Tk
                            + 8 * th_i * th_j * th_k * Ti * Tj * Tk) * KernBase
                     
        return KernGrad_grad_x  

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
        
        KernGrad = KernelSqExpGrad.sq_exp_calc_KernGrad(Rtensor, theta, hp_kernel)
        
        return KernelSqExpGrad.sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad)

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad):
        '''
        See method sq_exp_calc_KernGrad_grad_th() for documentation
        '''

        [dim, n1, n2] = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes'

        KernBase         = KernGrad[:n1, :n1]
        Rtensor_sq      = Rtensor**2
        KernGrad_grad_th = np.zeros((dim, n1*(dim+1), n1*(dim+1)))
        
        for d in range(dim):
            
            # Covariance obj with obj
            KernGrad_grad_th[d, :n1, :n1] = -Rtensor_sq[d,:,:] * KernBase
            
            d1 = (d + 1) * n1
            d2 = d1 + n1
            
            # Covariance obj with grad for d == i
            term = 2 * Rtensor[d,:,:] * KernBase
            KernGrad_grad_th[d, d1:d2, :n1] -= term
            KernGrad_grad_th[d, :n1, d1:d2] += term
            
            # Covariance grad with grad for case of d == i == j
            KernGrad_grad_th[d, d1:d2, d1:d2] += 2 * KernBase
            
            for i in range(dim):
                ri1 = (i + 1) * n1
                ri2 = ri1 + n1 
                
                # Covariance obj with grad
                KernGrad_grad_th[d, ri1:ri2, :n1] -= Rtensor_sq[d,:,:] * KernGrad[ri1:ri2, :n1] 
                KernGrad_grad_th[d, :n1, ri1:ri2] -= Rtensor_sq[d,:,:] * KernGrad[:n1, ri1:ri2] 
                
                # Covariance grad with grad when i == d and j == d
                term = -4 * theta[i] * Rtensor[d,:,:] * Rtensor[i,:,:] * KernBase
                KernGrad_grad_th[d, d1:d2, ri1:ri2] += term
                KernGrad_grad_th[d, ri1:ri2, d1:d2] += term
                
                # Covariance grad with grad when i == j
                KernGrad_grad_th[d, ri1:ri2, ri1:ri2] -= Rtensor_sq[d,:,:] * KernGrad[ri1:ri2, ri1:ri2]
                    
                # Covariance grad with grad when i != j
                for j in range(i+1,dim):
                    c1a = (j + 1) * n1 
                    c2a = c1a + n1
                    term = -Rtensor_sq[d,:,:] * KernGrad[ri1:ri2, c1a:c2a]
                    KernGrad_grad_th[d, ri1:ri2, c1a:c2a] += term
                    KernGrad_grad_th[d, c1a:c2a, ri1:ri2] += term
                
        return KernGrad_grad_th
    
    @staticmethod
    def sq_exp_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the squared exponential kernel')


class KernelSqExpGradMod:

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad(Rtensor, theta, hp_kernel, 
                             bvec_use_grad1 = None, bvec_use_grad2 = None):
        
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
        
        ''' Modify terms for case where not all grads are used '''
        
        if (bvec_use_grad1 is None) and (bvec_use_grad2 is None):
            n1_grad = n1
            n2_grad = n2
            
            Rtensor_g1  = Rtensor_g2  = Rtensor_gg  = Rtensor
            KernBase_g1 = KernBase_g2 = KernBase_gg = KernBase
        else:
            if bvec_use_grad1 is None:
                n1_grad     = n1
                Rtensor_g1  = Rtensor
                KernBase_g1 = KernBase
            else:
                n1_grad     = np.sum(bvec_use_grad1)
                Rtensor_g1  = Rtensor[:, bvec_use_grad1,:]
                KernBase_g1 = KernBase[bvec_use_grad1, :]
            
            if bvec_use_grad2 is None:
                n2_grad     = n2
                Rtensor_g2  = Rtensor
                Rtensor_gg  = Rtensor_g1
                KernBase_gg = KernBase_g1
                KernBase_g2 = KernBase
            else:
                n2_grad     = np.sum(bvec_use_grad2)
                Rtensor_g2  = Rtensor[:,:, bvec_use_grad2]
                Rtensor_gg  = Rtensor_g1[:,:, bvec_use_grad2]
                
                KernBase_g2 = KernBase[:, bvec_use_grad2]
                KernBase_gg = KernBase_g1[:, bvec_use_grad2]
    
        ''' Calculate the Kernel '''
    
        KernGrad = np.zeros((n1 + n1_grad * dim, n2 + n2_grad * dim))
        KernGrad[:n1, :n2] = KernBase
    
        for i in range(dim):
            ri1 = n1 + i * n1_grad
            ri2 = ri1 + n1_grad
            
            ci1 = n2 + i * n2_grad
            ci2 = ci1 + n2_grad
            
            # Covariance obj with grad
            KernGrad[ri1:ri2, :n2] = -2*theta[i] * Rtensor_g1[i] * KernBase_g1
            KernGrad[:n1, ci1:ci2] =  2*theta[i] * Rtensor_g2[i] * KernBase_g2
            
            # Covariance grad with grad for i == j
            KernGrad[ri1:ri2,ci1:ci2] = (2 * theta[i] -4*theta[i]**2 * Rtensor_gg[i,:,:]**2) * KernBase_gg
            
            # Covariance grad with grad for i != j
            for j in range(i+1, dim):
                rj1 = n1 + j * n1_grad
                rj2 = rj1 + n1_grad
                
                cj1 = n2 + j * n2_grad
                cj2 = cj1 + n2_grad
                
                term = -4 * theta[i] * theta[j] * (Rtensor_gg[i] * Rtensor_gg[j] * KernBase_gg) 
                KernGrad[ri1:ri2, cj1:cj2] += term
                KernGrad[rj1:rj2, ci1:ci2] += term
    
        return KernGrad
    
    @staticmethod
    def sq_exp_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel, bvec_use_grad2 = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_x : 3d numpy array of floats
            Derivative of KernGrad wrt first set of nodal locations X
        '''
        
        KernBase        = KernelSqExpBase.sq_exp_calc_KernBase(Rtensor, theta, hp_kernel)
        KernBase_hess_x = KernelSqExpBase.sq_exp_calc_KernBase_hess_x_jit(Rtensor, theta, hp_kernel, KernBase)
        
        return KernelSqExpGradMod.sq_exp_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, 
                                                                  KernBase, KernBase_hess_x, bvec_use_grad2)
        
    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase, 
                                        KernBase_hess_x, bvec_use_grad2 = None):
        
        dim, n1, n2 = Rtensor.shape
        
        if bvec_use_grad2 is None:
            n2_grad     = n2
            Rtensor_g2  = Rtensor
            KernBase_g2 = KernBase
        else:
            n2_grad     = np.sum(bvec_use_grad2)
            Rtensor_g2  = Rtensor[:,:, bvec_use_grad2]
            KernBase_g2 = KernBase[:, bvec_use_grad2]
        
        KernGrad_grad_x          = np.zeros((dim, n1*dim, n2 + n2_grad * dim))
        KernGrad_grad_x[:,:,:n2] = KernBase_hess_x
        
        for k in range(dim):
            for i in range(dim):
                delta_ik = int(i == k)
                ri1 = i * n1
                ri2 = ri1 + n1
                
                for j in range(dim):
                    delta_jk = int(j == k)
                    delta_ij = int(i == j)
                    
                    cj1 = n2 + j * n2_grad
                    cj2 = cj1 + n2 
                    
                    KernGrad_grad_x[k, ri1:ri2, cj1:cj2] \
                        = (-4* theta[i] * theta[j] * (delta_ik * Rtensor_g2[j] + delta_jk * Rtensor_g2[i])
                            - 4 * delta_ij * theta[i] * theta[k] * Rtensor_g2[k]
                            + 8 * theta[i] * theta[j] * theta[k] * Rtensor_g2[i] * Rtensor_g2[j] * Rtensor_g2[k]) * KernBase_g2
                     
        return KernGrad_grad_x  

    @staticmethod
    def sq_exp_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel, 
                                     bvec_use_grad = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''
        
        KernGrad = KernelSqExpGradMod.sq_exp_calc_KernGrad(Rtensor, theta, hp_kernel, 
                                                           bvec_use_grad, bvec_use_grad)
        
        return KernelSqExpGradMod.sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad, 
                                                                   bvec_use_grad)

    @staticmethod
    @jit(nopython=True)
    def sq_exp_calc_KernGrad_grad_th_jit(Rtensor, theta, hp_kernel, KernGrad, 
                                         bvec_use_grad = None):
        '''
        See method sq_exp_calc_KernGrad_grad_th() for documentation
        '''

        [dim, n1, n2] = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes'

        KernBase   = KernGrad[:n1, :n1]
        Rtensor_sq = Rtensor**2
        
        ''' Modify terms for case where not all grads are used '''
        
        if bvec_use_grad is None:
            n_grad        = n1
            Rtensor_g1    = Rtensor
            Rtensor_sq_g1 = Rtensor_sq_gg = Rtensor_sq
            KernBase_g1   = KernBase_gg   = KernBase
        else:
            n_grad        = np.sum(bvec_use_grad)
            
            Rtensor_g1    = Rtensor[:, bvec_use_grad,:]
            
            Rtensor_sq_g1 = Rtensor_sq[:,bvec_use_grad,:]
            Rtensor_sq_gg = Rtensor_sq_g1[:,:,bvec_use_grad]
            
            KernBase_g1   = KernBase[bvec_use_grad, :]
            KernBase_gg   = KernBase_g1[:, bvec_use_grad]
        
        ''' Calculate KernGrad_grad_th '''
        
        n_rows = n1 + n_grad * dim
        KernGrad_grad_th = np.zeros((dim, n_rows, n_rows))
        
        for d in range(dim):
            # Covariance obj with obj
            KernGrad_grad_th[d, :n1, :n1] = -Rtensor_sq[d,:,:] * KernBase
            
            rd1 = n1 + d * n_grad
            rd2 = rd1 + n_grad
            
            # Covariance obj with grad for d == i and d == j
            term = -2 * Rtensor_g1[d] * KernBase_g1
            KernGrad_grad_th[d, rd1:rd2, :n1] += term
            KernGrad_grad_th[d, :n1, rd1:rd2] += term.T
            
            # Covariance grad with grad for case of d == i == j
            KernGrad_grad_th[d, rd1:rd2, rd1:rd2] += 2 * KernBase_gg
            
            for i in range(dim):
                ri1 = n1 + i * n_grad
                ri2 = ri1 + n_grad 
                
                # Covariance obj with grad
                term = -Rtensor_sq_g1[d,:,:] * KernGrad[ri1:ri2, :n1] 
                KernGrad_grad_th[d, ri1:ri2, :n1] += term
                KernGrad_grad_th[d, :n1, ri1:ri2] += term.T
                
                # Covariance grad with grad when i == d and j == d
                term = -4 * theta[i] * Rtensor[d,:,:] * Rtensor[i,:,:] * KernBase
                KernGrad_grad_th[d, rd1:rd2, ri1:ri2] += term
                KernGrad_grad_th[d, ri1:ri2, rd1:rd2] += term
                
                # Covariance grad with grad when i == j
                KernGrad_grad_th[d, ri1:ri2, ri1:ri2] -= Rtensor_sq_gg[d,:,:] * KernGrad[ri1:ri2, ri1:ri2]
                    
                # Covariance grad with grad when i != j
                for j in range(i+1,dim):
                    cj1 = n1 + j * n_grad
                    cj2 = cj1 + n_grad
                    
                    term = -Rtensor_sq_gg[d,:,:] * KernGrad[ri1:ri2, cj1:cj2]
                    KernGrad_grad_th[d, ri1:ri2, cj1:cj2] += term
                    KernGrad_grad_th[d, cj1:cj2, ri1:ri2] += term
                
        return KernGrad_grad_th
    
    @staticmethod
    def sq_exp_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the squared exponential kernel')


class KernelSqExp(KernelSqExpBase, KernelSqExpGradMod):

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
    def sq_exp_Kern_precon(n_eval, n_grad, theta, calc_grad = False, b_return_vec = False):
        
        # Calculate the precondition matrix
        gamma    = KernelSqExp.sq_exp_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n_eval), np.kron(gamma, np.ones(n_grad))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 1 / gamma
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n_eval, n_grad, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon
