#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:38:59 2022

@author: andremarchildon
"""

import numpy as np
from numba import jit

from . import KernelCommon

class KernelMatern5f2Base:

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernBase(Rtensor, theta, hp_kernel, *args):
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
        
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        
        sqrt5    = np.sqrt(5)
        KernBase = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * np.exp(-sqrt5 * nu_mat)
            
        return KernBase

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernBase_hess_x(Rtensor, theta, hp_kernel, *args):
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
    @jit(nopython=True)
    def matern_5f2_calc_KernBase_grad_th(Rtensor, theta, hp_kernel, *args):
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
        # inv_nu_mat = np.divide(0.5, nu_mat, out=np.zeros_like(nu_mat[0,0]), where=nu_mat!=0)

        sqrt5      = np.sqrt(5)
        Rtensor_sq = Rtensor**2
        KernBase   = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * np.exp(-sqrt5 * nu_mat)

        KernBase_grad_th = np.zeros((dim, n1, n1))

        for d in range(dim):
            KernBase_grad_th[d,:,:] = -sqrt5 * Rtensor_sq[d,:,:] * inv_nu_mat * KernBase

        return KernBase_grad_th
    
    @staticmethod
    def matern_5f2_calc_KernBase_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the Matern 5/2 kernel')

class KernelMatern5f2Grad:

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad(Rtensor, theta, hp_kernel, *args):
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

        Abase   = np.exp(-np.sqrt(5) * nu_mat)
        sqrt5   = np.sqrt(5)
        mat1    = (5.0/3.0) * (1 + sqrt5 * nu_mat) * Abase
        
        KernGrad = np.zeros((n1*(dim+ 1), n2*(dim + 1)))
        KernGrad[:n1, :n2] = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * Abase
        
        for i in range(dim):
            r1 = (i + 1) * n1
            r2 = r1 + n1
            
            c1 = (i + 1) * n2
            c2 = c1 + n2
            
            # Covariance obj with grad
            term = mat1 * Rtensor[i,:,:] * theta[i]
            KernGrad[r1:r2, :n2] = -term
            KernGrad[:n1, c1:c2] =  term
            
            # Covariance grad with grad for i == j
            KernGrad[r1:r2,c1:c2] = mat1 * theta[i] - (25./3.) * theta[i]**2 * Rtensor[i,:,:]**2 * Abase
            
            # Covariance grad with grad for i != j
            for j in range(i+1,dim):
                term = -(25./3.) * theta[i] * theta[j] * Rtensor[i,:,:] * Rtensor[j,:,:] * Abase
                KernGrad[r1:r2,               n2*(j+1):n2*(j+2)] += term
                KernGrad[n1*(j+1):n1*(j+2), n2*(i+1):n2*(i+2)] += term
                
        return KernGrad

    @staticmethod
    def matern_5f2_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel, *args):
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
    def matern_5f2_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase_hess_x, *args):
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
        # inv_nu_mat = np.divide(1, nu_mat, out=np.zeros_like(nu_mat), where=nu_mat!=0)

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
    def matern_5f2_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel, *args):
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
        # inv_nu_mat = np.divide(1, nu_mat, out=np.zeros_like(nu_mat), where=nu_mat!=0)
        
        sqrt5  = np.sqrt(5)
        Abase  = np.exp(-sqrt5 * nu_mat)
        mat1   = (5./3.) * (1 + sqrt5 * nu_mat) * Abase

        KernGrad_grad_th = np.zeros((dim, n1*(dim+1), n1*(dim+1)))
        
        for d in range(dim):
            # Covariance obj with obj
            KernGrad_grad_th[d,:n1,:n1] = -0.5 * (Rtensor_sq[d] * mat1)
            # KernGrad_grad_th[d,:n1,:n1] = -(5/6) * Rtensor_sq[d,:,:] * (1 + sqrt5*nu_mat) * Abase
            
            d1 = (d + 1) * n1
            d2 = d1 + n1
            
            # Covariance obj with grad for d == i
            term = mat1 * Rtensor[d,:,:]
            KernGrad_grad_th[d, d1:d2, :n1] -= term
            KernGrad_grad_th[d, :n1, d1:d2] += term
            
            # Covariance grad with grad for case of d == i == j
            KernGrad_grad_th[d, d1:d2, d1:d2] += mat1
            
            for i in range(dim):
                r1 = (i + 1) * n1
                r2 = r1 + n1
                
                # Covariance obj with grad
                term = -(25. / 6.) * theta[i] * Rtensor[i] * Rtensor_sq[d,:,:] * Abase
                KernGrad_grad_th[d, r1:r2, :n1] -= term
                KernGrad_grad_th[d, :n1, r1:r2] += term
                
                # Covariance grad with grad when i == d and j == d
                term = -((25./3.) * theta[i]) * Rtensor[i] * Rtensor[d] * Abase
                KernGrad_grad_th[d, d1:d2, r1:r2] += term
                KernGrad_grad_th[d, r1:r2, d1:d2] += term
                
                # Covariance grad with grad when i == j
                KernGrad_grad_th[d, r1:r2, r1:r2] -= ((25./6.) * theta[i]) * Rtensor_sq[d,:,:] * Abase
                KernGrad_grad_th[d, r1:r2, r1:r2] += ((25 * sqrt5 / 6.) * theta[i]**2) * Rtensor_sq[i] * Rtensor_sq[d,:,:] * inv_nu_mat * Abase
                    
                for j in range(i+1,dim):
                    c1 = (j + 1) * n1
                    c2 = c1 + n1
                    term = ((25 * sqrt5 / 6.) * theta[i] * theta[j]) * Rtensor[i] * Rtensor[j] * Rtensor_sq[d,:,:] * inv_nu_mat * Abase
                    KernGrad_grad_th[d, r1:r2, c1:c2] += term
                    KernGrad_grad_th[d, c1:c2, r1:r2] += term
                
        return KernGrad_grad_th
        
    @staticmethod
    def matern_5f2_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the Matern 5/2 kernel')
       
class KernelMatern5f2GradMod:

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad(Rtensor, theta, hp_kernel, 
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
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat  = np.sqrt(nu_mat)
        Abase   = np.exp(-np.sqrt(5) * nu_mat)
        sqrt5   = np.sqrt(5)
        mat1    = (5.0/3.0) * (1 + sqrt5 * nu_mat) * Abase
        
        ''' Modify terms for case where not all grads are used '''
        
        if (bvec_use_grad1 is None) and (bvec_use_grad2 is None):
            n1_grad = n1
            n2_grad = n2
            
            Rtensor_g1 = Rtensor_g2 = Rtensor_gg = Rtensor
            mat1_g1    = mat1_g2    = mat1_gg    = mat1
            Abase_gg   = Abase
            
        else:
            if bvec_use_grad1 is None:
                n1_grad     = n1
                Rtensor_g1  = Rtensor
                Abase_g1    = Abase
                mat1_g1     = mat1
            else:
                n1_grad     = np.sum(bvec_use_grad1)
                Rtensor_g1  = Rtensor[:, bvec_use_grad1,:]
                Abase_g1    = Abase[bvec_use_grad1, :]
                mat1_g1     = mat1[bvec_use_grad1, :]
            
            if bvec_use_grad2 is None:
                n2_grad     = n2
                Rtensor_g2  = Rtensor
                mat1_g2     = mat1
                
                Rtensor_gg  = Rtensor_g1
                Abase_gg    = Abase_g1
                mat1_gg     = mat1_g1
            else:
                n2_grad     = np.sum(bvec_use_grad2)
                Rtensor_g2  = Rtensor[:,:, bvec_use_grad2]
                mat1_g2     = mat1[:, bvec_use_grad2]
                
                Rtensor_gg  = Rtensor_g1[:,:, bvec_use_grad2]
                Abase_gg    = Abase_g1[:, bvec_use_grad2]
                mat1_gg     = mat1_g1[:, bvec_use_grad2]
        
        ''' Calculate the Kernel '''
        
        KernGrad = np.zeros((n1 + n1_grad * dim, n2 + n2_grad * dim))
        KernGrad[:n1, :n2] = (1 + sqrt5 * nu_mat + (5.0/3.0) * nu_mat**2) * Abase
        
        for i in range(dim):
            ri1 = n1 + i * n1_grad
            ri2 = ri1 + n1_grad
            
            ci1 = n2 + i * n2_grad
            ci2 = ci1 + n2_grad
            
            # Covariance obj with grad
            KernGrad[ri1:ri2, :n2] = -theta[i] * Rtensor_g1[i] * mat1_g1
            KernGrad[:n1, ci1:ci2] =  theta[i] * Rtensor_g2[i] * mat1_g2
            
            # Covariance grad with grad for i == j
            KernGrad[ri1:ri2,ci1:ci2] = theta[i] * mat1_gg - ((25./3.) * theta[i]**2) * Rtensor_gg[i]**2 * Abase_gg
            
            # Covariance grad with grad for i != j
            for j in range(i+1,dim):
                rj1 = n1 + j * n1_grad
                rj2 = rj1 + n1_grad
                
                cj1 = n2 + j * n2_grad
                cj2 = cj1 + n2_grad
                
                term = (-(25./3.) * theta[i] * theta[j]) * Rtensor_gg[i] * Rtensor_gg[j] * Abase_gg
                KernGrad[ri1:ri2, cj1:cj2] += term
                KernGrad[rj1:rj2, ci1:ci2] += term
                
        return KernGrad

    @staticmethod
    def matern_5f2_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel, bvec_use_grad2 = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_x : 3d numpy array of floats
            Derivative of KernGrad wrt first set of nodal locations X
        '''
        
        KernBase_hess_x = KernelMatern5f2Base.matern_5f2_calc_KernBase_hess_x(Rtensor, theta, hp_kernel)
        
        return KernelMatern5f2.matern_5f2_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, KernBase_hess_x, bvec_use_grad2)

    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad_grad_x_jit(Rtensor, theta, hp_kernel, 
                                            KernBase_hess_x, bvec_use_grad2 = None):
        ''' See matern_5f2_calc_KernGrad_grad_x() for documentation '''
        
        dim, n1, n2 = Rtensor.shape
        
        nu_mat = np.zeros((n1, n2))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
            
        nu_mat = np.sqrt(nu_mat)
        
        if bvec_use_grad2 is None:
            n2_grad     = n2
            Rtensor_g2  = Rtensor
            nu_mat_g2   = nu_mat
        else:
            n2_grad     = np.sum(bvec_use_grad2)
            Rtensor_g2  = Rtensor[:,:, bvec_use_grad2]
            nu_mat_g2   = nu_mat[:, bvec_use_grad2]
        
        Abase_g2 = np.exp(-np.sqrt(5) * nu_mat_g2)
        
        KernGrad_grad_x          = np.zeros((dim, n1*dim, n2 + n2_grad * dim))
        KernGrad_grad_x[:,:,:n2] = KernBase_hess_x

        # Take the inverse of nu_mat without causing error for values of zero
        # Use 1e-16 as min value, does not change final calculations since
        # nu_mat[i,j] is zero iff Rtensor[k,i,j] is zero for all 0<=k<=(dim-1) 
        inv_nu_mat_g2 = 1 / np.maximum(nu_mat_g2, 1e-16)
        # inv_nu_mat_g2 = np.divide(1, nu_mat_g2, out=np.zeros_like(nu_mat_g2), where=nu_mat_g2!=0)
        
        ''' Calculate the Kernel '''

        for k in range(dim):
            Rk_til = Rtensor_g2[k] * theta[k]
            
            for i in range(dim):
                delta_ik = int(i == k)
                Ri_til   = Rtensor_g2[i] * theta[i]
                
                ri1 = i * n1
                ri2 = ri1 + n1
                
                for j in range(dim):
                    delta_jk = int(j == k)
                    delta_ij = int(i == j)
                    Rj_til   = Rtensor_g2[j] * theta[j]
                    
                    cj1 = n2 + j * n2_grad
                    cj2 = cj1 + n2 
                    
                    KernGrad_grad_x[k, ri1:ri2, cj1:cj2] \
                        = -(25/3) * (theta[i]   * delta_ik * Rj_til 
                                     + theta[j] * delta_jk * Ri_til 
                                     + theta[i] * delta_ij * Rk_til 
                                     - np.sqrt(5) * inv_nu_mat_g2 * Ri_til * Rj_til * Rk_til) * Abase_g2
        
        return KernGrad_grad_x  
    
    @staticmethod
    @jit(nopython=True)
    def matern_5f2_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel, bvec_use_grad = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''

        ''' Calculate the base Kernel '''

        dim, n1, n2 = Rtensor.shape
        assert n1 == n2, 'n1 and n2 must match'
        
        Rtensor_sq = Rtensor**2
        nu_mat = np.zeros((n1, n1))
        
        for i in range(dim):
            nu_mat += theta[i] * Rtensor[i,:,:]**2
        
        nu_mat = np.sqrt(nu_mat)
        
        sqrt5  = np.sqrt(5)
        Abase  = np.exp(-sqrt5 * nu_mat)
        mat1   = (5./3.) * (1 + sqrt5 * nu_mat) * Abase

        ''' Modify terms for case where not all grads are used '''
        
        if bvec_use_grad is None:
            n_grad = n1
            
            Rtensor_g1    = Rtensor_gg    = Rtensor
            Rtensor_sq_g1 = Rtensor_sq_gg = Rtensor_sq
            Abase_g1      = Abase_gg      = Abase
            mat1_g1       = mat1 
            nu_mat_gg     = nu_mat
        else:
            n_grad        = np.sum(bvec_use_grad)
            
            Rtensor_g1    = Rtensor[:, bvec_use_grad,:]
            Rtensor_gg    = Rtensor_g1[:,:, bvec_use_grad]
            
            Rtensor_sq_g1 = Rtensor_sq[:,bvec_use_grad,:]
            Rtensor_sq_gg = Rtensor_sq_g1[:,:,bvec_use_grad]
            
            Abase_g1      = Abase[bvec_use_grad, :]
            Abase_gg      = Abase_g1[:, bvec_use_grad]
            
            mat1_g1       = mat1[bvec_use_grad, :]
            
            nu_mat_g1     = nu_mat[bvec_use_grad,:]
            nu_mat_gg     = nu_mat_g1[:,bvec_use_grad]
        
        # Take the inverse of nu_mat without causing error for values of zero
        # Use 1e-16 as min value, does not change final calculations since
        # nu_mat[i,j] is zero iff Rtensor[k,i,j] is zero for all 0<=k<=(dim-1) 
        inv_nu_mat_gg = 1 / np.maximum(nu_mat_gg, 1e-16)
        # inv_nu_mat_gg = np.divide(1, nu_mat_gg, out=np.zeros_like(nu_mat_gg), where=nu_mat_gg!=0)
        
        ''' Calculate the Kernel '''

        n_rows = n1 + n_grad * dim
        KernGrad_grad_th = np.zeros((dim, n_rows, n_rows))
        
        for d in range(dim):
            # Covariance obj with obj
            KernGrad_grad_th[d,:n1,:n1] = -0.5 * (Rtensor_sq[d] * mat1)
            
            rd1 = n1 + d * n_grad
            rd2 = rd1 + n_grad
            
            # Covariance obj with grad for d == i or d == j
            term = -Rtensor_g1[d] * mat1_g1
            KernGrad_grad_th[d, rd1:rd2, :n1] += term
            KernGrad_grad_th[d, :n1, rd1:rd2] += term.T
            
            # Covariance grad with grad for case of d == i == j
            KernGrad_grad_th[d, rd1:rd2, rd1:rd2] += mat1
            
            for i in range(dim):
                ri1 = n1 + i * n_grad
                ri2 = ri1 + n_grad 
                
                # Covariance obj with grad
                term = (25. / 6.) * theta[i] * Rtensor_g1[i] * Rtensor_sq_g1[d] * Abase_g1
                KernGrad_grad_th[d, ri1:ri2, :n1] += term
                KernGrad_grad_th[d, :n1, ri1:ri2] += term.T
                
                # Covariance grad with grad when i == d and j == d
                term = -((25./3.) * theta[i]) * Rtensor_g1[i] * Rtensor_g1[d] * Abase_g1
                KernGrad_grad_th[d, rd1:rd2, ri1:ri2] += term
                KernGrad_grad_th[d, ri1:ri2, rd1:rd2] += term.T
                
                # Covariance grad with grad when i == j
                KernGrad_grad_th[d, ri1:ri2, ri1:ri2] += (-((25./6.) * theta[i]) * Rtensor_sq_gg[d] * Abase_gg
                                                          +((25 * sqrt5 / 6.) * theta[i]**2) * Rtensor_sq_gg[i] * Rtensor_sq_gg[d] * inv_nu_mat_gg * Abase_gg)
                    
                for j in range(i+1,dim):
                    cj1 = n1 + j * n_grad
                    cj2 = cj1 + n_grad
                    
                    term = ((25 * sqrt5 / 6.) * theta[i] * theta[j]) * Rtensor_gg[i] * Rtensor_gg[j] * Rtensor_sq_gg[d] * inv_nu_mat_gg * Abase_gg
                    KernGrad_grad_th[d, ri1:ri2, cj1:cj2] += term
                    KernGrad_grad_th[d, cj1:cj2, ri1:ri2] += term
                
        return KernGrad_grad_th
        
    @staticmethod
    def matern_5f2_calc_KernGrad_grad_alpha(*args):
        raise Exception('There are no kernel hyperparameters for the Matern 5/2 kernel')    
       
# class KernelMatern5f2(KernelMatern5f2Base, KernelMatern5f2Grad):
class KernelMatern5f2(KernelMatern5f2Base, KernelMatern5f2GradMod):

    matern_5f2_hp_kernel_default = None
    matern_5f2_hp_kernel_range   = [np.nan, np.nan]
    
    @staticmethod
    def matern_5f2_theta2gamma(theta):
        # Convert hyperparameter theta to gamma
        return np.sqrt((5.0/3.0) * theta)
    
    @staticmethod
    def matern_5f2_gamma2theta(gamma):
        # Convert hyperparameter gamma to theta
        return (3.0/5.0) * gamma**2

    @staticmethod
    def matern_5f2_Kern_precon(n_eval, n_grad, theta, calc_grad = False, b_return_vec = False):
        # Calculate the precondition matrix
        
        gamma    = KernelMatern5f2.matern_5f2_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n_eval), np.kron(gamma, np.ones(n_grad))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 5.0 / (6.0 * gamma)
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n_eval, n_grad, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon

    @staticmethod
    def matern_5f2_calc_nu_mat(theta, Rtensor_sq):
        return np.sqrt(np.tensordot(theta, Rtensor_sq, axes=1))
    