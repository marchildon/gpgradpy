#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:52:46 2021

@author: andremarchildon
"""

import numpy as np
from numba import jit

from . import KernelCommon

class KernelRatQuadBase:
    
    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernBase(Rtensor, theta, hp_kernel):
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
        hp_kernel : float
            Hyperparameter for the kernel.

        Returns
        -------
        KernBase : 2d numpy array of floats
            Gradient-free kernel matrix.
        '''
        
        dim, n1, n2 = Rtensor.shape
        alpha = hp_kernel

        Rtensor_sq = Rtensor **2

        temp = np.zeros((n1, n2))

        for d in range(dim):
            temp += theta[d] * Rtensor_sq[d, :, :]
            
        KernBase = (1 + temp / alpha)**(-alpha)

        return KernBase

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernBase_hess_x(Rtensor, theta, hp_kernel):
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
        
        ''' Preliminary calculations '''

        alpha = hp_kernel
        
        Rtensor_sq = Rtensor **2
        
        B = np.zeros((n1, n2))
        
        for d in range(dim):
            B += theta[d] * Rtensor_sq[d, :, :]
        
        B = 1 + B / alpha
        
        B_m_alpha_m1 = B**(-alpha - 1)
        B_m_alpha_m2 = B**(-alpha - 2)
    
        ''' Calculate the Kernel '''
        
        KernBase_hess_x = np.zeros((dim, n1*dim, n2))
        
        scalar1 = 4*(1 + 1 / hp_kernel)
        
        for k in range(dim):
            KernBase_hess_x[k, k*n1:(k+1)*n1, :] -= 2 * theta[k] * B_m_alpha_m1
            
            for i in range(dim):
                KernBase_hess_x[k, i*n1:(i+1)* n1, :] += (scalar1 * theta[i] * theta[k]) * Rtensor[i,:,:] * Rtensor[k,:,:] * B_m_alpha_m2
                
        return KernBase_hess_x  
    
    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernBase_grad_th(Rtensor, theta, hp_kernel):
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
        assert n1 == n2

        alpha       = hp_kernel
        base        = np.zeros((n1, n1))
        Rtensor_sq  = Rtensor**2

        for d in range(dim):
            base += theta[d] * Rtensor_sq[d, :, :]
            
        B_m_alpha_m1 = (1 + base / alpha)**(-alpha - 1)

        KernBase_grad_th = np.zeros((dim, n1, n1))
        
        for d in range(dim):
            KernBase_grad_th[d,:,:] = -Rtensor_sq[d, :,:] * B_m_alpha_m1

        return KernBase_grad_th

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernBase_grad_alpha(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernBase_grad_th : 3d numpy array of floats
            Derivative of KernBase wrt alpha (ie hp_kernel).
        '''

        ''' Check input parameters '''

        alpha       = hp_kernel
        dim, n1, n2 = Rtensor.shape
        Rtensor_sq  = Rtensor**2

        ''' Calculate the gradient of the Kernel '''

        mat_1 = np.zeros((n1, n1))
        
        for d in range(dim):
            mat_1 += theta[d] * Rtensor_sq[d, :, :]
        
        B = 1 + mat_1 / alpha
        
        KernBase_grad_alpha        = np.zeros((1, n1, n1))
        KernBase_grad_alpha[0,:,:] = B**(-alpha-1) * (mat_1 / alpha - B * np.log(B))
        
        return KernBase_grad_alpha
    
class KernelRatQuadGrad:
    
    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad : 2d numpy array of floats
            Gradient-enhanced kernel matrix.
        '''

        ''' Preliminary calculations '''

        dim, n1, n2 = Rtensor.shape
        alpha = hp_kernel
        
        Rtensor_sq = Rtensor **2
        
        temp = np.zeros((n1, n2))
        
        for d in range(dim):
            temp += theta[d] * Rtensor_sq[d, :, :]
        
        mat_1 = 1.0 + temp / alpha
        
        B_m_alpha    = mat_1**(-alpha)
        B_m_alpha_m1 = mat_1**(-alpha - 1)
        B_m_alpha_m2 = mat_1**(-alpha - 2)
        
        ''' Calculate the Kernel '''
        
        KernGrad            = np.zeros((n1*(dim+1), n2*(dim+1)))
        KernGrad[:n1, :n2]  = B_m_alpha
        
        const = 4*( 1 + 1 / alpha)
        
        for i in range(dim):
            r1 = (i + 1) * n1
            r2 = r1 + n1
            
            c1 = (i + 1) * n2
            c2 = c1 + n2
            
            # Covariance obj with grad
            term = 2*theta[i] * Rtensor[i] * B_m_alpha_m1
            KernGrad[r1:r2, :n2] = -term
            KernGrad[:n1, c1:c2] =  term
            
            # Covariance grad with grad when i == j
            KernGrad[r1:r2,c1:c2] = (2 * theta[i]) * B_m_alpha_m1 - (const * theta[i]**2) * Rtensor_sq[i] * B_m_alpha_m2
            
            for j in range(i+1, dim):
                term = (-const * theta[i] * theta[j]) * Rtensor[i,:,:] * Rtensor[j,:,:] * B_m_alpha_m2
                KernGrad[r1:r2,               n2*(j+1):n2*(j+2)] += term
                KernGrad[n1*(j+1):n1*(j+2), n2*(i+1):n2*(i+2)] += term

        return KernGrad

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel):
        '''
        See sq_exp_calc_KernBase_hess_x() for documentation
        '''
        
        dim, n1, n2 = Rtensor.shape
    
        ''' Preliminary calculations '''

        alpha = hp_kernel
        
        Rtensor_sq = Rtensor **2
        
        B = np.zeros((n1, n2))
        
        for d in range(dim):
            B += theta[d] * Rtensor_sq[d, :, :]
        
        B            = 1 + B / alpha
        B_m_alpha_m1 = B**(-alpha - 1)
        B_m_alpha_m2 = B**(-alpha - 2)
        B_m_alpha_m3 = B**(-alpha - 3)
    
        ''' Calculate the Kernel '''
        
        KernGrad_grad_x  = np.zeros((dim, n1*dim, n2*(1+dim)))
    
        scalar1 = 4*(1 + 1 / hp_kernel)
        scalar2 = 2*(1 + 2 / hp_kernel) * scalar1

        for k in range(dim):
            KernGrad_grad_x[k, k*n1:(k+1)*n1, :n2] -= 2 * theta[k] * B_m_alpha_m1
            
            for i in range(dim):
                r1 = i  * n1
                r2 = r1 + n1
                
                # Obj with grad
                KernGrad_grad_x[k, r1:r2, :n2] += (scalar1 * theta[i] * theta[k]) * Rtensor[i,:,:] * Rtensor[k,:,:] * B_m_alpha_m2
                
                # i == j
                KernGrad_grad_x[k, r1:r2, (i+1)*n2:(i+2)*n2] -= (scalar1 * theta[i] * theta[k]) * Rtensor[k,:,:] * B_m_alpha_m2
                
                # i == k and j == k
                term = -(scalar1 * theta[i] * theta[k]) * Rtensor[i,:,:] * B_m_alpha_m2
                KernGrad_grad_x[k, r1:r2, (k+1)*n2:(k+2)*n2]         += term
                KernGrad_grad_x[k, k*n1:(k+1)*n1, (i+1)*n2:(i+2)*n2] += term
                
                for j in range(dim):
                    
                    KernGrad_grad_x[k, r1:r2, (j+1)*n2:(j+2)*n2] \
                        += (scalar2 * theta[i] * theta[j] * theta[k]) * Rtensor[i,:,:] * Rtensor[j,:,:] * Rtensor[k,:,:] * B_m_alpha_m3
                    
        return KernGrad_grad_x  

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''

        ''' Check inputs '''

        [dim, n1, n2] = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes'

        ''' Calculate the derivative wrt the length hyperparameter '''

        alpha      = hp_kernel
        Rtensor_sq = Rtensor**2
        
        # Calculate base term
        base        = np.zeros((n1, n1))
        Rtensor_sq = Rtensor**2

        for d in range(dim):
            base += theta[d] * Rtensor_sq[d, :, :]
            
        B_m_alpha_m1 = (1 + base / alpha)**(-alpha - 1)
        B_m_alpha_m2 = (1 + base / alpha)**(-alpha - 2)
        B_m_alpha_m3 = (1 + base / alpha)**(-alpha - 3)
        
        KernGrad_grad_th = np.zeros((dim, n1*(dim+1), n1*(dim+1)))
        
        scalar1 = 1 + 1/alpha
        scalar2 = scalar1 * (1 + 2/alpha)
        
        for d in range(dim):
            
            # Covariance obj with obj
            KernGrad_grad_th[d, :n1, :n1] = -Rtensor_sq[d, :,:] * B_m_alpha_m1
            
            d1 = (d + 1) * n1
            d2 = d1 + n1
            
            # Covariance obj with grad for d == i
            term = -2 * Rtensor[d,:,:] * B_m_alpha_m1
            KernGrad_grad_th[d, d1:d2, :n1] += term
            KernGrad_grad_th[d, :n1, d1:d2] += term.T
            
            # Covariance grad with grad for d == i == j
            KernGrad_grad_th[d, d1:d2, d1:d2] += 2 * B_m_alpha_m1
            
            for i in range(dim):
                r1 = (i + 1) * n1
                r2 = r1 + n1 
                
                # Covariance obj with grad
                term = 2*scalar1 * theta[i] * Rtensor[i,:,:] * Rtensor_sq[d, :,:] * B_m_alpha_m2
                KernGrad_grad_th[d, r1:r2, :n1] += term
                KernGrad_grad_th[d, :n1, r1:r2] += term.T
                
                # Covariance grad with grad for i == d and j == d
                term = (-4*scalar1 * theta[i]) * Rtensor[i,:,:] * Rtensor[d,:,:] * B_m_alpha_m2
                KernGrad_grad_th[d, d1:d2, r1:r2] += term
                KernGrad_grad_th[d, r1:r2, d1:d2] += term
                
                # Covariance grad with grad for i == j
                KernGrad_grad_th[d, r1:r2, r1:r2] += ((-2*scalar1 * theta[i]) * Rtensor_sq[d, :,:] * B_m_alpha_m2
                                                      + (4 * scalar2 * theta[i]**2) * Rtensor_sq[i,:,:] * Rtensor_sq[d, :,:] * B_m_alpha_m3)
                
                # Covariance grad with grad for i != j
                for j in range(i+1,dim):
                    c1 = (j + 1) * n1 
                    c2 = c1 + n1
                    term = (4 * scalar2 * theta[i] * theta[j]) * Rtensor[i] * Rtensor[j] * Rtensor_sq[d] * B_m_alpha_m3
                    KernGrad_grad_th[d, r1:r2, c1:c2] += term
                    KernGrad_grad_th[d, c1:c2, r1:r2] += term
                
        return KernGrad_grad_th

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_alpha(Rtensor, theta, hp_kernel):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt alpha (ie hp_kernel).
        '''

        ''' Check inputs '''

        alpha            = hp_kernel
        dim, n1, n2      = Rtensor.shape
        Rtensor_sq = Rtensor**2

        assert n1 == n2, 'Incompatible shapes'
        
        ''' Calculate the gradient of the Kernel '''
        
        mat_1 = np.zeros((n1, n1))
        
        for d in range(dim):
            mat_1 += theta[d] * Rtensor_sq[d, :, :]
        
        B = 1 + mat_1 / alpha
        B_m_alpha_m2 = B**(-alpha - 2)
        
        B_lnB = B * np.log(B) 
        
        T0 = 2* B_m_alpha_m2 * (((-alpha-1)/alpha**2) * mat_1 + B_lnB)
        T1 = 2 * B_m_alpha_m2 *( ((alpha + 1)/alpha**2) * mat_1 - B_lnB)
        T2 = 4 * B**(-alpha -3) *(B / alpha**2 - ((1 + 1/alpha)*(alpha+2)/alpha**2) * mat_1 + (1 + 1/alpha) * B_lnB)
        
        KernGrad_grad_alpha = np.zeros((1, n1*(dim+1), n1*(dim+1)))
        
        # Covariance obj with obj
        KernGrad_grad_alpha[0,:n1,:n1] = B**(-alpha-1) * (mat_1 / alpha - B * np.log(B))
        
        for i in range(dim):
            r1 = (i+1)*n1 
            r2 = r1 + n1
            
            # Covariance obj with grad
            term = theta[i] * Rtensor[i,:,:] * T0
            KernGrad_grad_alpha[0, r1:r2, :n1] += term
            KernGrad_grad_alpha[0, :n1, r1:r2] -= term
            
            # Covariance grad with grad for i == j
            KernGrad_grad_alpha[0, r1:r2,r1:r2] += theta[i] * T1 + theta[i]**2 * Rtensor_sq[i,:,:] * T2
            
            # Covariance grad with grad for i != j
            for j in range(i+1,dim):
                c1 = (j + 1) * n1 
                c2 = c1 + n1
                term = (theta[i] * theta[j]) * Rtensor[i,:,:] * Rtensor[j,:,:] * T2
                KernGrad_grad_alpha[0, r1:r2, c1:c2] += term
                KernGrad_grad_alpha[0, c1:c2, r1:r2] += term
                
        return KernGrad_grad_alpha
    

class KernelRatQuadGradMod:
    
    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad(Rtensor, theta, hp_kernel, 
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

        ''' Preliminary calculations '''

        dim, n1, n2 = Rtensor.shape
        alpha = hp_kernel
        
        Rtensor_sq = Rtensor **2
        
        temp = np.zeros((n1, n2))
        
        for d in range(dim):
            temp += theta[d] * Rtensor_sq[d, :, :]
        
        mat_1 = 1.0 + temp / alpha
        
        B_m_alpha    = mat_1**(-alpha)
        B_m_alpha_m1 = mat_1**(-alpha - 1)
        B_m_alpha_m2 = mat_1**(-alpha - 2)
        
        ''' Modify terms for case where not all grads are used '''
        
        if (bvec_use_grad1 is None) and (bvec_use_grad2 is None):
            n1_grad = n1
            n2_grad = n2
            
            Rtensor_g1      = Rtensor_g2      = Rtensor_gg      = Rtensor
            B_m_alpha_m1_g1 = B_m_alpha_m1_g2 = B_m_alpha_m1_gg = B_m_alpha_m1
            Rtensor_sq_gg   = Rtensor_sq
            B_m_alpha_m2_gg = B_m_alpha_m2
        else:
            if bvec_use_grad1 is None:
                n1_grad         = n1
                Rtensor_g1      = Rtensor
                Rtensor_sq_g1   = Rtensor_sq
                B_m_alpha_m1_g1 = B_m_alpha_m1
                B_m_alpha_m2_g1 = B_m_alpha_m2
            else:
                n1_grad         = np.sum(bvec_use_grad1)
                Rtensor_g1      = Rtensor[:, bvec_use_grad1,:]
                Rtensor_sq_g1   = Rtensor_sq[:, bvec_use_grad1,:]
                B_m_alpha_m1_g1 = B_m_alpha_m1[bvec_use_grad1,:]
                B_m_alpha_m2_g1 = B_m_alpha_m2[bvec_use_grad1,:]
            
            if bvec_use_grad2 is None:
                n2_grad         = n2
                
                Rtensor_g2      = Rtensor
                Rtensor_gg      = Rtensor_g1
                
                Rtensor_sq_gg   = Rtensor_sq_g1
                
                B_m_alpha_m1_g2 = B_m_alpha_m1_g1
                B_m_alpha_m1_gg = B_m_alpha_m1_g1
                
                B_m_alpha_m2_gg = B_m_alpha_m2_g1
            else:
                n2_grad         = np.sum(bvec_use_grad2)
                
                Rtensor_g2      = Rtensor[:,:, bvec_use_grad2]
                Rtensor_gg      = Rtensor_g1[:,:, bvec_use_grad2]
                
                Rtensor_sq_gg   = Rtensor_sq_g1[:, :, bvec_use_grad2]
                
                B_m_alpha_m1_g2 = B_m_alpha_m1[:, bvec_use_grad2]
                B_m_alpha_m1_gg = B_m_alpha_m1_g1[:, bvec_use_grad2]
                
                B_m_alpha_m2_gg = B_m_alpha_m2_g1[:, bvec_use_grad2]
        
        ''' Calculate the Kernel '''
        
        KernGrad            = np.zeros((n1 + n1_grad * dim, n2 + n2_grad * dim))
        KernGrad[:n1, :n2]  = B_m_alpha
        
        const = 4*( 1 + 1 / alpha)
        
        for i in range(dim):
            ri1 = n1 + i * n1_grad
            ri2 = ri1 + n1_grad
            
            ci1 = n2 + i * n2_grad
            ci2 = ci1 + n2_grad
            
            # Covariance obj with grad
            KernGrad[ri1:ri2, :n2] = -2*theta[i] * Rtensor_g1[i] * B_m_alpha_m1_g1
            KernGrad[:n1, ci1:ci2] =  2*theta[i] * Rtensor_g2[i] * B_m_alpha_m1_g2
            
            # Covariance grad with grad when i == j
            KernGrad[ri1:ri2,ci1:ci2] = (2 * theta[i]) * B_m_alpha_m1_gg - (const * theta[i]**2) * Rtensor_sq_gg[i] * B_m_alpha_m2_gg
            
            # Covariance grad with grad for i != j
            for j in range(i+1, dim):
                rj1 = n1 + j * n1_grad
                rj2 = rj1 + n1_grad
                
                cj1 = n2 + j * n2_grad
                cj2 = cj1 + n2_grad
                
                term = (-const * theta[i] * theta[j]) * Rtensor_gg[i] * Rtensor_gg[j] * B_m_alpha_m2_gg
                KernGrad[ri1:ri2, cj1:cj2] += term
                KernGrad[rj1:rj2, ci1:ci2] += term

        return KernGrad

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_x(Rtensor, theta, hp_kernel, bvec_use_grad2 = None):
        '''
        See sq_exp_calc_KernBase_hess_x() for documentation
        '''
        
        dim, n1, n2 = Rtensor.shape
    
        ''' Preliminary calculations '''

        alpha = hp_kernel
        
        Rtensor_sq = Rtensor **2
        
        B = np.zeros((n1, n2))
        
        for d in range(dim):
            B += theta[d] * Rtensor_sq[d, :, :]
        
        B            = 1 + B / alpha
        B_m_alpha_m1 = B**(-alpha - 1)
        B_m_alpha_m2 = B**(-alpha - 2)
        B_m_alpha_m3 = B**(-alpha - 3)
        
        ''' Modify terms for case where not all grads are used '''
        
        if bvec_use_grad2 is None:
            n2_grad         = n2
            Rtensor_g2      = Rtensor
            B_m_alpha_m2_g2 = B_m_alpha_m2
            B_m_alpha_m3_g2 = B_m_alpha_m3
        else:
            n2_grad         = np.sum(bvec_use_grad2)
            Rtensor_g2      = Rtensor[:,:, bvec_use_grad2]
            B_m_alpha_m2_g2 = B_m_alpha_m2[:, bvec_use_grad2]
            B_m_alpha_m3_g2 = B_m_alpha_m3[:, bvec_use_grad2]
    
        ''' Calculate the Kernel '''
        
        KernGrad_grad_x  = np.zeros((dim, n1*dim, n2 + n2_grad * dim))
    
        scalar1 = 4*(1 + 1 / hp_kernel)
        scalar2 = 2*(1 + 2 / hp_kernel) * scalar1

        for k in range(dim):
            rk1 = k * n1 
            rk2 = rk1 + n1
            
            ck1 = n2 + k * n2_grad
            ck2 = ck1 + n2 
            
            KernGrad_grad_x[k, rk1:rk2, :n2] -= 2 * theta[k] * B_m_alpha_m1
            
            for i in range(dim):
                ri1 = i * n1
                ri2 = ri1 + n1
                
                ci1 = n2 + i * n2_grad
                ci2 = ci1 + n2 
                
                # Covariance obj with grad
                KernGrad_grad_x[k, ri1:ri2, :n2] += (scalar1 * theta[i] * theta[k]) * Rtensor[i,:,:] * Rtensor[k,:,:] * B_m_alpha_m2
                
                # i == j
                KernGrad_grad_x[k, ri1:ri2, ci1:ci2] -= (scalar1 * theta[i] * theta[k]) * Rtensor_g2[k,:,:] * B_m_alpha_m2_g2
                
                # i == k and j == k
                KernGrad_grad_x[k, ri1:ri2, ck1:ck2] -= (scalar1 * theta[i] * theta[k]) * Rtensor_g2[i] * B_m_alpha_m2_g2
                KernGrad_grad_x[k, rk1:rk2, ci1:ci2] -= (scalar1 * theta[i] * theta[k]) * Rtensor_g2[i] * B_m_alpha_m2_g2
                
                for j in range(dim):
                    cj1 = n2 + j * n2_grad
                    cj2 = cj1 + n2 
                    
                    KernGrad_grad_x[k, ri1:ri2, cj1:cj2] \
                        += (scalar2 * theta[i] * theta[j] * theta[k]) * Rtensor_g2[i] * Rtensor_g2[j] * Rtensor_g2[k] * B_m_alpha_m3_g2
                    
        return KernGrad_grad_x  

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_th(Rtensor, theta, hp_kernel, bvec_use_grad = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt theta
        '''

        ''' Check inputs '''

        [dim, n1, n2] = Rtensor.shape
        assert n1 == n2, 'Incompatible shapes'

        ''' Calculate the derivative wrt the length hyperparameter '''

        alpha      = hp_kernel
        Rtensor_sq = Rtensor**2
        
        # Calculate base term
        base       = np.zeros((n1, n1))
        Rtensor_sq = Rtensor**2

        for d in range(dim):
            base += theta[d] * Rtensor_sq[d, :, :]
            
        B_m_alpha_m1 = (1 + base / alpha)**(-alpha - 1)
        B_m_alpha_m2 = (1 + base / alpha)**(-alpha - 2)
        B_m_alpha_m3 = (1 + base / alpha)**(-alpha - 3)
        
        ''' Modify terms for case where not all grads are used '''
        
        if bvec_use_grad is None:
            n_grad          = n1
            Rtensor_g1      = Rtensor_gg    = Rtensor
            Rtensor_sq_g1   = Rtensor_sq_gg = Rtensor_sq
            
            B_m_alpha_m1_g1 = B_m_alpha_m1_gg = B_m_alpha_m1
            B_m_alpha_m2_g1 = B_m_alpha_m2_gg = B_m_alpha_m2
            B_m_alpha_m3_g1 = B_m_alpha_m3_gg = B_m_alpha_m3
        else:
            n_grad          = np.sum(bvec_use_grad)
            
            Rtensor_g1      = Rtensor[:, bvec_use_grad,:]
            Rtensor_gg      = Rtensor_g1[:, :, bvec_use_grad]
            
            Rtensor_sq_g1   = Rtensor_sq[:,bvec_use_grad,:]
            Rtensor_sq_gg   = Rtensor_sq_g1[:,:,bvec_use_grad]
            
            B_m_alpha_m1_g1 = B_m_alpha_m1[bvec_use_grad, :]
            B_m_alpha_m1_gg = B_m_alpha_m1_g1[:,bvec_use_grad]
            
            B_m_alpha_m2_g1 = B_m_alpha_m2[bvec_use_grad, :]
            B_m_alpha_m2_gg = B_m_alpha_m2_g1[:, bvec_use_grad]
            
            B_m_alpha_m3_g1 = B_m_alpha_m3[bvec_use_grad, :]
            B_m_alpha_m3_gg = B_m_alpha_m3_g1[:,bvec_use_grad]
            
        ''' Calculate KernGrad_grad_th '''
        
        n_rows = n1 + n_grad * dim
        KernGrad_grad_th = np.zeros((dim, n_rows, n_rows))
        
        scalar1 = 1 + 1/alpha
        scalar2 = scalar1 * (1 + 2/alpha)
        
        for d in range(dim):
            
            rd1 = n1 + d * n_grad
            rd2 = rd1 + n_grad
            
            # Covariance obj with obj
            KernGrad_grad_th[d, :n1, :n1] = -Rtensor_sq[d, :,:] * B_m_alpha_m1
            
            # Covariance obj with grad for d == i
            term = -2 * Rtensor_g1[d,:,:] * B_m_alpha_m1_g1
            KernGrad_grad_th[d, rd1:rd2, :n1] += term
            KernGrad_grad_th[d, :n1, rd1:rd2] += term.T
            
            # Covariance grad with grad for d == i == j
            KernGrad_grad_th[d, rd1:rd2, rd1:rd2] += 2 * B_m_alpha_m1_gg
            
            for i in range(dim):
                ri1 = n1 + i * n_grad
                ri2 = ri1 + n_grad 
                
                # Covariance obj with grad
                term = 2*scalar1 * theta[i] * Rtensor_g1[i] * Rtensor_sq_g1[d] * B_m_alpha_m2_g1
                KernGrad_grad_th[d, ri1:ri2, :n1] += term
                KernGrad_grad_th[d, :n1, ri1:ri2] += term.T
                
                # Covariance grad with grad for i == d and j == d
                term = (-4*scalar1 * theta[i]) * Rtensor_gg[i] * Rtensor_gg[d] * B_m_alpha_m2_gg
                KernGrad_grad_th[d, rd1:rd2, ri1:ri2] += term
                KernGrad_grad_th[d, ri1:ri2, rd1:rd2] += term
                
                # Covariance grad with grad for i == j
                KernGrad_grad_th[d, ri1:ri2, ri1:ri2] \
                    += ((-2*scalar1 * theta[i]) * Rtensor_sq_gg[d] * B_m_alpha_m2_gg
                        + (4 * scalar2 * theta[i]**2) * Rtensor_sq_gg[i] * Rtensor_sq_gg[d] * B_m_alpha_m3_gg)
                
                # Covariance grad with grad for i != j
                for j in range(i+1,dim):
                    cj1 = n1 + j * n_grad
                    cj2 = cj1 + n_grad
                    term = (4 * scalar2 * theta[i] * theta[j]) * Rtensor_gg[i] * Rtensor_gg[j] * Rtensor_sq_gg[d] * B_m_alpha_m3_gg
                    KernGrad_grad_th[d, ri1:ri2, cj1:cj2] += term
                    KernGrad_grad_th[d, cj1:cj2, ri1:ri2] += term
                
        return KernGrad_grad_th

    @staticmethod
    @jit(nopython=True)
    def rat_quad_calc_KernGrad_grad_alpha(Rtensor, theta, hp_kernel, bvec_use_grad = None):
        '''
        Parameters
        ----------
        See method sq_exp_calc_KernBase()

        Returns
        -------
        KernGrad_grad_th : 3d numpy array of floats
            Derivative of KernGrad wrt alpha (ie hp_kernel).
        '''

        ''' Check inputs '''

        alpha       = hp_kernel
        dim, n1, n2 = Rtensor.shape
        Rtensor_sq  = Rtensor**2

        assert n1 == n2, 'Incompatible shapes'
        
        ''' Calculate the gradient of the Kernel '''
        
        mat_1 = np.zeros((n1, n1))
        
        for d in range(dim):
            mat_1 += theta[d] * Rtensor_sq[d, :, :]
        
        B = 1 + mat_1 / alpha
        B_m_alpha_m2 = B**(-alpha - 2)
        
        B_lnB = B * np.log(B) 
        
        T0 = 2 * B_m_alpha_m2 * (((-alpha-1)/alpha**2) * mat_1 + B_lnB)
        T1 = 2 * B_m_alpha_m2 *( ((alpha + 1)/alpha**2) * mat_1 - B_lnB)
        T2 = 4 * B**(-alpha -3) *(B / alpha**2 - ((1 + 1/alpha)*(alpha+2)/alpha**2) * mat_1 + (1 + 1/alpha) * B_lnB)
        
        ''' Modify terms for case where not all grads are used '''
        
        if bvec_use_grad is None:
            n_grad          = n1
            Rtensor_g1      = Rtensor_gg    = Rtensor
            Rtensor_sq_g1   = Rtensor_sq_gg = Rtensor_sq
            
            T0_g1 = T0
            T1_gg = T1
            T2_gg = T2
        else:
            n_grad          = np.sum(bvec_use_grad)
            
            Rtensor_g1      = Rtensor[:, bvec_use_grad,:]
            Rtensor_gg      = Rtensor_g1[:, :, bvec_use_grad]
            
            Rtensor_sq_g1   = Rtensor_sq[:,bvec_use_grad,:]
            Rtensor_sq_gg   = Rtensor_sq_g1[:,:,bvec_use_grad]
            
            T0_g1 = T0[bvec_use_grad,:]
            
            T1_g1 = T1[bvec_use_grad, :]
            T1_gg = T1_g1[:,bvec_use_grad]
            
            T2_g1 = T2[bvec_use_grad, :]
            T2_gg = T2_g1[:,bvec_use_grad]
            
        ''' Calculate KernGrad_grad_th '''
        
        n_rows = n1 + n_grad * dim
        KernGrad_grad_alpha = np.zeros((1, n_rows, n_rows))
        
        # Covariance obj with obj
        KernGrad_grad_alpha[0,:n1,:n1] = B**(-alpha-1) * (mat_1 / alpha - B * np.log(B))
        
        for i in range(dim):
            ri1 = n1 + i * n_grad
            ri2 = ri1 + n_grad 
            
            # Covariance obj with grad
            term = theta[i] * Rtensor_g1[i] * T0_g1
            KernGrad_grad_alpha[0, ri1:ri2, :n1] += term
            KernGrad_grad_alpha[0, :n1, ri1:ri2] += term.T
            
            # Covariance grad with grad for i == j
            KernGrad_grad_alpha[0, ri1:ri2,ri1:ri2] += theta[i] * T1_gg + theta[i]**2 * Rtensor_sq_gg[i] * T2_gg
            
            # Covariance grad with grad for i != j
            for j in range(i+1,dim):
                cj1 = n1 + j * n_grad
                cj2 = cj1 + n_grad
                term = (theta[i] * theta[j]) * Rtensor_gg[i] * Rtensor_gg[j] * T2_gg
                KernGrad_grad_alpha[0, ri1:ri2, cj1:cj2] += term
                KernGrad_grad_alpha[0, cj1:cj2, ri1:ri2] += term
                
        return KernGrad_grad_alpha
    
    
# class KernelRatQuad(KernelRatQuadBase, KernelRatQuadGrad):
class KernelRatQuad(KernelRatQuadBase, KernelRatQuadGradMod):
    
    rat_quad_hp_kernel_default = 2
    rat_quad_range_hp_kernel   = [1e-3, 10]

    @staticmethod
    def rat_quad_theta2gamma(theta):
        # Convert hyperparameter theta to gamma
        return np.sqrt(2 * theta) 
    
    @staticmethod
    def rat_quad_gamma2theta(gamma):
        # Convert hyperparameter gamma to theta
        return 0.5 * gamma**2
    
    @staticmethod
    def rat_quad_Kern_precon(n_eval, n_grad, theta, calc_grad = False, b_return_vec = False):
        
        # Calculate the precondition matrix
        gamma    = KernelRatQuad.rat_quad_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n_eval), np.kron(gamma, np.ones(n_grad))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 1 / gamma
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n_eval, n_grad, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon
    