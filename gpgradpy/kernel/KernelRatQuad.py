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
        
        for ir in range(dim):
            r1 = (ir + 1) * n1
            r2 = r1 + n1
            
            c1 = (ir + 1) * n2
            c2 = c1 + n2
            
            # Covariance obj with grad
            term = 2*theta[ir] * Rtensor[ir] * B_m_alpha_m1
            KernGrad[r1:r2, :n2] = -term
            KernGrad[:n1, c1:c2] =  term
            
            # Covariance grad with grad when ir == ic
            KernGrad[r1:r2,c1:c2] = (2 * theta[ir]) * B_m_alpha_m1 - (const * theta[ir]**2) * Rtensor_sq[ir] * B_m_alpha_m2
            
            for ic in range(ir+1, dim):
                term = (-const * theta[ir] * theta[ic]) * Rtensor[ir,:,:] * Rtensor[ic,:,:] * B_m_alpha_m2
                KernGrad[r1:r2,               n2*(ic+1):n2*(ic+2)] += term
                KernGrad[n1*(ic+1):n1*(ic+2), n2*(ir+1):n2*(ir+2)] += term

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
            
            # Covariance obj with grad for d == ir
            term = -2 * Rtensor[d,:,:] * B_m_alpha_m1
            KernGrad_grad_th[d, d1:d2, :n1] += term
            KernGrad_grad_th[d, :n1, d1:d2] += term.T
            
            # Covariance grad with grad for d == ir == ic
            KernGrad_grad_th[d, d1:d2, d1:d2] += 2 * B_m_alpha_m1
            
            for ir in range(dim):
                r1 = (ir + 1) * n1
                r2 = r1 + n1 
                
                # Covariance obj with grad
                term = 2*scalar1 * theta[ir] * Rtensor[ir,:,:] * Rtensor_sq[d, :,:] * B_m_alpha_m2
                KernGrad_grad_th[d, r1:r2, :n1] += term
                KernGrad_grad_th[d, :n1, r1:r2] += term.T
                
                # Covariance grad with grad for ir == d and ic == d
                term = (-4*scalar1 * theta[ir]) * Rtensor[ir,:,:] * Rtensor[d,:,:] * B_m_alpha_m2
                KernGrad_grad_th[d, d1:d2, r1:r2] += term
                KernGrad_grad_th[d, r1:r2, d1:d2] += term
                
                # Covariance grad with grad for ir == ic
                KernGrad_grad_th[d, r1:r2, r1:r2] += ((-2*scalar1 * theta[ir]) * Rtensor_sq[d, :,:] * B_m_alpha_m2
                                                      + (4 * scalar2 * theta[ir]**2) * Rtensor_sq[ir,:,:] * Rtensor_sq[d, :,:] * B_m_alpha_m3)
                
                # Covariance grad with grad for ir != ic
                for ic in range(ir+1,dim):
                    c1 = (ic + 1) * n1 
                    c2 = c1 + n1
                    term = (4 * scalar2 * theta[ir] * theta[ic]) * Rtensor[ir,:,:] * Rtensor[ic,:,:] * Rtensor_sq[d, :,:] * B_m_alpha_m3
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
        
        for ir in range(dim):
            r1 = (ir+1)*n1 
            r2 = r1 + n1
            
            # Covariance obj with grad
            term = theta[ir] * Rtensor[ir,:,:] * T0
            KernGrad_grad_alpha[0, r1:r2, :n1] += term
            KernGrad_grad_alpha[0, :n1, r1:r2] -= term
            
            # Covariance grad with grad for ir == ic
            KernGrad_grad_alpha[0, r1:r2,r1:r2] += theta[ir] * T1 + theta[ir]**2 * Rtensor_sq[ir,:,:] * T2
            
            # Covariance grad with grad for ir != ic
            for ic in range(ir+1,dim):
                c1 = (ic + 1) * n1 
                c2 = c1 + n1
                term = (theta[ir] * theta[ic]) * Rtensor[ir,:,:] * Rtensor[ic,:,:] * T2
                KernGrad_grad_alpha[0, r1:r2, c1:c2] += term
                KernGrad_grad_alpha[0, c1:c2, r1:r2] += term
                
        return KernGrad_grad_alpha
    
class KernelRatQuad(KernelRatQuadBase, KernelRatQuadGrad):
    
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
    def rat_quad_Kern_precon(theta, n1, calc_grad = False, b_return_vec = False):
        
        # Calculate the precondition matrix
        gamma    = KernelRatQuad.rat_quad_theta2gamma(theta)
        pvec     = np.hstack((np.ones(n1), np.kron(gamma, np.ones(n1))))
        pvec_inv = 1 / pvec
        
        gamma_grad_theta = 1 / gamma
        
        grad_precon = KernelCommon.calc_grad_precon_matrix(n1, gamma_grad_theta, b_return_vec)
        
        if b_return_vec:
            return pvec, pvec_inv, grad_precon
        else:
            return np.diag(pvec), np.diag(pvec_inv), grad_precon
    