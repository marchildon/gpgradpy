#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 15:59:10 2022

@author: andremarchildon
"""

import numpy as np 

from . import CommonFun

'''
The clases in this file are used to rescale the data in the parameter space. 
This is used by the rescaling method, which is presented in the following paper:
    A Non-intrusive Solution to the Ill-Conditioning Problem of the 
    Gradient-Enhanced Gaussian Covariance Matrix for Gaussian Processes
'''

class RescalingXdata:
    
    def x_init_2_scl(self, x_init):
        if x_init.ndim == 1:
            return (x_init - self.x_shift) * self.xvec_scale
        elif x_init.ndim == 2:
            return (x_init - self.x_shift[None,:]) * self.xvec_scale[None,:]
        else:
            raise Exception('Shape of x_init = {x_init.shape}')
    
    def x_scl_2_init(self, x_scl):
        if x_scl.ndim == 1:
            return x_scl / self.xvec_scale + self.x_shift
        elif x_scl.ndim == 2:
            return x_scl / self.xvec_scale[None,:] + self.x_shift[None,:]
        else:
            raise Exception('Shape of x_scl = {x_scl.shape}')
        
    def dist_init_2_scl(self, dist_init):
        return dist_init * np.mean(self.xvec_scale) # Scale by the mean of xvec_scale 
    
    def dist_scl_2_init(self, dist_scl):
        return dist_scl / np.mean(self.xvec_scale) # Scale by the mean of xvec_scale  
    
    def set_xscale_data(self, x_shift_in = None, xvec_scale_in = None):
        
        ''' Rescale x data '''
            
        x_shift, xvec_scale = self._calc_x_shift_n_scale(x_shift_in, xvec_scale_in)
        self.x_shift        = x_shift
        self.xvec_scale     = xvec_scale
        
        # Scale init data 
        self._calc_n_set_scl_x()
        
        ''' Rescale the rest of the data '''
        
        if self._obj_data_set:
            self._calc_n_set_scl_obj()
            
        if self._nlc_data_set:
            self._calc_n_set_scl_nlc_data()
            
        if self._boxcon_set:
            self._calc_n_set_scl_boxcon()
            
        if self._lincon_set:
            self._calc_n_set_scl_lincon()
        
        if self._nlc_data_set:
            self._calc_n_set_scl_nonlincon()
    
    def _calc_x_shift_n_scale(self, x_shift_in = None, xvec_scale_in = None):
        
        # The inputs x_shift_in and xvec_scale_in are used to scale the data 
        # The vector xvec_scale_in is scaled by a positive constant to ensure 
        # self.x_scl_method is satisfied, which gives xvec_scale
        
        ''' Check inputs '''
        
        if x_shift_in is None:
            if self.use_x_shift:
                x_shift = self.x_init[self.idx_xbest, :]
            else:
                x_shift = np.zeros(self.dim)
        else:
            assert x_shift_in.size == self.dim, f'Wrong size for x_shift_in: {x_shift_in.shape}'
            assert x_shift_in.dim <= 1, f'Wrong shape for x_shift_in: {x_shift_in.shape}'
            x_shift = 1 * np.atlest_1d(x_shift_in)
        
        if xvec_scale_in is None:
            xvec_scale_in = np.ones(self.dim)
        else:
            assert xvec_scale_in.size == self.dim, f'Wrong shape for xvec_scale_in: {xvec_scale_in.shape}'
            assert np.all(xvec_scale_in > 0), f'All entries must be positive: {xvec_scale_in}'
        
        ''' Scale xvec_scale_in to ensure the x_scl_method is satisfied '''
        
        # Initial scaling of data
        x_scl_v1 = (self.x_init - x_shift[None,:]) * xvec_scale_in[None,:]
        
        if (self.nx == 1) or (self.x_scl_method is None):
            coeff_x_scl = 1
            # x_shift     = np.zeros(self.dim)
        elif self.x_scl_method == 'set_vmin':
            dist_set    = self.dist_set if (self.dist_set is not None) else self.vmin_dflt
            coeff_x_scl = self._calc_x2_isoscaling(x_scl_v1, dist_set, True)
        elif self.x_scl_method == 'set_vmax':
            dist_set    = self.dist_set if (self.dist_set is not None) else self.vmax_dflt
            coeff_x_scl = self._calc_x2_isoscaling(x_scl_v1, dist_set, False)
        else:
            raise Exception('Method of x_scl_method = {self.x_scl_method} is unavailable')
            
        xvec_scale = xvec_scale_in * coeff_x_scl
        
        return x_shift, xvec_scale
    
    def _calc_x2_isoscaling(self, x_init, set_dist, set_vmin = True):
        
        if set_vmin:
            dist_init = CommonFun.calc_dist_min(x_init)
            dist_init = np.max((self.tol_min_dist_x, dist_init))
        else:
            dist_init = CommonFun.calc_dist_max(x_init)
            
        return set_dist / dist_init
    
    def _calc_n_set_scl_x(self):
        
        self.x_scl       = self.x_init_2_scl(self.x_init)
        self.Rtensor_scl = CommonFun.calc_Rtensor(self.x_scl, self.x_scl, 1)
    
class RescalingObjData:
    
    def obj_init_2_scl(self, mu_in = None, sig_in      = None, 
                       dmudx_in    = None, dsigdx_in   = None, 
                       d2mudx2_in  = None, d2sigdx2_in = None):
        
        # The rescaling is the same for the gradient of the standard deviation 
        # of the model's uncertainty and for the standard deviation of the 
        # uncertainty of the gradient. The same applies to the Hessian
    
        assert self._obj_data_set, 'Must call set_obj_data prior to this method'
        
        scale_inv    = 1 / self.xvec_scale[None,:]
        scl_grad_vec = scale_inv    * self.obj_scale
        scl_hess_vec = scale_inv**2 * self.obj_scale
        
        # Unscale the mean and standard deviation of the data
        mu_scl       = None if mu_in       is None else (mu_in - self.obj_shift) * self.obj_scale
        sig_scl      = None if sig_in      is None else sig_in * self.obj_scale 
        
        dmudx_scl    = None if dmudx_in    is None else dmudx_in    * scl_grad_vec
        dsigdx_scl   = None if dsigdx_in   is None else dsigdx_in   * scl_grad_vec
        
        d2mudx2_scl  = None if d2mudx2_in  is None else d2mudx2_in  * scl_hess_vec
        d2sigdx2_scl = None if d2sigdx2_in is None else d2sigdx2_in * scl_hess_vec
        
        return mu_scl, sig_scl, dmudx_scl, dsigdx_scl, d2mudx2_scl, d2sigdx2_scl

    def obj_scl_2_init(self, mu_scl = None, sig_scl      = None, 
                       dmudx_scl    = None, dsigdx_scl   = None, 
                       d2mudx2_scl  = None, d2sigdx2_scl = None):
        
        # The rescaling is the same for the gradient of the standard deviation 
        # of the model's uncertainty and for the standard deviation of the 
        # uncertainty of the gradient. The same applies to the Hessian
        
        assert self._obj_data_set, 'Must call set_obj_data prior to this method'
        
        scl_grad_vec = self.xvec_scale[None,:]    / self.obj_scale
        scl_hess_vec = self.xvec_scale[None,:]**2 / self.obj_scale
        
        # Unscale the mean and standard deviation of the data
        mu       = None if mu_scl       is None else mu_scl  / self.obj_scale + self.obj_shift
        sig      = None if sig_scl      is None else sig_scl / self.obj_scale
        
        dmudx    = None if dmudx_scl    is None else dmudx_scl    * scl_grad_vec
        dsigdx   = None if dsigdx_scl   is None else dsigdx_scl   * scl_grad_vec
        
        d2mudx2  = None if d2mudx2_scl  is None else d2mudx2_scl  * scl_hess_vec
        d2sigdx2 = None if d2sigdx2_scl is None else d2sigdx2_scl * scl_hess_vec
        
        return mu, sig, dmudx, dsigdx, d2mudx2, d2sigdx2
    
    def set_obj_scaling(self, obj_shift = None, obj_scale = None):
        
        assert self._obj_data_set, 'Must call set_obj_data prior to this method'
        
        # Store obj data scaling
        if obj_shift is not None:
            self.obj_shift = obj_shift
        
        if obj_scale is not None:
            self.obj_scale = obj_scale
        
        # Scale init data 
        self._calc_n_set_scl_obj()
           
    def _calc_obj_scaling(self, obj_init):
        
        if self.use_obj_shift:
            obj_shift = obj_init[self.idx_xbest]
        else:
            obj_shift = 0
        
        if (obj_init.size == 1) or (self.obj_scl_method is None):
            obj_scale = 1
        elif self.obj_scl_method =='dflt_max':
            range_init = np.max((self.tol_min_range_obj, np.max(obj_init) - np.min(obj_init)))
            obj_scale  = self.rangeobj_max_dflt / range_init
        else:
            raise Exception(f'Unavailable method of obj_scl_method = {self.obj_scl_method}')
        
        return obj_shift, obj_scale
        
    def _calc_n_set_scl_obj(self):
        
        # Recalculate scaling for the objective data
        self.obj_scl, self.std_obj_scl, self.grad_scl, self.std_grad_scl \
            = self.obj_init_2_scl(self.obj_init,  self.std_obj_init, 
                                  self.grad_init, self.std_grad_init)[:4]
 
class RescalingLincon:

    def boxcon_init_2_scl(self, lb_init, ub_init):
        
        assert self._xdata_set, 'Must call the method set_xdata prior to this method'
        
        lb_scl = self.xvec_scale * (lb_init - self.x_shift)
        ub_scl = self.xvec_scale * (ub_init - self.x_shift)
        
        return lb_scl, ub_scl
    
    def _calc_n_set_scl_boxcon(self):
        
        self.boxcon_lb_scl, self.boxcon_ub_scl \
            = self.boxcon_init_2_scl(self.boxcon_lb_init, self.boxcon_ub_init)
    
    def lincon_init_2_scl(self, A_init, lb_init, ub_init):
        
        assert self._xdata_set, 'Must call the method set_xdata prior to this method'
        
        A_scl  = A_init * (1 / self.xvec_scale[None,:])
        lb_scl = lb_init - np.dot(A_init, self.x_shift)
        ub_scl = ub_init - np.dot(A_init, self.x_shift)
        
        return A_scl, lb_scl, ub_scl 
    
    def _calc_n_set_scl_lincon(self):
        
        self.lincon_A_scl, self.lincon_lb_scl, self.lincon_ub_scl  \
            = self.lincon_init_2_scl(self.lincon_A_init, self.lincon_lb_init, self.lincon_ub_init)
    
class RescalingNonlincon:
    
    def nlc_init_2_scl(self, mu_in, sig_in, 
                       dmudx_in   = None, dsigdx_in   = None, 
                       d2mudx2_in = None, d2sigdx2_in = None):
    
        assert self._nlc_data_set, 'Must call set_nlc_data prior to this method'
        assert mu_in.ndim    == 2, f'Unexpected shape for mu_in of {mu_in.shape}'
        
        if dmudx_in is not None:
            assert dmudx_in.ndim == 3, f'Unexpected shape for dmudx_in of {dmudx_in.shape}'
        
        normz_xvec_scale_inv = 1 / self.xvec_scale[None,None,:]
        
        # Unscale the mean and standard deviation of the data
        mu_scl       = (mu_in - self.nlc_shift[None,:]) * self.nlc_scale
        sig_scl      = sig_in * self.nlc_scale 
        
        # Scale the gradients
        scl_vec      = normz_xvec_scale_inv * self.nlc_scale
        dmudx_scl    = None if dmudx_in  is None else dmudx_in  * scl_vec
        dsigdx_scl   = None if dsigdx_in is None else dsigdx_in * scl_vec
        
        # Scale the Hessians
        scl_vec      = normz_xvec_scale_inv**2 * self.nlc_scale
        d2mudx2_scl  = None if d2mudx2_in  is None else d2mudx2_in  * scl_vec
        d2sigdx2_scl = None if d2sigdx2_in is None else d2sigdx2_in * scl_vec
        
        return mu_scl, sig_scl, dmudx_scl, dsigdx_scl, d2mudx2_scl, d2sigdx2_scl

    def nlc_scl_2_init(self, mu_scl = None, sig_scl      = None, 
                       dmudx_scl    = None, dsigdx_scl   = None, 
                       d2mudx2_scl  = None, d2sigdx2_scl = None):
        
        # The rescaling is the same for the gradient of the standard deviation 
        # of the model's uncertainty and for the standard deviation of the 
        # uncertainty of the gradient. The same applies to the Hessian
        
        assert self._nlc_data_set, 'Must call set_nlc_data prior to this method'
        assert mu_scl.ndim == 2,  f'Unexpected shape for mu_scl of {mu_scl.shape}'
        
        scl_grad_vec  = self.xvec_scale[None,None,:]    / self.nlc_scale
        scl_hess_vec  = self.xvec_scale[None,None,:]**2 / self.nlc_scale
        
        mu       = None if mu_scl  is None else mu_scl  / self.nlc_scale + self.nlc_shift[None,:]
        sig      = None if sig_scl is None else sig_scl / self.nlc_scale
        
        if dmudx_scl is None:
            dmudx = None 
        else:
            assert dmudx_scl.ndim == 3, f'Unexpected shape for dmudx_scl of {dmudx_scl.shape}'
            dmudx = None if dmudx_scl is None else dmudx_scl  * scl_grad_vec
            
        dsigdx   = None if dsigdx_scl is None else dsigdx_scl * scl_grad_vec
        
        d2mudx2  = None if d2mudx2_scl  is None else d2mudx2_scl  * scl_hess_vec
        d2sigdx2 = None if d2sigdx2_scl is None else d2sigdx2_scl * scl_hess_vec
        
        return mu, sig, dmudx, dsigdx, d2mudx2, d2sigdx2
    
    def set_nlc_scaling(self, nlc_shift = None, nlc_scale = None):
        
        # The rescaling is the same for the gradient of the standard deviation 
        # of the model's uncertainty and for the standard deviation of the 
        # uncertainty of the gradient. The same applies to the Hessian
        
        assert self._nlc_data_set, 'Must call set_nlc_data prior to this method'
        
        # Store obj data scaling
        if nlc_shift is not None:
            self.nlc_shift = nlc_shift
        
        if nlc_scale is not None:
            self.nlc_scale = nlc_scale
        
        # Scale init data 
        self._calc_n_set_scl_nlc()
           
    def _calc_nlc_scaling(self, nlc_init):
        
        if self.use_nlc_shift:
            nlc_shift = nlc_init[self.idx_xbest,:]
        else:
            nlc_shift = np.zeros(self.nlc_n)
        
        nx = self.x_init.shape[0]
        if (nx == 1) or (self.nlc_scl_method is None):
            nlc_scale = 1
        elif self.nlc_scl_method == 'obj_scl':
            assert self._obj_data_set, 'Must call set_obj_data prior to this method'
            nlc_scale = self.obj_scale
        else:
            raise Exception(f'Unavailable method of obj_scl_method = {self.obj_scl_method}')
        
        return nlc_shift, nlc_scale
        
    def _calc_n_set_scl_nlc(self):
        
        # Recalculate scaling for the objective data
        self.nlc_val_scl, self.nlc_std_val_scl, self.nlc_grad_scl, self.nlc_std_grad_scl \
            = self.nlc_init_2_scl(self.nlc_val_init,  self.nlc_std_val_init, 
                                  self.nlc_grad_init, self.nlc_std_grad_init)[:4]
    
class Rescaling(RescalingXdata, RescalingLincon, RescalingObjData, RescalingNonlincon):
    
    # Tolerances
    tol_min_range_obj = 1e-20
    tol_min_dist_x    = 1e-14
    
    ''' Default parameters '''
    
    _xdata_set           = False
    _obj_data_set        = False 
    _nlc_data_set        = False
    _boxcon_set          = False
    _lincon_set          = False
    
    use_x_shift          = True
    x_scl_method         = None
    x_scl_method_avail   = ['set_vmin', 'set_vmax', None]
    
    use_obj_shift        = True
    obj_scl_method       = None
    obj_scl_method_avail = ['dflt_max', None]
    
    use_nlc_shift        = True
    nlc_scl_method       = None
    nlc_scl_method_avail = ['obj_scl', None]
    
    dist_set             = None
    vmin_dflt            = 1 
    vmax_dflt            = 1
    rangeobj_max_dflt    = 100
    
    ''' Initialize data '''
    
    # Init data
    idx_xbest         = None
    x_init            = None
    
    obj_init          = None
    std_obj_init      = None
    grad_init         = None
    std_grad_init     = None
    
    nlc_val_init      = None
    nlc_std_val_init  = None
    nlc_grad_init     = None
    nlc_std_grad_init = None
    
    # Scaled data
    x_scl             = None
    
    obj_scl           = None
    std_obj_scl       = None
    grad_scl          = None
    std_grad_scl      = None
    
    nlc_val_scl       = None
    nlc_std_val_scl   = None
    nlc_grad_scl      = None
    nlc_std_grad_scl  = None
    
    # Scaling parameters
    x_shift    = np.nan
    xvec_scale = np.nan
    
    obj_shift  = np.nan
    obj_scale  = np.nan
    
    nlc_shift  = np.nan
    nlc_scale  = np.nan
    
    def __init__(self, idx_xbest, x_init, use_x_shift = True, x_scl_method = None, dist_set = None):
        
        assert x_init.ndim == 2, f'x_init needs to be 2D but it has the shape {x_init.shape}'
        assert x_scl_method in self.x_scl_method_avail, f'Requested x_scl_method = {x_scl_method} is not available'
        
        self._xdata_set     = True
        self.nx, self.dim   = x_init.shape
        
        # Store init data
        self.idx_xbest      = idx_xbest
        self.x_init         = x_init
        
        self.use_x_shift    = use_x_shift
        self.x_scl_method   = x_scl_method
        self.dist_set       = dist_set # If None then either vmin_dflt or vmax_dflt is used
        
        # Calculate and set scaling parameters
        self.set_xscale_data()
    
    def get_init_x(self):
        return self.x_init
    
    def get_scl_x(self):
        return self.x_scl
    
    def get_scl_x_w_dist(self):
        return self.x_scl, self.Rtensor_scl
    
    def set_obj_data(self, obj_init, std_obj_init, grad_init, std_grad_init, use_obj_shift = True, obj_scl_method = 'dflt_max'):
        
        assert self._xdata_set, 'Must call the method set_xdata prior to set_obj_data'
        assert obj_scl_method in self.obj_scl_method_avail, f'Requested obj_scl_method = {obj_scl_method} is not available'
        assert self.nx == obj_init.size, 'Dimension of nx = {self.nx} do not match with size of obj_init: {obj_init.size}'
        
        self._obj_data_set   = True
        
        # Store init data
        self.obj_init        = obj_init
        self.std_obj_init    = std_obj_init
        self.grad_init       = grad_init
        self.std_grad_init   = std_grad_init
        
        self.obj_scl_method  = obj_scl_method
        self.use_obj_shift   = use_obj_shift
        
        # Calculate and set scaling parameters
        obj_shift, obj_scale = self._calc_obj_scaling(obj_init)
        self.set_obj_scaling(obj_shift, obj_scale)
        
    def get_init_obj_data(self):
        assert self._obj_data_set, 'Must call the method set_obj_data prior to this method'
        return self.obj_init, self.std_obj_init, self.grad_init, self.std_grad_init
        
    def get_scl_obj_data(self):
        assert self._obj_data_set, 'Must call the method set_obj_data prior to this method'
        return self.obj_scl, self.std_obj_scl, self.grad_scl, self.std_grad_scl
        
    def set_nlc_data(self, nlc_val_init, nlc_std_val_init, nlc_grad_init, nlc_std_grad_init, use_nlc_shift = False, nlc_scl_method = 'dflt_max'):
        
        nx, self.nlc_n = nlc_val_init.shape
        assert self._xdata_set, 'Must call the method set_xdata prior to set_obj_data'
        assert nx == self.nx, f'Unecpected shape for nlc_val_init of {nlc_val_init.shape} when nx = {nx}'
        assert nlc_scl_method in self.nlc_scl_method_avail, f'Requested nlc_scl_method = {nlc_scl_method} is not available'
        
        self._nlc_data_set      = True
        
        # Store init data
        self.nlc_val_init       = nlc_val_init
        self.nlc_std_val_init   = nlc_std_val_init
        self.nlc_grad_init      = nlc_grad_init
        self.nlc_std_grad_init  = nlc_std_grad_init
        
        self.nlc_scl_method     = nlc_scl_method
        self.use_nlc_shift      = use_nlc_shift
        
        # Calculate and set scaling parameters
        nlc_shift, nlc_scale = self._calc_nlc_scaling(nlc_val_init)
        self.set_nlc_scaling(nlc_shift, nlc_scale)
    
    def get_init_nlc_data(self):
        assert self._nlc_data_set, 'Must call the method set_nlc_data prior to this method'
        return self.nlc_val_init, self.nlc_std_val_init, self.nlc_grad_init, self.nlc_std_grad_init
    
    def get_scl_nlc_data(self):
        assert self._nlc_data_set, 'Must call the method set_nlc_data prior to this method'
        return self.nlc_val_scl, self.nlc_std_val_scl, self.nlc_grad_scl, self.nlc_std_grad_scl
    
    ''' Set and get scalled constraints '''    
    
    def set_boxcon(self, lb_init, ub_init):
        
        assert self._xdata_set, 'Must call set_xdata before this method'
        
        self._boxcon_set    = True
        
        self.boxcon_lb_init = lb_init 
        self.boxcon_ub_init = ub_init
        
        self._calc_n_set_scl_boxcon()
   
    def get_init_boxcon(self):
        assert self._boxcon_set, 'Must call set_boxcon before this method'
        return self.boxcon_lb_init, self.boxcon_ub_init
        
    def get_scl_boxcon(self):
        assert self._boxcon_set, 'Must call set_boxcon before this method'
        return self.boxcon_lb_scl, self.boxcon_ub_scl
        
    def set_lincon(self, A_init, lb_init, ub_init):
        
        assert self._xdata_set, 'Must call set_xdata before this method'
        
        self._lincon_set    = True
        
        self.lincon_A_init  = A_init
        self.lincon_lb_init = lb_init 
        self.lincon_ub_init = ub_init
        
        self._calc_n_set_scl_lincon()
      
    def get_init_lincon(self):
        assert self._lincon_set, 'Must call set_lincon before this method'
        return self.lincon_A_init, self.lincon_lb_init, self.lincon_ub_init 
      
    def get_scl_lincon(self):
        assert self._lincon_set, 'Must call set_lincon before this method'
        return self.lincon_A_scl, self.lincon_lb_scl, self.lincon_ub_scl
    