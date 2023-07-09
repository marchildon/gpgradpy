#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:00:59 2020

@author: andremarchildon
"""

import numpy as np
from os import path
from scipy.optimize import NonlinearConstraint

from .base   import Rescaling

from .base   import CommonFun
from .base   import GpInfo
from .base   import GpHpara
from .base   import GpParaDef
from .base   import GpWellCond


from .eval   import GpEvalModel
from .kernel import Kernel
from .optz   import GpHparaOptz
    
class GaussianProcess(CommonFun, GpInfo, GpHpara, GpParaDef, GpWellCond, 
                      GpEvalModel, Kernel, GpHparaOptz):

    ''' Default options for printing and saving the data '''  
    
    print_txt_data      = False
    save_data_npz       = False
    save_data_txt       = False

    ''' Default sub-optimization options '''
    
    optz_log_hp_theta   = True # More effective optz when this is True
    optz_log_hp_var     = True # Optimize the log of varK, var_fval, and var_fgrad 
    optz_log_hp_kernel  = True # Optimize the log of the hyperparameter of the kernel (if any)
    
    hp_optz_method      = 'SLSQP' # 'trust-constr' or 'SLSQP' (SLSQP is recommended since it has been found to be faster and more robust)
    n_surr_optz_start   = 5     # No. of starts for the optz of the hyperparameters (not used if lkd_optz_start == 'hp_best')
    n_surr_optz_iter    = 250   # Max no. of iter for the optz of the hyperparameters
    hp_optz_obj_tol     = 1e-12 # Optz termination criteria for delta obj
    hp_optz_xtol        = 1e-12 # Optz termination criteria for dist between sol
    
    ''' Options related to the condition number '''
    
    # Ensures the gradient-enahnced correlation matrix is well-conditioned 
    wellcond_mtd_avail = [None,            # no method is used
                          'req_vmin',      # ensure min dist is sufficiently large
                          'req_vmin_frac', # use a vreq * vreq_frac
                          'precon',        # modify how the corr matrix is constructed 
                          'dflt_vmin',     # set min xdist to dist_min_dflt
                          'dflt_vmax']     # set max xdist to dist_max_dflt

    set_eta_mtd_dflt    = 'Kbase_eta' # Used if selected wellcond_mtd does not have a default method, ie wellcond_mtd == None
                                      # 'Kbase_eta' : Use req eta for Kbase, ie eta =  nx / (condmax - 1)
                                      # 'Kbase_eta_w_dim' : Use eta =  nx (dim + 1) / (condmax - 1)
                                      # 'dflt_eta'  : min_nugget_dflt is used

    cond_max            = 1e10  # Used for the optz constraint of the hyperparameters
    cond_max_target     = 1e10  # Used to select the nugget value, also require cond_max_target <= cond_max
    condmax_absolute    = 1e16  # If the condition number is greater than this the Cholesky decomposition is not attempted
    condnum_norm        = 2     # The type of norm used to calculate the condition number (2 or 'fro'), using 2 is recommended
    
    min_nugget_dflt     = 1e-8  # Default min nugget value if set_eta_mtd_dflt == dflt_eta
    dist_min_dflt       = 1     # Used if wellcond_mtd == dflt_vmin
    dist_max_dflt       = 1     # Used if wellcond_mtd == dflt_vmax
    
    # Options if wellcond_mtd == 'req_vmin' or 'req_vmin_frac'
    vreq_frac           = 0.1
    max_vreq_optz_iter  = 5    # Max no. iteration 
    tol_dist2diag_vreq  = 1e-1 # Stop iterating once dist from point log(thetavec) and theta 1vec is less than this tolerance
    
    ''' Options related to the hyperparameters'''
    
    # Methods on how to initialize the optimization for 'max_lkd'
    lkd_optz_start_avail = ['hp_best' 'lhs'] # hp_best requires the use of wellcond_mtd = 'precon'
    lkd_optz_start       = 'hp_best' 
    n_surr_hp_eval       = 40 # No. of points to calculate the lkd if lkd_optz_start = 'hp_best' 
    
    # Initial hyperparameters are used until the no. of eval is equal to this constant
    n_eval_hp_const      = 1   
    
    # Initial hyperparamters values are used until n_eval > n_eval_hp_const
    hp_theta_init        = 1e-2
    hp_varK_init         = 1.0 # Used if varK is optimized numerically, ie not in closed form solution
    hp_var_fval_init     = 0.0 
    hp_var_fgrad_init    = 0.0

    range_theta          = [1e-10, 1e24] 
    range_varK           = [1e-24, 1e8] # Used if varK is optimized numerically (when there is noisy data)
    range_var_fval       = [1e-8, 1e8]
    range_var_fgrad      = [1e-8, 1e8]
    
    b_optz_hp_kernel     = True # Default is to optz the kernel hyperpara if there are any

    ''' Initiate variables ''' 
    
    b_use_data_scl   = None # Depends on wellcond_mtd
    b_has_noisy_data = None
    b_optz_var_fval  = None
    b_optz_var_fgrad = None
    
    _cond_val       = np.nan
    _cond_grad      = np.nan
    
    # Initial data    
    _x_eval_in      = None
    _fval_in        = None
    _std_fval_in    = None
    _grad_in        = None
    _std_grad_in    = None
    _idx_xbest      = None 
    
    # Initiate timers
    _time_chofac    = 0

    def __init__(self, dim, use_grad,
                 kernel_type     = 'SqExp',
                 wellcond_mtd    = 'precon',
                 #
                 folder_name     = None,
                 path_data_surr  = 'baye_data_surr',
                 surr_name       = 'obj_'):        

        ''' Add inputs '''
        
        assert isinstance(dim, int),         'dim must be an integer'
        assert isinstance(use_grad, bool),   'use_grad must be of type bool'
        assert isinstance(kernel_type, str), 'kernel_type must be of type str'
        
        assert wellcond_mtd in self.wellcond_mtd_avail, \
            f'Requested method not available, wellcond_mtd : {wellcond_mtd}'
        
        self.dim      = dim
        self.use_grad = use_grad
        
        if use_grad is False:
            wellcond_mtd = None
        
        self.wellcond_mtd    = wellcond_mtd
        
        self.path_data_surr  = path_data_surr
        self.surr_name       = surr_name
        
        Kernel.__init__(self, kernel_type)
        
        ''' Setup for the path to the data '''
        
        if folder_name is None:
            path_data_surr_all  = path_data_surr
        else:
            path_data_surr_all  = path.join(folder_name, path_data_surr)
        
        self.path_surr_txt      = path_data_surr_all + '.txt'
        self.path_surr_old_txt  = path_data_surr_all + '_old.txt'
        self.path_surr_npz      = path_data_surr_all + '.npz'
        self.path_surr_old_npz  = path_data_surr_all + '_old.npz'

        ''' Set other parameters '''
        
        self.set_mean_fun_op(mean_fun_type = 'poly_ord_0')

        if self.wellcond_mtd == 'precon':
            self.b_use_cond_cstr = False
        else:
            self.b_use_cond_cstr = True
            self.condnum_nlc = NonlinearConstraint(self.return_cond_val, -np.inf, self.cond_max,
                                                   jac = self.return_cond_grad)
            
        if ((self.wellcond_mtd == 'req_vmin') or (self.wellcond_mtd == 'req_vmin_frac') 
            or (self.wellcond_mtd == 'dflt_vmin') or (self.wellcond_mtd == 'dflt_vmax')):
            self.b_use_data_scl = True
        else:
            self.b_use_data_scl = False
        
    def set_data(self, x_eval_in, fval_in, std_fval_in, 
                 grad_in = None, std_grad_in = None, idx_xbest = None):
        
        ''' Check inputs '''
        
        n_eval  = x_eval_in.shape[0]
        fval_in = np.atleast_1d(fval_in).ravel()
        
        assert x_eval_in.ndim == 2, f'x_eval_in must be a 2 array but x_eval_in.ndim = {x_eval_in.ndim}'
        assert n_eval == fval_in.size, 'No. of points do not match with x_eval and fval'
        
        if (std_fval_in is None) or np.any(np.isnan(std_fval_in)):
            self.known_eps_fval = False
        else:
            self.known_eps_fval = True
            std_fval_in = np.atleast_1d(std_fval_in).ravel()
            assert n_eval == std_fval_in.size, f'Size of std_fval is {std_fval_in.size} while it should be {n_eval}'
        
        if grad_in is None:
            assert self.use_grad is False, 'No grad info provided but use_grad was set to True'
            self.has_grad_info   = False
            self.known_eps_fgrad = False
        else:
            assert self.use_grad, 'Grad info provided but use_grad was set to False'
            self.has_grad_info = True
            assert grad_in.ndim == 2, f'grad_in must be a 2 array but grad_in.ndim = {grad_in.ndim}'
            
            assert grad_in.shape == x_eval_in.shape, 'Shape of grad does not match x_eval'
            
            if (std_grad_in is None) or np.any(np.isnan(std_grad_in)):
                self.known_eps_fgrad = False
            else:
                self.known_eps_fgrad = True
                assert std_grad_in.ndim == 2, f'std_grad_in must be a 2 array but std_grad_in.ndim = {std_grad_in.ndim}'
                assert grad_in.shape == std_grad_in.shape, 'Shape of grad does not match std_grad'
        
        # For constrained optz the best sol so far is the one with the lowest 
        # merit function, which may not match with the point with the smallest 
        # objective. If idx_xbest is not provided then it is calculated below
        if idx_xbest is None:
            idx_xbest = np.argmin(fval_in)
        
        ''' Store input data '''
        
        self.n_eval          = fval_in.size
        
        self._x_eval_in      = x_eval_in
        self._fval_in        = fval_in
        self._std_fval_in    = std_fval_in if self.known_eps_fval  else None
        self._grad_in        = grad_in 
        self._std_grad_in    = std_grad_in if self.known_eps_fgrad else None
        
        self._idx_xbest      = idx_xbest
        
        ''' Check if the data is noisy and if its variance is known '''
        
        if self.known_eps_fval:
            self.b_optz_var_fval    = False 
            self.b_fval_zero        = True if (np.max(std_fval_in) < 1e-10) else False
        else:
            self.b_optz_var_fval    = True 
            self.b_fval_zero        = False
            
        if (self.use_grad is False):
            self.b_optz_var_fgrad   = False 
            self.b_fgrad_zero       = True
        elif self.known_eps_fgrad:
            self.b_optz_var_fgrad   = False 
            self.b_fgrad_zero       = True if (np.max(std_grad_in) < 1e-10) else False
        else:
            self.b_optz_var_fgrad   = True 
            self.b_fgrad_zero       = False
        
        if self.b_fval_zero and self.b_fgrad_zero:
            self.b_has_noisy_data   = False # Optimize varK with closed form sol for max lkd
        else:
            self.b_has_noisy_data   = True  # Optimize varK numerically
        
        ''' Remaining setup '''
        
        # Set min nugget 
        self._eta_Kbase, self._eta_Kgrad = self.calc_nugget(self.n_eval)
        self._etaK = self._eta_Kgrad if self.use_grad else self._eta_Kbase
        
        # Data related to rescaling to help with the condition number 
        vreq = self.calc_vreq(n_eval, self.dim)
        
        self._vmin_init     = self.calc_dist_min(x_eval_in)
        self._vmin_req_grad = vreq
        
        self.setup_hp_idx4optz()
        
        if self.b_use_data_scl:
            if self.wellcond_mtd == 'req_vmin':
                x_scl_method = 'set_vmin'
                dist_set     = vreq
            elif self.wellcond_mtd == 'req_vmin_frac':
                x_scl_method = 'set_vmin'
                dist_set     = vreq * self.vreq_frac
            elif self.wellcond_mtd == 'dflt_vmin':
                x_scl_method = 'set_vmin'
                dist_set     = self.dist_min_dflt
            elif self.wellcond_mtd == 'dflt_vmax':
                x_scl_method = 'set_vmax'
                dist_set     = self.dist_max_dflt
            else:
                raise Exception(f'Unknown method wellcond_mtd = {self.wellcond_mtd}')
            
            self.DataScl = Rescaling(idx_xbest, x_eval_in, 
                                      x_scl_method = x_scl_method, dist_set = dist_set)
        
            self.DataScl.set_obj_data(fval_in, std_fval_in, grad_in, std_grad_in)
        else:
            self.Rtensor_init = self.calc_Rtensor(x_eval_in, x_eval_in, 1)
            
    def set_hpara(self, method2set_hp, i_optz, hp_vals = None):
        
        assert type(method2set_hp) is str, 'method2set_hp must be a string'
        
        '''
        Call this method to set up the surrogate for the optimization of the 
        acquisition function. This method will:
          1) store the inputs depending on value of 'method2set_hp'
          2) a) 'stored':   use the hyperparameters at iteration i_optz
             b) 'optz':     optimize the hyperparameters
             c) 'current':  use the current hyperparameters
             d) 'set'       use the data class hp_vals to set the hyperparameters
          3) calculate the Cholesky factorization of the SPD matrix Kern or Kcov
        '''
        
        if method2set_hp == 'stored':
            assert i_optz > 0
            self.set_hp_from_idx(i_optz)
        elif method2set_hp == 'optz':
            self.optz_hp(i_optz)
        elif method2set_hp == 'current':
            assert i_optz > 0
            assert self.hp_vals is not None, 'Cannot use current hp_vals if they have not been set yet'
        elif method2set_hp == 'set':
            assert hp_vals is not None, 'If method2set_hp == "set", then the class hp_vals must be provided'
            self.hp_vals = hp_vals
        else:
            raise Exception(f'Unknown method to set GP hp: method2set_hp = {method2set_hp}')
            
        # Setup for the evaluation of the model
        self.setup_eval_model()
    
    ''' Get initial and scaled data '''
    
    def get_scl_x_w_dist(self):
        
        if self.b_use_data_scl:
            return self.DataScl.get_scl_x_w_dist()
        else:
            return self._x_eval_in, self.Rtensor_init
    
    def x_init_2_scl(self, x_init):
        
        if self.b_use_data_scl:
            return self.DataScl.x_init_2_scl(x_init)
        else:
            return x_init
    
    def x_scl_2_init(self, x_scl):
        
        if self.b_use_data_scl:
            return self.DataScl.x_scl_2_init(x_scl)
        else:
            return x_scl
    
    def get_init_eval_data(self):
        return self._fval_in, self._std_fval_in, self._grad_in, self._std_grad_in
    
    def get_scl_eval_data(self, theta):
        
        fval_init, std_fval_init, fgrad_init, std_fgrad_init = self.get_init_eval_data()
        
        fval_scl, std_fval_scl, fgrad_scl, std_fgrad_scl \
            = self.data_init_2_scl(theta, fval_init, std_fval_init, fgrad_init, std_fgrad_init)[:4]
        
        std_fval_scl_out  = std_fval_scl  if self.known_eps_fval  else None
        std_fgrad_scl_out = std_fgrad_scl if self.known_eps_fgrad else None
        
        return fval_scl, std_fval_scl_out, fgrad_scl, std_fgrad_scl_out
        
    def data_init_2_scl(self, theta, 
                        mu_in      = None, sig_in      = None, 
                        dmudx_in   = None, dsigdx_in   = None, 
                        d2mudx2_in = None, d2sigdx2_in = None):
        
        if self.b_use_data_scl:
            return self.DataScl.obj_init_2_scl(mu_in,      sig_in, 
                                               dmudx_in,   dsigdx_in, 
                                               d2mudx2_in, d2sigdx2_in)
        else:
            return mu_in, sig_in, dmudx_in, dsigdx_in, d2mudx2_in, d2sigdx2_in
    
    def data_scl_2_init(self, theta, 
                        mu_scl      = None, sig_scl      = None, 
                        dmudx_scl   = None, dsigdx_scl   = None, 
                        d2mudx2_scl = None, d2sigdx2_scl = None):
        
        if self.b_use_data_scl:
            return self.DataScl.obj_scl_2_init(mu_scl,      sig_scl, 
                                               dmudx_scl,   dsigdx_scl, 
                                               d2mudx2_scl, d2sigdx2_scl)
        else:
            return mu_scl, sig_scl, dmudx_scl, dsigdx_scl, d2mudx2_scl, d2sigdx2_scl
