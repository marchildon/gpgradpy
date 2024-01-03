#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:00:59 2020

@author: andremarchildon
"""

import numpy as np
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

    ''' Options for the optimization of the hyperparameters '''
    
    optz_mtd            = 'SLSQP' # 'trust-constr' or 'SLSQP' (SLSQP is recommended since it has been found to be faster and more robust)
    optz_n_x0           = 5     # No. of starts for the optz of the hyperparameters (not used if lkd_optz_start_mtd == 'hp_best')
    optz_iter_max       = 250   # Max no. of iter for the optz of the hyperparameters
    optz_tol_obj        = 1e-12 # Optz termination criteria for delta obj
    optz_tol_x          = 1e-12 # Optz termination criteria for dist between sol
    
    optz_log_hp_theta   = True # More effective optz when this is True
    optz_log_hp_var     = True # Optimize the log of varK, var_fval, and var_fgrad 
    optz_log_hp_kernel  = True # Optimize the log of the hyperparameter of the kernel (if any)
    
    ''' Options for the marginal log-likelihood '''
    
    lkd_use_adj_mtd     = True # The adjoint method is more efficient than the forward method
    
    # Methods on how to initialize the optimization for 'max_lkd'
    lkd_optz_start_avail = ['hp_best' 'lhs'] # hp_best requires the use of wellcond_mtd = 'precon'
    lkd_optz_start_mtd   = 'hp_best' 
    lkd_hp_best_n_eval   = 40 # No. of points to calculate the lkd if lkd_optz_start_mtd = 'hp_best' 
    
    # Penalty to avoid having large values of sigK
    lkd_varK_pnlt_use    = False
    lkd_varK_pnlt_lb_var = 0.1 # Min value for the variance of the evaluation of f
    lkd_varK_pnlt_c1     = 1.0
    lkd_varK_pnlt_c2     = 10.0
    
    ''' Options related to the hyperparameters'''
    
    # Initial hyperparameters are used until the no. of eval is equal to this constant
    hp_const_n_eval      = 1   
    
    # Initial hyperparamters values are used until n_eval > hp_const_n_eval
    hp_theta_init        = 1e-2
    hp_varK_init         = 1.0 # Used if varK is optimized numerically, ie not in closed form solution
    hp_kernel_init       = np.nan
    hp_var_fval_init     = 0.0 
    hp_var_fgrad_init    = 0.0

    hp_theta_range       = [1e-18 , 1e24] 
    hp_varK_range        = [1e-24, 1e14] # Used if varK is optimized numerically (when there is noisy data)
    hp_kernel_range      = [np.nan, np.nan]
    hp_var_fval_range    = [1e-8, 1e8]
    hp_var_fgrad_range   = [1e-8, 1e8]
    
    ''' Options related to the ill-conditioning of the covariance matrix '''
    
    # Ensures the gradient-enahnced correlation matrix is well-conditioned 
    wellcond_mtd_avail  = [None,            # no method is used
                           'req_vmin',      # ensure min dist is sufficiently large
                           'req_vmin_frac', # use a vreq * cond_vreq_frac
                           'precon',        # modify how the corr matrix is constructed 
                           'dflt_vmin',     # set min xdist to cond_dist_min_dflt
                           'dflt_vmax']     # set max xdist to cond_dist_max_dflt
    
    cond_eta_set_mtd    = 'Kbase_eta' # Used if selected wellcond_mtd does not have a default method, ie wellcond_mtd == None
                                      # 'Kbase_eta' : Use req eta for Kbase, ie eta =  n_eval / (cond_max - 1)
                                      # 'Kbase_eta_w_dim' : Use eta =  n_eval (dim + 1) / (cond_max - 1)
                                      # 'dflt_eta'  : cond_eta_dflt is used
    cond_eta_is_const   = True  # If False for the precon method then a variable nugget is used
    cond_eta_dflt       = 1e-8  # Default min nugget value if cond_eta_set_mtd == dflt_eta
    

    # Condition number: cond_max_target <= cond_max <= cond_max_abs
    cond_max_target     = 1e10  # Used to select the nugget value
    cond_max            = 1e10  # Used for the optz constraint of the hyperparameters
    cond_max_abs        = 1e16  # Cholesky decomposition is not attempted if the condition number is greater than this value 
    cond_norm           = 2     # The type of norm used to calculate the condition number (2 or 'fro'), using 2 is recommended
    
    cond_dist_min_dflt  = 1     # Used if wellcond_mtd == dflt_vmin
    cond_dist_max_dflt  = 1     # Used if wellcond_mtd == dflt_vmax
    
    # Options if wellcond_mtd == 'req_vmin' or 'req_vmin_frac'
    cond_vreq_frac      = 0.1
    cond_vreq_max_iter  = 5    # Max no. iteration 
    cond_vreq_iter_tol  = 1e-1 # Stop iterating once dist from point log(thetavec) and theta 1vec is less than this tolerance
    
    ''' Initiate variables ''' 
    
    b_optz_hp_kernel    = True # Default is to optz the kernel hyperpara if there are any
    b_use_data_scl      = None # Depends on wellcond_mtd
    b_has_noisy_data    = None
    b_optz_var_fval     = None
    b_optz_var_fgrad    = None
    
    _cond_val           = np.nan
    _cond_grad          = np.nan
    
    _x_eval_in          = None
    _fval_in            = None
    _std_fval_in        = None
    _grad_in            = None
    _std_grad_in        = None
    bvec_use_grad       = None
    
    # Initiate timers
    _time_chofac        = 0

    def __init__(self, dim, use_grad,
                 kernel_type    = 'SqExp',
                 wellcond_mtd   = 'precon',
                 mean_fun_type  = 'poly_ord_0',
                 path_data_surr = 'baye_data_surr',
                 surr_name      = 'obj_'):
        '''
        Parameters
        ----------
        dim : int
            Dimension of the parameter space.
        use_grad : bool
            True if gradients will be provided.
        kernel_type : str, optional
            Indicates the type of kernel to use. See the class Kernel for all 
            options. The default is 'SqExp'.
        wellcond_mtd : str, optional
            Indicates the method used to overcome the ill-conditioning of the 
            covariance matrix. The default is 'precon'.
        path_data_surr : str, optional
            Full path with the name of file to save the data from the GP. 
            The default is 'baye_data_surr'.
        surr_name : str, optional
            When saving the data this string is prepended to the variable 
            names, see the class GpParaDef. This is used if there are several 
            GPs that are constructed. For example when there is an objective 
            and nonlinear constraints. The default is 'obj_'.
        '''

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
        
        self.path_surr_txt      = path_data_surr + '.txt'
        self.path_surr_old_txt  = path_data_surr + '_old.txt'
        self.path_surr_npz      = path_data_surr + '.npz'
        self.path_surr_old_npz  = path_data_surr + '_old.npz'

        ''' Set other parameters '''
        
        self.set_mean_fun_op(mean_fun_type = mean_fun_type)

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
        
    def set_data(self, x_eval, fval, std_fval, 
                 grad = None, std_grad = None, bvec_use_grad = None):
        '''
        Parameters
        ----------
        x_eval : 2D numpy array of size [n_eval, dim]
            Evaluations points.
        fval : 1D numpy array of length n_eval
            Function evaluation at the rows in x_eval.
        std_fval : 1D numpy array of length n_eval
            Standard deviations for the uncertainty in the evaluation of fval.
        grad : 2D numpy array of size [n_eval, dim], optional
            Gradient evaluations at the rows of x_eval. The default is None.
        std_grad : 2D numpy array of size [n_eval, dim], optional
            Standard deviations for the uncertainty in the evaluation of grad. 
            The default is None.
        bvec_use_grad : 1D numpy array of bool of length n_eval, optional
            Indicates which gradients in grad are used to construct the 
            surrogate. The default is None, in which case all gradients are used.
            The default is None.
        '''
        
        ''' Check inputs '''
        
        # Calculate the no. of function evaluations and gradients to use
        n_eval = fval.size
        
        if self.use_grad:
            if bvec_use_grad is None:
                n_grad = n_eval 
            else:
                n_grad = np.sum(bvec_use_grad)
                # assert self.b_use_data_scl == False, 'The class Rescaling is not setup for the case when not all gradients are used'
                assert bvec_use_grad.size == n_eval, \
                    f'Length of bvec_use_grad is {bvec_use_grad.size} but it should be n_eval = {n_eval}'
                assert grad.shape[0] == n_grad, \
                    f'No. of rows of grad is {grad.shape[0]} but it should be n_grad = {n_grad}'
        else:
            assert bvec_use_grad is None, 'bvec_use_grad must be None if grads are not used for the GP'
            n_grad = 0
            
        
        self.n_eval = n_eval
        self.n_grad = n_grad
        self.n_data = n_eval + n_grad * self.dim
        
        fval = np.atleast_1d(fval).ravel()
        
        assert x_eval.ndim == 2, f'x_eval must be a 2 array but x_eval.ndim = {x_eval.ndim}'
        assert n_eval == fval.size, 'No. of points do not match with x_eval and fval'
        
        if (std_fval is None) or np.any(np.isnan(std_fval)):
            self.known_eps_fval = False
        else:
            self.known_eps_fval = True
            std_fval = np.atleast_1d(std_fval).ravel()
            assert n_eval == std_fval.size, f'Size of std_fval is {std_fval.size} while it should be {n_eval}'
        
        if grad is None:
            assert self.use_grad is False, 'No grad info provided but use_grad was set to True'
            self.has_grad_info   = False
            self.known_eps_fgrad = False
        else:
            assert self.use_grad, 'Grad info provided but use_grad was set to False'
            self.has_grad_info = True
            assert grad.ndim == 2, f'grad must be a 2 array but grad.ndim = {grad.ndim}'
            
            assert grad.shape == (n_grad, self.dim), 'Shape of grad does not match x_eval'
            
            if (std_grad is None) or np.any(np.isnan(std_grad)):
                self.known_eps_fgrad = False
            else:
                self.known_eps_fgrad = True
                assert std_grad.ndim == 2, f'std_grad must be a 2 array but std_grad.ndim = {std_grad.ndim}'
                assert grad.shape == std_grad.shape, 'Shape of grad does not match std_grad'
        
        ''' Store input data '''
        
        self._x_eval_in    = x_eval
        self._fval_in      = fval
        self._std_fval_in  = std_fval if self.known_eps_fval  else None
        self._grad_in      = grad 
        self._std_grad_in  = std_grad if self.known_eps_fgrad else None
        
        self.bvec_use_grad = bvec_use_grad
        
        ''' Check if the data is noisy and if its variance is known '''
        
        if self.known_eps_fval:
            self.b_optz_var_fval    = False 
            self.b_fval_zero        = True if (np.max(std_fval) < 1e-10) else False
        else:
            self.b_optz_var_fval    = True 
            self.b_fval_zero        = False
            
        if (self.use_grad is False):
            self.b_optz_var_fgrad   = False 
            self.b_fgrad_zero       = True
        elif self.known_eps_fgrad:
            self.b_optz_var_fgrad   = False 
            self.b_fgrad_zero       = True if (np.max(std_grad) < 1e-10) else False
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
        
        self._vmin_init     = self.calc_dist_min(x_eval)
        self._vmin_req_grad = vreq
        
        self.setup_hp_idx4optz()
        
        if self.b_use_data_scl:
            if self.wellcond_mtd == 'req_vmin':
                x_scl_method = 'set_vmin'
                dist_set     = vreq
            elif self.wellcond_mtd == 'req_vmin_frac':
                x_scl_method = 'set_vmin'
                dist_set     = vreq * self.cond_vreq_frac
            elif self.wellcond_mtd == 'dflt_vmin':
                x_scl_method = 'set_vmin'
                dist_set     = self.cond_dist_min_dflt
            elif self.wellcond_mtd == 'dflt_vmax':
                x_scl_method = 'set_vmax'
                dist_set     = self.cond_dist_max_dflt
            else:
                raise Exception(f'Unknown method wellcond_mtd = {self.wellcond_mtd}')
            
            self.DataScl = Rescaling(x_eval, x_scl_method = x_scl_method, dist_set = dist_set)
            self.DataScl.set_obj_data(fval, std_fval, grad, std_grad)
        else:
            self.Rtensor_init = self.calc_Rtensor(x_eval, x_eval, 1)
            
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
    
    def get_scl_eval_data(self):
        
        fval_init, std_fval_init, fgrad_init, std_fgrad_init = self.get_init_eval_data()
        
        fval_scl, std_fval_scl, fgrad_scl, std_fgrad_scl \
            = self.data_init_2_scl(fval_init, std_fval_init, fgrad_init, std_fgrad_init)[:4]
        
        std_fval_scl_out  = std_fval_scl  if self.known_eps_fval  else None
        std_fgrad_scl_out = std_fgrad_scl if self.known_eps_fgrad else None
        
        return fval_scl, std_fval_scl_out, fgrad_scl, std_fgrad_scl_out
        
    def data_init_2_scl(self, 
                        mu_in      = None, sig_in      = None, 
                        dmudx_in   = None, dsigdx_in   = None, 
                        d2mudx2_in = None, d2sigdx2_in = None):
        
        if self.b_use_data_scl:
            return self.DataScl.obj_init_2_scl(mu_in,      sig_in, 
                                               dmudx_in,   dsigdx_in, 
                                               d2mudx2_in, d2sigdx2_in)
        else:
            return mu_in, sig_in, dmudx_in, dsigdx_in, d2mudx2_in, d2sigdx2_in
    
    def data_scl_2_init(self,
                        mu_scl      = None, sig_scl      = None, 
                        dmudx_scl   = None, dsigdx_scl   = None, 
                        d2mudx2_scl = None, d2sigdx2_scl = None):
        
        if self.b_use_data_scl:
            return self.DataScl.obj_scl_2_init(mu_scl,      sig_scl, 
                                               dmudx_scl,   dsigdx_scl, 
                                               d2mudx2_scl, d2sigdx2_scl)
        else:
            return mu_scl, sig_scl, dmudx_scl, dsigdx_scl, d2mudx2_scl, d2sigdx2_scl
