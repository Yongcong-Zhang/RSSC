import numpy as np
from scipy import sparse

class EasyLM:
    """
    Simple LM optimizer with manifold-aware updates
    """
    
    def __init__(self, opt_func, update_func, **kwargs):
        """
        Initialize the LM optimizer

        Parameters:
        -----------
        opt_func: objective function returning the residual (and the Jacobian if
            SpecifyObjectiveGradient=True)
        update_func: update function applying the increment to the parameters
            (supports manifold updates)
        **kwargs: optional parameters
            MaxIteration: maximum number of iterations, default 50
            miu: initial damping factor, default 0.01
            tolX: tolerance on parameter change, default 1e-6
            tolFun: tolerance on cost function change, default 1e-6
            tolOpt: optimality tolerance, default 1e-10
            SpecifyObjectiveGradient: whether an analytical Jacobian is provided, default False
            CheckGradient: whether to check the Jacobian, default False
            ParallelNumericalDiff: whether to use parallel numerical differentiation, default False
            FixParameter: indices of parameters to fix, default None
            Debug: debug mode, default 0 (no output)
        """
        # Set default parameters
        self.params = {
            'MaxIteration': 50,
            'miu': 0.001,
            'tolX': 1e-8,
            'tolFun': 1e-8,
            'tolOpt': 1e-10,
            'SpecifyObjectiveGradient': False,
            'CheckGradient': False,
            'ParallelNumericalDiff': False,
            'FixParameter': None,
            'Debug': 0
        }
        
        # Update with user-provided parameters
        self.params.update(kwargs)
        
        # Store the functions
        self.opt_func = opt_func
        self.update_func = update_func
        
        # Fixed-parameter handling
        self.fix_id = None
        self.unfix_id = None
        
        if self.params['FixParameter'] is not None:
            self.fix_id = np.array(self.params['FixParameter'], dtype=int)
            n_params = 0  
            
    def solve(self, x):
        """
        Run the LM optimization

        Parameters:
        -----------
        x: initial parameter vector

        Returns:
        --------
        x_opt: optimized parameters
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1) 
        
        # Determine the parameter dimension and set fixed-parameter indices
        n_params = x.shape[1]
        if self.fix_id is not None:
            all_indices = np.arange(n_params)
            self.unfix_id = np.setdiff1d(all_indices, self.fix_id)
        else:
            self.unfix_id = np.arange(n_params)
        
        # Extract parameters
        miu = self.params['miu']
        tolX = self.params['tolX']
        tolFun = self.params['tolFun']
        tolOpt = self.params['tolOpt']
        max_iter = self.params['MaxIteration']
        
        # Initialize
        iter_count = 0
        nu = 2
        sqrt_eps = np.sqrt(np.finfo(float).eps)
        
        # Compute initial residuals and Jacobian
        F, J = self._calculate_F_J(x)
        sq_sum_F = np.dot(F.T, F)
        residual = sq_sum_F
        
        # If there are fixed parameters, keep only the corresponding Jacobian columns
        if self.fix_id is not None:
            J = J[:, self.unfix_id]
        
        H = J.T @ J
        JtF = J.T @ F
        
        self._disp_local(f"0 : {sq_sum_F}")
        
        # Main LM loop
        while iter_count < max_iter:
            iter_count += 1
            
            # Extract optimizable parameters
            p_ori = x[0, self.unfix_id]
            
            # Compute the update step
            n_unfix = len(self.unfix_id)
            H_LM = sparse.csr_matrix(H + sparse.diags([miu]*n_unfix, 0, format='csr'))
            dp = -sparse.linalg.spsolve(H_LM, JtF)
            
            # Build the full dx
            dx = np.zeros_like(x)
            dx[0, self.unfix_id] = dp
            
            # Use the update function (supports manifolds)
            x_LM = self.update_func(x, dx)
            
            # Compute new residuals and Jacobian
            F_LM, J_LM = self._calculate_F_J(x_LM)
            
            if self.fix_id is not None:
                J_LM = J_LM[:, self.unfix_id]
            
            sq_sum_F_LM = np.dot(F_LM.T, F_LM)
            JtF_LM = J_LM.T @ F_LM
            
            # Convergence checks
            if np.linalg.norm(dp) < tolX * (sqrt_eps + np.linalg.norm(x)):
                self._disp_local('Finished (tolX)')
                break
            
            if np.linalg.norm(JtF_LM, np.inf) < tolOpt:
                self._disp_local('Finished (tolOpt)')
                break
            
            if abs(sq_sum_F_LM - sq_sum_F) <= tolFun * (sqrt_eps + sq_sum_F):
                self._disp_local('Finished (tolFun)')
                break
            
            # Compute rho (simplified)
            varrho = -(sq_sum_F_LM - sq_sum_F)
            
            # Update strategy (simplified)
            if varrho > 0:
                miu = miu * 0.1
                x = x_LM
                sq_sum_F = sq_sum_F_LM
                residual = np.append(residual, sq_sum_F)
                H = J_LM.T @ J_LM
                JtF = JtF_LM
                self._disp_local(f"{iter_count} : total_cost={sq_sum_F[0, 0]:.4f}, fx={x[0][-5]:.4f}, fy={x[0][-4]:.4f}, cx={x[0][-3]:.4f}, cy={x[0][-2]:.4f}, gamma={x[0][-1]:.4f}")
            else:
                miu = miu * 10
        
        if iter_count >= max_iter:
            self._disp_local('Finished (max_iter)')
        
        return x_LM.flatten()
    
    def _calculate_F_J(self, x):
        """Compute residuals and Jacobian"""
        if self.params['SpecifyObjectiveGradient']:
            result = self.opt_func(x)
            if len(result) == 2:
                F, J = result
            else:
                F = result
                J = self._numerical_diff(x, F)
                
            if self.params['CheckGradient']:
                J_num = self._numerical_diff(x, F)
                self._disp_local(f"Check Gradient: {np.linalg.norm(J[:,:]- J_num[:,:])}")
        else:
            F = self.opt_func(x)
            J = self._numerical_diff(x, F)
        
        return F, J
    
    def _numerical_diff(self, x, F):
        """Compute the Jacobian by numerical differentiation"""
        n_params = x.shape[1]
        n_residuals = F.shape[0]

        J = np.zeros((n_residuals, n_params))
        eps = 1e-6
        
        for i in range(n_params):
            # Forward perturbation
            dx_p = np.zeros_like(x)
            dx_p[0, i] = eps
            x_p = self.update_func(x, dx_p)
            F_p = self.opt_func(x_p)
            
            # Backward perturbation
            dx_n = np.zeros_like(x)
            dx_n[0, i] = -eps
            x_n = self.update_func(x, dx_n)
            F_n = self.opt_func(x_n)
            
            dif = (F_p[0] - F_n[0]) / (2 * eps)
            # Central difference
            J[:, i] = dif.flatten()

        
        return J
    
    def _disp_local(self, message):
        """Print debug information"""
        if self.params['Debug'] > 0:
            print(message)