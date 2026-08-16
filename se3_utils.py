import numpy as np
from scipy.spatial.transform import Rotation


class SE3:
    """
    SE(3) utility class (left perturbation / left-trivialized)
    Lie algebra: xi = [phi, rho]
    Transformation:
        R = exp([phi]_x)
        t = J(phi) * rho
    """

    eps = 1e-8

    @staticmethod
    def skew(v):
        v = np.asarray(v)
        single = (v.ndim == 1)
        if single:
            v = v[None]
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        O = np.zeros_like(vx)

        result = np.stack([
            np.stack([O,   -vz,  vy], axis=-1),
            np.stack([vz,   O,  -vx], axis=-1),
            np.stack([-vy,  vx,   O], axis=-1)
        ], axis=-2)

        return result[0] if single else result

    @staticmethod
    def left_jacobian_SO3(phi):
        phi = np.asarray(phi)
        single = (phi.ndim == 1)
        if single:
            phi = phi[None]

        theta = np.linalg.norm(phi, axis=1)
        J = np.zeros((phi.shape[0], 3, 3))
        I = np.eye(3)

        small = theta < SE3.eps
        large = ~small


        if np.any(small):
            phi_s = phi[small]
            A = SE3.skew(phi_s)
            J[small] = I + 0.5 * A
            # J[small] = I + 0.5*A + (1.0/6.0)*(A@A)


        if np.any(large):
            phi_l = phi[large]
            theta_l = theta[large]

            a = phi_l / theta_l[:, None]
            a_hat = SE3.skew(a)

            sin_t = np.sin(theta_l)
            cos_t = np.cos(theta_l)

            A = sin_t / theta_l
            B = (1 - cos_t) / theta_l

            J[large] = (
                A[:, None, None] * I +
                (1 - A)[:, None, None] * np.einsum("bi,bj->bij", a, a) +
                B[:, None, None] * a_hat
            )

        return J[0] if single else J

    @staticmethod
    def left_jacobian_SO3_inv(phi, eps=1e-8):
        phi = np.asarray(phi)
        single = (phi.ndim == 1)
        if single:
            phi = phi[None]

        theta = np.linalg.norm(phi, axis=1)
        J_inv = np.zeros((phi.shape[0], 3, 3))
        I = np.eye(3)

        small = theta < eps
        large = ~small

        if np.any(small):
            phi_s = phi[small]
            A = SE3.skew(phi_s)
            J_inv[small] = I - 0.5 * A 
            #J_inv[small] = I - 0.5 * A + (1.0 / 12.0) * (A @ A)

        if np.any(large):
            phi_l = phi[large]
            theta_l = theta[large]

            a = phi_l / theta_l[:, None]
            a_hat = SE3.skew(a)

            half = 0.5 * theta_l
            cot_half = 1.0 / np.tan(half)

            J_inv[large] = (
                (half * cot_half)[:, None, None] * I +
                (1 - half * cot_half)[:, None, None]
                * np.einsum("bi,bj->bij", a, a) -
                half[:, None, None] * a_hat
            )

        return J_inv[0] if single else J_inv


    @staticmethod
    def left_jacobian_SE3(xi, eps=1e-8):
        xi = np.asarray(xi)
        single = (xi.ndim == 1)
        if single:
            xi = xi[None, :]  # convert to (1, 6)
        
        n = xi.shape[0]
        
        # Split into rotation and translation parts
        phi = xi[:, :3]  # (n, 3) rotation vector
        rho = xi[:, 3:]  # (n, 3) translation vector
        
        J_so3 = SE3.left_jacobian_SO3(phi)  # (n, 3, 3)
        
        theta = np.linalg.norm(phi, axis=1)  # (n,)
        
        J = np.zeros((n, 6, 6))
        J[:, :3, :3] = J_so3
        J[:, 3:, 3:] = J_so3
        
        small_mask = theta < eps
        large_mask = ~small_mask
        
        Q = np.zeros((n, 3, 3))
        
        if np.any(small_mask):
            phi_s = phi[small_mask]
            rho_s = rho[small_mask]
            n_small = phi_s.shape[0]
            
            A_s = SE3.skew(phi_s) 
            B_s = SE3.skew(rho_s)  
            
            Q_small = (
                0.5 * B_s 
                + (1.0 / 6.0) * (np.einsum('nij,njk->nik', A_s, B_s) + np.einsum('nij,njk->nik', B_s, A_s))
                + (1.0 / 24.0) * np.einsum('nij,njk,nkl->nil', A_s, B_s, A_s)
            )
            
            Q[small_mask] = Q_small
        
        if np.any(large_mask):
            phi_l = phi[large_mask]
            rho_l = rho[large_mask]
            theta_l = theta[large_mask]
            n_large = phi_l.shape[0]
            
            A_l = SE3.skew(phi_l) 
            B_l = SE3.skew(rho_l)  
            
            theta_l2 = theta_l * theta_l
            sin_t = np.sin(theta_l)
            cos_t = np.cos(theta_l)
            
            a = (theta_l - sin_t) / (theta_l2 * theta_l)  
            b = (1 - 0.5 * theta_l2 - cos_t) / (theta_l2 * theta_l2)  
            
            AB = np.einsum('nij,njk->nik', A_l, B_l)  # A @ B
            BA = np.einsum('nij,njk->nik', B_l, A_l)  # B @ A
            ABA = np.einsum('nij,njk,nkl->nil', A_l, B_l, A_l)  # A @ B @ A
            AAB = np.einsum('nij,njk,nkl->nil', A_l, A_l, B_l)  # A @ A @ B
            BAA = np.einsum('nij,njk,nkl->nil', B_l, A_l, A_l)  # B @ A @ A
            
            term1 = 0.5 * B_l
            term2 = a[:, None, None] * (AB + BA + ABA)
            term3 = -b[:, None, None] * (AAB + BAA - 3 * ABA)
            
            Q_large = term1 + term2 + term3
            Q[large_mask] = Q_large
        
        J[:, 3:, :3] = Q
        
        return J[0] if single else J



    @staticmethod
    def left_jacobian_SE3_inv(xi, eps=1e-8):
        xi = np.asarray(xi)
        single = (xi.ndim == 1)
        if single:
            xi = xi[None]
        
        n = xi.shape[0]
        phi = xi[:, :3]  # (n, 3)
        rho = xi[:, 3:]  # (n, 3)
        

        theta = np.linalg.norm(phi, axis=1)  # (n,)
        J_so3_inv = SE3.left_jacobian_SO3_inv(phi, eps)  # (n, 3, 3)
        J_inv = np.zeros((n, 6, 6))
        
        small_mask = theta < eps
        large_mask = ~small_mask
        

        Q = np.zeros((n, 3, 3))
        

        if np.any(small_mask):

            phi_small = phi[small_mask]  # (m, 3)
            rho_small = rho[small_mask]  # (m, 3)
            
            # Compute A and B in batch
            A_small = SE3.skew(phi_small)  # (m, 3, 3)
            B_small = SE3.skew(rho_small)  # (m, 3, 3)
            
            # Q = 0.5 * B + (1.0 / 6.0) * (A @ B + B @ A) + (1.0 / 24.0) * (A @ B @ A)
            

            term1 = 0.5 * B_small
            AB_small = np.einsum('bij,bjk->bik', A_small, B_small)
            BA_small = np.einsum('bij,bjk->bik', B_small, A_small)
            term2 = (1.0 / 6.0) * (AB_small + BA_small)
            # A @ B @ A
            ABA_small = np.einsum('bij,bjk,bkl->bil', A_small, B_small, A_small)
            term3 = (1.0 / 24.0) * ABA_small
            
            Q_small = term1 + term2 + term3
            Q[small_mask] = Q_small
        
        if np.any(large_mask):
            phi_large = phi[large_mask]  # (k, 3)
            rho_large = rho[large_mask]  # (k, 3)
            theta_large = theta[large_mask]  # (k,)
            

            A_large = SE3.skew(phi_large)  # (k, 3, 3)
            B_large = SE3.skew(rho_large)  # (k, 3, 3)
            theta2_large = theta_large * theta_large
            theta3_large = theta2_large * theta_large
            theta4_large = theta2_large * theta2_large
            sin_theta = np.sin(theta_large)
            cos_theta = np.cos(theta_large)
            

            a = (theta_large - sin_theta) / theta3_large
            b = (1.0 - 0.5 * theta2_large - cos_theta) / theta4_large
            
            # First term: 0.5 * B
            term1 = 0.5 * B_large
            
            # Second term: a * (A @ B + B @ A + A @ B @ A)
            # A @ B
            AB_large = np.einsum('bij,bjk->bik', A_large, B_large)
            # B @ A
            BA_large = np.einsum('bij,bjk->bik', B_large, A_large)
            # A @ B @ A
            ABA_large = np.einsum('bij,bjk,bkl->bil', A_large, B_large, A_large)
            
            term2 = a[:, None, None] * (AB_large + BA_large + ABA_large)
            
            # Third term: -b * (A @ A @ B + B @ A @ A - 3 * A @ B @ A)
            # A @ A @ B
            AAB_large = np.einsum('bij,bjk,bkl->bil', A_large, A_large, B_large)
            # B @ A @ A
            BAA_large = np.einsum('bij,bjk,bkl->bil', B_large, A_large, A_large)
            
            term3 = -b[:, None, None] * (AAB_large + BAA_large - 3 * ABA_large)
            
            Q_large = term1 + term2 + term3
            Q[large_mask] = Q_large
        

        J_inv[:, :3, :3] = J_so3_inv
        J_inv[:, 3:, 3:] = J_so3_inv
        Q_J_so3_inv = np.einsum('bij,bjk->bik', Q, J_so3_inv)
        bottom_left = -np.einsum('bij,bjk->bik', J_so3_inv, Q_J_so3_inv)
        J_inv[:, 3:, :3] = bottom_left
        
        return J_inv[0] if single else J_inv

    @staticmethod
    def right_jacobian_SO3(phi):
        return SE3.left_jacobian_SO3(-phi)

    @staticmethod  
    def right_jacobian_SO3_inv(phi):
        return SE3.left_jacobian_SO3_inv(-phi)

    @staticmethod
    def right_jacobian_SE3(xi):
        return SE3.left_jacobian_SE3(-xi)

    @staticmethod
    def right_jacobian_SE3_inv(xi):
        return SE3.left_jacobian_SE3_inv(-xi)

    # ------------------------
    # SE(3) exp / log
    # ------------------------

    @staticmethod
    def exp(xi):
        xi = np.asarray(xi)
        single = (xi.ndim == 1)
        if single:
            xi = xi[None]

        phi = xi[:, :3]
        rho = xi[:, 3:]

        R = Rotation.from_rotvec(phi).as_matrix()
        J = SE3.left_jacobian_SO3(phi)
        t = np.einsum("bij,bj->bi", J, rho)

        T = np.zeros((xi.shape[0], 4, 4))
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1.0

        return T[0] if single else T

    @staticmethod
    def log(T):
        T = np.asarray(T)
        single = (T.ndim == 2)
        if single:
            T = T[None]

        R = T[:, :3, :3]
        t = T[:, :3, 3]

        phi = Rotation.from_matrix(R).as_rotvec()
        J_inv = SE3.left_jacobian_SO3_inv(phi)
        rho = np.einsum("bij,bj->bi", J_inv, t)

        xi = np.concatenate([phi, rho], axis=1)
        return xi[0] if single else xi



    def skew_single(w):
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])

    def skew_batch(w):
        n = w.shape[0]
        W = np.zeros((n,3,3))
        W[:,0,1] = -w[:,2]
        W[:,0,2] =  w[:,1]
        W[:,1,0] =  w[:,2]
        W[:,1,2] = -w[:,0]
        W[:,2,0] = -w[:,1]
        W[:,2,1] =  w[:,0]
        return W


    @staticmethod
    def adjoint_SE3(T):
        R = T[:3, :3]   # 3x3 rotation matrix
        t = T[:3, 3]    # 3x1 translation vector

        t_wedge = SE3.skew(t)  # skew-symmetric matrix t^

        # Build the 6x6 adjoint matrix
        Ad = np.zeros((6, 6))
        Ad[:3, :3] = R
        Ad[:3, 3:] = 0  
        Ad[3:, :3] = t_wedge @ R 
        Ad[3:, 3:] = R
        
        return Ad

    @staticmethod
    def adjoint_SE3_batch(T_batch):
        # Ensure the input is a 3D array (n, 4, 4)
        if T_batch.ndim == 2:
            T_batch = T_batch[np.newaxis, ...]
        
        n = T_batch.shape[0]
        
        # Extract the rotation and translation parts
        R = T_batch[:, :3, :3]  # (n, 3, 3)
        t = T_batch[:, :3, 3]   # (n, 3)
        
        t_wedge = SE3.skew_batch(t)  # (n, 3, 3)
        Ad_batch = np.zeros((n, 6, 6))
        Ad_batch[:, :3, :3] = R
        Ad_batch[:, 3:, :3] = np.einsum('nij,njk->nik', t_wedge, R)
        Ad_batch[:, 3:, 3:] = R
        
        return Ad_batch