import numpy as np
from transformtion import *
from tool import *
from manifold_LM import EasyLM


def rs_residuals_jac_TE(all_x, meta, H, W, points2D_uv,  mask_TE):
    """
    Build residuals using the cumulative spline
    """
    all_pose, points3D_xyz, K, gamma = unflatten_params(all_x, meta)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    omegas, relative_poses = compute_relative_lie_algebras(all_pose)
    num_pose = len(all_pose)
    num_points = len(points3D_xyz)
    num_cameras = points2D_uv.shape[0]
    total_params = meta['gamma'] + 1
    

    # Prepare the RSSC-TE parameters
    total_residuals_TE = 0
    for i in range(num_cameras):
        valid = mask_TE[i]
        total_residuals_TE += 2 * np.sum(valid)
    jac_TE = np.zeros((total_residuals_TE, total_params))
    all_residuals_TE = np.zeros(total_residuals_TE)
    num_jac_all_TE = 0
    residual_start_idx_TE = 0


    for i in range(num_cameras):
        # Extract the valid 3D and 2D points
        valid = mask_TE[i]
        points3D_xyz_i = points3D_xyz[valid]
        points2D_uv_i = points2D_uv[i][valid]
        
        # Compute the scan time of each point
        v_i = points2D_uv_i[:, 1]  # shape (n,)
        scan_time = ((v_i / H) * gamma)  # shape (n,)
        time_ratio = scan_time
        
        # Compute the cumulative spline basis functions
        b_spline = cumulative_basis_vectorized(time_ratio)
        
        k = i + 1
        T0 = all_pose[k-1]  # (n,4,4)
        T1 = all_pose[k]    # (n,4,4)
        T2 = all_pose[k+1]  # (n,4,4)
        T3 = all_pose[k+2]  # (n,4,4)
        
        Omega_1_batch = omegas[k-1]    # (n,6)
        Omega_2_batch = omegas[k]      # (n,6)
        Omega_3_batch = omegas[k+1]    # (n,6)
        
        # B-spline weights
        S1 = b_spline[:, 1]  # (n,)
        S2 = b_spline[:, 2]  # (n,)
        S3 = b_spline[:, 3]  # (n,)
        
        a1 = S1[:, None] * Omega_1_batch  # (n,6)
        a2 = S2[:, None] * Omega_2_batch  # (n,6)
        a3 = S3[:, None] * Omega_3_batch  # (n,6)

        A1 = SE3.exp(a1)  # (n,4,4)
        A2 = SE3.exp(a2)  # (n,4,4)
        A3 = SE3.exp(a3)  # (n,4,4)
        
        Tt = np.einsum('ij,njk,nkl,nlm->nim', T0, A1, A2, A3)  # (n,4,4)
        Xi = SE3.log(Tt)  # (n,6)
        
        # Compute the projection residuals
        n_points_i = points3D_xyz_i.shape[0]
        Pw_h = np.hstack([points3D_xyz_i, np.ones((n_points_i, 1))])  # (n,4)
        Pc_h = np.einsum('nij,nj->ni', Tt, Pw_h)  # (n,4)
        
        Xc = Pc_h[:, 0]
        Yc = Pc_h[:, 1]
        Zc = Pc_h[:, 2]
        
        # Projection
        u_proj = fx * (Xc / Zc) + cx
        v_proj = fy * (Yc / Zc) + cy
        
        # Residuals
        du = u_proj - points2D_uv_i[:, 0]
        dv = v_proj - points2D_uv_i[:, 1]
        for j in range(n_points_i):
            all_residuals_TE[residual_start_idx_TE + j*2] = du[j]
            all_residuals_TE[residual_start_idx_TE + j*2 + 1] = dv[j]
        
        # ==================== Compute the Jacobian matrix ====================
        # Compute the Jacobian w.r.t. gamma
        u = time_ratio  # (n,)
        dS1_du = (1 - 2*u + u**2) / 2
        dS2_du = (1 + 2*u - 2*u**2) / 2
        dS3_du = u**2 / 2
        
        # Compute dXi/dS1, dXi/dS2, dXi/dS3
        adj_T0 = SE3.adjoint_SE3_batch(T0)  # (n,6,6)
        T0A1 = T0 @ A1  # (n,4,4)
        adj_T0A1 = SE3.adjoint_SE3_batch(T0A1)  # (n,6,6)
        T0A1A2 = np.einsum('nij,njk->nik', T0A1, A2)  # (n,4,4)
        adj_T0A1A2 = SE3.adjoint_SE3_batch(T0A1A2)  # (n,6,6)
        
        Jl_a1 = SE3.left_jacobian_SE3(a1)  # (n,6,6)
        Jl_a2 = SE3.left_jacobian_SE3(a2)  # (n,6,6)
        Jl_a3 = SE3.left_jacobian_SE3(a3)  # (n,6,6)
        Jl_Xi_inv = SE3.left_jacobian_SE3_inv(Xi)  # (n,6,6)
        

        Omega_1_batch_expanded = np.repeat(Omega_1_batch[np.newaxis, :], Jl_a1.shape[0], axis=0)  # (8, 6)
        Omega_2_batch_expanded = np.repeat(Omega_2_batch[np.newaxis, :], Jl_a2.shape[0], axis=0)  # (8, 6)
        Omega_3_batch_expanded = np.repeat(Omega_3_batch[np.newaxis, :], Jl_a3.shape[0], axis=0)  # (8, 6)
        dXi_dS1 = np.einsum('nij,jk,nk->ni', Jl_Xi_inv, adj_T0[0], 
                        np.einsum('nij,nj->ni', Jl_a1, Omega_1_batch_expanded))  # (n,6)
        dXi_dS2 = np.einsum('nij,njk,nk->ni', Jl_Xi_inv, adj_T0A1,
                        np.einsum('nij,nj->ni', Jl_a2, Omega_2_batch_expanded))  # (n,6)
        dXi_dS3 = np.einsum('nij,njk,nk->ni', Jl_Xi_inv, adj_T0A1A2,
                        np.einsum('nij,nj->ni', Jl_a3, Omega_3_batch_expanded))  # (n,6)
        
        # Compute dXi/du
        dXi_du = (dXi_dS1 * dS1_du[:, None] + dXi_dS2 * dS2_du[:, None] + dXi_dS3 * dS3_du[:, None])  # (n,6)
        
        # Compute du/dgamma
        du_dgamma = v_i / H
        
        # Compute dXi/dgamma
        dXi_dgamma = dXi_du * du_dgamma[:, None]  # (n,6)
        
        # Compute the projection Jacobian Jp
        Z2 = Zc**2
        Jp = np.zeros((n_points_i, 2, 6))
        
        Jp[:, 0, 0] = fx * Xc * Yc / Z2
        Jp[:, 0, 1] = -(fx + fx * Xc**2 / Z2)
        Jp[:, 0, 2] = fx * Yc / Zc
        Jp[:, 0, 3] = -fx / Zc
        Jp[:, 0, 4] = 0.0
        Jp[:, 0, 5] = fx * Xc / Z2
        
        Jp[:, 1, 0] = fy + fy * Yc**2 / Z2
        Jp[:, 1, 1] = -fy * Xc * Yc / Z2
        Jp[:, 1, 2] = -fy * Xc / Zc
        Jp[:, 1, 3] = 0.0
        Jp[:, 1, 4] = -fy / Zc
        Jp[:, 1, 5] = fy * Yc / Z2
        Jp = -Jp
        
        # Compute J_gamma_points (derivative of the residuals w.r.t. gamma)
        Jl = SE3.left_jacobian_SE3(Xi)  # (n,6,6)
        Jl_dxi = np.einsum('nij,nj->ni', Jl, dXi_dgamma)  # (n,6)
        J_gamma_points = np.einsum('nij,nj->ni', Jp, Jl_dxi)  # (n,2)
        
        # Compute the derivatives of T(t) w.r.t. the control poses
        # dXi/dXi0
        da1_Xi0 = -np.einsum('i,jk->ijk', S1, np.eye(6)) @ SE3.left_jacobian_SE3_inv(Omega_1_batch) @ SE3.adjoint_SE3_batch(np.linalg.inv(T0))
        dXi_Xi0 = Jl_Xi_inv + np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0 @ Jl_a1) @ da1_Xi0
        J_tmp0 = np.einsum('nij,njk->nik', Jp, Jl @ dXi_Xi0)
        dXi0_unweighted = J_tmp0.reshape(n_points_i*2, 6)
        
        # dXi/dXi1
        da1_Xi1 = - da1_Xi0
        da2_Xi1 = -np.einsum('i,jk->ijk', S2, np.eye(6)) @ SE3.left_jacobian_SE3_inv(Omega_2_batch) @ SE3.adjoint_SE3_batch(np.linalg.inv(T1))
        dXi_Xi1 = np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0 @ Jl_a1) @ da1_Xi1 + np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0A1 @ Jl_a2) @ da2_Xi1
        J_tmp1 = np.einsum('nij,njk->nik', Jp, Jl @ dXi_Xi1)
        dXi1_unweighted = J_tmp1.reshape(n_points_i*2, 6)
        
        # dXi/dXi2
        da2_Xi2 = -da2_Xi1
        da3_Xi2 = -np.einsum('i,jk->ijk', S3, np.eye(6)) @ SE3.left_jacobian_SE3_inv(Omega_3_batch) @ SE3.adjoint_SE3_batch(np.linalg.inv(T2))
        dXi_Xi2 = np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0A1 @ Jl_a2) @ da2_Xi2 + np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0A1A2 @ Jl_a3) @ da3_Xi2
        J_tmp2 = np.einsum('nij,njk->nik', Jp, Jl @ dXi_Xi2)
        dXi2_unweighted = J_tmp2.reshape(n_points_i*2, 6)
        
        # dXi/dXi3
        da3_Xi3 = -da3_Xi2
        dXi_Xi3 = np.einsum('nij,njk->nik', Jl_Xi_inv, adj_T0A1A2 @ Jl_a3) @ da3_Xi3
        J_tmp3 = np.einsum('nij,njk->nik', Jp, Jl @ dXi_Xi3)
        dXi3_unweighted = J_tmp3.reshape(n_points_i*2, 6)
        
        # Jacobian w.r.t. the 3D points
        Rt = Tt[:, :3, :3]
        de_P_unweighted = Jp[:, :, 3:6] @ Rt  # (n, 2, 3)
        
        # Intrinsics Jacobian
        X_norm = Xc / Zc
        Y_norm = Yc / Zc
        J_intrinsic_unweighted = np.zeros((n_points_i, 2, 4))
        J_intrinsic_unweighted[:, 0, 0] = X_norm
        J_intrinsic_unweighted[:, 1, 0] = 0
        J_intrinsic_unweighted[:, 0, 1] = 0
        J_intrinsic_unweighted[:, 1, 1] = Y_norm
        J_intrinsic_unweighted[:, 0, 2] = 1
        J_intrinsic_unweighted[:, 1, 2] = 0
        J_intrinsic_unweighted[:, 0, 3] = 0
        J_intrinsic_unweighted[:, 1, 3] = 1
        
        # gamma Jacobian
        J_gamma_unweighted = J_gamma_points
        
        # Fill in the Jacobian entries
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6:i*6+6] = dXi0_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+6:i*6+12] = dXi1_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+12:i*6+18] = dXi2_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+18:i*6+24] = dXi3_unweighted

        # Fill in the Jacobian for each valid 3D point
        valid_indices = np.where(valid)[0] 
        num_poses = num_pose * 6 
        point_start_idx = num_poses
        for j in range(n_points_i):
            point_idx = valid_indices[j]
            param_start = point_start_idx + point_idx * 3
            param_end = param_start + 3
            # Fill the Jacobian matrix
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, param_start:param_end] = de_P_unweighted[j]

        # Fill the intrinsics Jacobian
        for j in range(n_points_i):
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, -5:-1] = J_intrinsic_unweighted[j]

        # Fill the gamma Jacobian
        for j in range(n_points_i):
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, -1] = J_gamma_unweighted[j]

        # Update indices
        num_jac_all_TE += n_points_i * 2
        residual_start_idx_TE += n_points_i * 2
    

    return all_residuals_TE, jac_TE





def optimize_RS_masked_manifold_TE(x_init, x_if, meta, H, W, points2D_uv, mask, max_nfev=50):
    """
    Optimize using EasyLM with the analytical Jacobian
    """
    import numpy as np
    
    # 1. Prepare the objective function (returns residuals and Jacobian)
    def opt_func_with_jac(x_flat):
        x = x_flat.flatten()
        residuals, jac = rs_residuals_jac_TE(x, meta, H, W, points2D_uv,  mask)
        return residuals.reshape(-1, 1), jac
    
    # 2. Prepare the update function
    def update_func(x_current, dx):
        x = x_current.flatten().copy()
        delta = dx.flatten()
        n_params = len(x)
        
        i = 0
        while i < n_params:
            # Check if this is a pose parameter
            is_pose = False
            for start, end in meta['pose']:
                if start <= i < end:
                    is_pose = True
                    pose_start, pose_end = start, end
                    break
            
            if is_pose:
                if i == pose_start:  # Handle each pose
                    # Current pose
                    xi_current = x[pose_start:pose_end]
                    pose_delta = delta[pose_start:pose_end]
                    
                    # Manifold update
                    if np.any(np.abs(pose_delta) > 1e-12):
                        T_current = SE3.exp(xi_current)
                        delta_T = SE3.exp(pose_delta)
                        T_new = delta_T @ T_current
                        xi_new = SE3.log(T_new)
                        x[pose_start:pose_end] = xi_new
                    
                    i = pose_end
                    continue
            else:
                # Euclidean update
                x[i] += delta[i]
            
            i += 1
        
        return x.reshape(1, -1)
    
    # 3. Fixed-parameter setup
    fix_param = []
    for i, should_opt in enumerate(x_if):
        if not should_opt:
            fix_param.append(i)
    
    # 4. Optimize
    lm = EasyLM(
        opt_func=opt_func_with_jac,
        update_func=update_func,
        MaxIteration=max_nfev,
        miu=0.01,
        tolX=1e-8,
        tolFun=1e-8,
        tolOpt=1e-10,
        SpecifyObjectiveGradient=True,  
        CheckGradient = False,
        FixParameter=fix_param if fix_param else None,
        Debug=1
    )
    
    # 5. Run
    x_opt = lm.solve(x_init.reshape(1, -1))
    
    # 6. Return the result
    from scipy.optimize import OptimizeResult
    
    final_residuals, _ = opt_func_with_jac(x_opt.reshape(1, -1))
    final_cost = 0.5 * np.dot(final_residuals.T, final_residuals).item()
    
    result = OptimizeResult()
    result.x = x_opt.flatten()
    result.fun = final_residuals.flatten()
    result.cost = final_cost
    result.success = True
    result.message = "Success"
    
    return result, x_opt.flatten()

