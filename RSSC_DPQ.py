import numpy as np
from transformtion import *
from tool import *
from manifold_LM import EasyLM


def rs_residuals_jac_DPQ(all_x, meta, H, W, points2D_uv, points2D_back1, points2D_forward1, mask_TE, mask_DPQ, t_target):
    """
    Build the residuals with cumulative B-splines
    """
    all_pose, points3D_xyz, K, gamma = unflatten_params(all_x, meta)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    # Compute the relative Lie algebras and relative poses
    omegas, relative_poses = compute_relative_lie_algebras(all_pose)
    num_pose = len(all_pose)
    num_points = len(points3D_xyz)
    num_cameras = points2D_uv.shape[0]
    total_params = meta['gamma'] + 1
    

    # Prepare the TE parameters
    # 1. Compute the total number of TE residuals
    total_residuals_TE = 0
    for i in range(num_cameras):
        valid = mask_TE[i]
        total_residuals_TE += 2 * np.sum(valid)
    # 3. Initialize the Jacobian matrix
    jac_TE = np.zeros((total_residuals_TE, total_params))
    # 4. Initialize the residual vector
    all_residuals_TE = np.zeros(total_residuals_TE)
    num_jac_all_TE = 0
    residual_start_idx_TE = 0



    # Prepare the DPQ parameters
    points2D_uv_DPQ = []
    for idx in range(2,num_cameras-1):
        q0 = points2D_back1[idx-1, :, :]
        q1 = points2D_uv[idx, :, :]
        q2 = points2D_forward1[idx+1, :, :]
        t0 = gamma*(q0[:, 1]/H)
        t1 = gamma*(q1[:, 1]/H) + 1
        t2 = gamma*(q2[:, 1]/H) + 2
        q1_DPQ = lagrange_interp_batch(q0, q1, q2, t0, t1, t2, t_target)
        if idx == 2:
            points2D_uv_DPQ.append(q1_DPQ)
            points2D_uv_DPQ.append(q1_DPQ)
        points2D_uv_DPQ.append(q1_DPQ)
    points2D_uv_DPQ.append(q1_DPQ)

    interval = 1
    total_residuals_DPQ = 0
    for i in range(2,num_cameras-1,interval):
        valid = mask_DPQ[i]
        total_residuals_DPQ += 2 * np.sum(valid)
    all_residuals_DPQ = np.zeros(total_residuals_DPQ)
    jac_DPQ = np.zeros((total_residuals_DPQ, total_params))

    num_jac_all_DPQ = 0
    residual_start_idx_DPQ = 0
    
    for i in range(num_cameras):
        valid = mask_TE[i]
        points3D_xyz_i = points3D_xyz[valid]
        points2D_uv_i = points2D_uv[i][valid]
        
        # Compute the scan time of each point
        v_i = points2D_uv_i[:, 1]  
        scan_time = ((v_i / H) * gamma)  
        time_ratio = scan_time
        
        # Compute the cumulative basis function of the B-spline
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
        
        # Compute a = S * Omega
        a1 = S1[:, None] * Omega_1_batch  # (n,6)
        a2 = S2[:, None] * Omega_2_batch  # (n,6)
        a3 = S3[:, None] * Omega_3_batch  # (n,6)
        
        # Batched SE3 exponential map
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
        
        # Residuals (unweighted)
        du = u_proj - points2D_uv_i[:, 0]
        dv = v_proj - points2D_uv_i[:, 1]

        
        # Store the unweighted residuals
        for j in range(n_points_i):
            all_residuals_TE[residual_start_idx_TE + j*2] = du[j]
            all_residuals_TE[residual_start_idx_TE + j*2 + 1] = dv[j]
        


        # ==================== Compute weight-related quantities ====================
        # Compute the Jacobian w.r.t. gamma
        u = time_ratio  # (n,)
        dS1_du = (1 - 2*u + u**2) / 2
        dS2_du = (1 + 2*u - 2*u**2) / 2
        dS3_du = u**2 / 2
        
        # Compute dXi/dS1, dXi/dS2, dXi/dS3
        # First compute the necessary intermediate quantities
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
        
        # Jacobian w.r.t. 3D points
        Rt = Tt[:, :3, :3]
        de_P_unweighted = Jp[:, :, 3:6] @ Rt  # (n, 2, 3)
        
        # Intrinsic parameter Jacobian
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
        
        # Fill the Jacobian
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6:i*6+6] = dXi0_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+6:i*6+12] = dXi1_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+12:i*6+18] = dXi2_unweighted
        jac_TE[num_jac_all_TE: num_jac_all_TE+n_points_i*2, i*6+18:i*6+24] = dXi3_unweighted

        # Fill the Jacobian for each valid 3D point
        valid_indices = np.where(valid)[0]  
        num_poses = num_pose * 6  
        point_start_idx = num_poses
        for j in range(n_points_i):
            point_idx = valid_indices[j]
            param_start = point_start_idx + point_idx * 3
            param_end = param_start + 3
            # Fill the Jacobian matrix
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, param_start:param_end] = de_P_unweighted[j]

        # Fill the intrinsic parameter Jacobian
        for j in range(n_points_i):
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, -5:-1] = J_intrinsic_unweighted[j]

        # Fill the gamma Jacobian
        for j in range(n_points_i):
            jac_TE[num_jac_all_TE + j*2: num_jac_all_TE + j*2 + 2, -1] = J_gamma_unweighted[j]

        # Update the indices
        num_jac_all_TE += n_points_i * 2
        residual_start_idx_TE += n_points_i * 2
    

        if 1 < i < num_cameras-2 and i % interval ==0:
            valid_DPQ = mask_DPQ[i]
            points3D_xyz_i_DPQ = points3D_xyz[valid_DPQ]
            points2D_uv_i_DPQ = points2D_uv_DPQ[i][valid_DPQ]         # → (n, 2)
            # Compute the projection residuals
            n_points_i_DPQ = points3D_xyz_i_DPQ.shape[0]
            Pw_h_DPQ = np.hstack([points3D_xyz_i_DPQ, np.ones((n_points_i_DPQ, 1))])  # (n,4)

            b_spline_DPQ = cumulative_basis_vectorized(np.array([t_target-1]))
            # B-spline weights
            S1_DPQ = b_spline_DPQ[0][1]
            S2_DPQ = b_spline_DPQ[0][2] 
            S3_DPQ = b_spline_DPQ[0][3]

            a1_DPQ = S1_DPQ * Omega_1_batch 
            a2_DPQ = S2_DPQ * Omega_2_batch  
            a3_DPQ = S3_DPQ * Omega_3_batch 

            A1_DPQ = SE3.exp(a1_DPQ)  # (n,4,4)
            A2_DPQ = SE3.exp(a2_DPQ)  # (n,4,4)
            A3_DPQ = SE3.exp(a3_DPQ)  # (n,4,4)


            # Obtain the current pose
            Tt_DPQ = T0 @ A1_DPQ @ A2_DPQ @ A3_DPQ
            # World points -> camera coordinates
            Pc_h_DPQ = np.einsum('ij,nj->ni', Tt_DPQ, Pw_h_DPQ)
            Xc_DPQ = Pc_h_DPQ[:, 0]
            Yc_DPQ = Pc_h_DPQ[:, 1]
            Zc_DPQ = Pc_h_DPQ[:, 2]

            # Projection
            u_proj_DPQ = fx * (Xc_DPQ / Zc_DPQ) + cx
            v_proj_DPQ = fy * (Yc_DPQ / Zc_DPQ) + cy
            # Residuals
            du_DPQ = u_proj_DPQ - points2D_uv_i_DPQ[:, 0]
            dv_DPQ = v_proj_DPQ - points2D_uv_i_DPQ[:, 1]
            # Store the residuals
            for j in range(n_points_i_DPQ):
                all_residuals_DPQ[residual_start_idx_DPQ + j*2] = du_DPQ[j]
                all_residuals_DPQ[residual_start_idx_DPQ + j*2 + 1] = dv_DPQ[j]

            # Compute the projection Jacobian Jp
            Z2_DPQ = Zc_DPQ**2
            Jp_DPQ = np.zeros((n_points_i_DPQ, 2, 6))
            Jp_DPQ[:, 0, 0] = fx * Xc_DPQ * Yc_DPQ / Z2_DPQ
            Jp_DPQ[:, 0, 1] = -(fx + fx * Xc_DPQ**2 / Z2_DPQ)
            Jp_DPQ[:, 0, 2] = fx * Yc_DPQ / Zc_DPQ
            Jp_DPQ[:, 0, 3] = -fx / Zc_DPQ
            Jp_DPQ[:, 0, 4] = 0.0
            Jp_DPQ[:, 0, 5] = fx * Xc_DPQ / Z2_DPQ
            Jp_DPQ[:, 1, 0] = fy + fy * Yc_DPQ**2 / Z2_DPQ
            Jp_DPQ[:, 1, 1] = -fy * Xc_DPQ * Yc_DPQ / Z2_DPQ
            Jp_DPQ[:, 1, 2] = -fy * Xc_DPQ / Zc_DPQ
            Jp_DPQ[:, 1, 3] = 0.0
            Jp_DPQ[:, 1, 4] = -fy / Zc_DPQ
            Jp_DPQ[:, 1, 5] = fy * Yc_DPQ / Z2_DPQ
            Jp_DPQ = -Jp_DPQ

            Xi = SE3.log(Tt_DPQ)     #(6,)
            Jl = SE3.left_jacobian_SE3(Xi)  # (6,6)
            Jl_Xi_inv = SE3.left_jacobian_SE3_inv(Xi)  # (6,6)

            adj_T0 = SE3.adjoint_SE3_batch(T0)  # (6,6)
            T0A1 = T0 @ A1_DPQ  # (n,4,4)
            adj_T0A1 = SE3.adjoint_SE3_batch(T0A1)  # (6,6)
            T0A1A2 =  T0A1@A2_DPQ  # (n,4,4)
            adj_T0A1A2 = SE3.adjoint_SE3(T0A1A2)  # (6,6)

            Jl_a1 = SE3.left_jacobian_SE3(a1_DPQ)  # (6,6)
            Jl_a2 = SE3.left_jacobian_SE3(a2_DPQ)  # (6,6)
            Jl_a3 = SE3.left_jacobian_SE3(a3_DPQ)  # (6,6)


            # Compute the derivatives of T(t) w.r.t. the control poses
            # dXi/dXi0
            da1_Xi0 = -S1_DPQ * SE3.left_jacobian_SE3_inv(Omega_1_batch) @ SE3.adjoint_SE3(np.linalg.inv(T0))   # (6,6)
            dXi_Xi0 = Jl_Xi_inv +  Jl_Xi_inv@ adj_T0 @ Jl_a1 @ da1_Xi0
            J_tmp0 = np.einsum('nij,njk->nik', Jp_DPQ, Jl @ dXi_Xi0)
            dXi0_unweighted = J_tmp0.reshape(n_points_i_DPQ*2, 6)

            # dXi/dXi1
            da1_Xi1 = - da1_Xi0
            da2_Xi1 = -S2_DPQ * SE3.left_jacobian_SE3_inv(Omega_2_batch) @ SE3.adjoint_SE3(np.linalg.inv(T1))
            dXi_Xi1 = Jl_Xi_inv@ adj_T0 @ Jl_a1 @ da1_Xi1 +  Jl_Xi_inv @ adj_T0A1 @ Jl_a2 @ da2_Xi1
            J_tmp1 = np.einsum('nij,njk->nik', Jp_DPQ, Jl @ dXi_Xi1)
            dXi1_unweighted = J_tmp1.reshape(n_points_i_DPQ*2, 6)

            # dXi/dXi2
            da2_Xi2 = -da2_Xi1
            da3_Xi2 = -S3_DPQ * SE3.left_jacobian_SE3_inv(Omega_3_batch) @ SE3.adjoint_SE3(np.linalg.inv(T2))
            dXi_Xi2 =  Jl_Xi_inv @ adj_T0A1 @ Jl_a2 @ da2_Xi2 +  Jl_Xi_inv @ adj_T0A1A2 @ Jl_a3 @ da3_Xi2
            J_tmp2 = np.einsum('nij,njk->nik', Jp_DPQ, Jl @ dXi_Xi2)
            dXi2_unweighted = J_tmp2.reshape(n_points_i_DPQ*2, 6)

            # dXi/dXi3
            da3_Xi3 = -da3_Xi2
            dXi_Xi3 = Jl_Xi_inv @ adj_T0A1A2 @ Jl_a3 @ da3_Xi3
            J_tmp3 = Jp_DPQ @ Jl @ dXi_Xi3
            dXi3_unweighted = J_tmp3.reshape(n_points_i_DPQ*2, 6)

            # Fill the Jacobian
            jac_DPQ[num_jac_all_DPQ: num_jac_all_DPQ+n_points_i_DPQ*2, i*6:i*6+6] = dXi0_unweighted
            jac_DPQ[num_jac_all_DPQ: num_jac_all_DPQ+n_points_i_DPQ*2, i*6+6:i*6+12] = dXi1_unweighted
            jac_DPQ[num_jac_all_DPQ: num_jac_all_DPQ+n_points_i_DPQ*2, i*6+12:i*6+18] = dXi2_unweighted
            jac_DPQ[num_jac_all_DPQ: num_jac_all_DPQ+n_points_i_DPQ*2, i*6+18:i*6+24] = dXi3_unweighted

            # Jacobian w.r.t. 3D points
            Rt_DPQ = Tt_DPQ[:3, :3]
            de_P_DPQ = Jp_DPQ[:, :, 3:6] @ Rt_DPQ  # (n, 2, 3)
            # Fill the Jacobian for each valid 3D point
            valid_indices = np.where(valid_DPQ)[0] 
            num_poses = num_pose * 6  
            point_start_idx = num_poses
            for j in range(n_points_i_DPQ):
                point_idx = valid_indices[j]
                param_start = point_start_idx + point_idx * 3
                param_end = param_start + 3
                # Fill the Jacobian matrix
                jac_DPQ[num_jac_all_DPQ + j*2: num_jac_all_DPQ + j*2 + 2, param_start:param_end] = de_P_DPQ[j]


            # Intrinsic parameter Jacobian
            X_norm = Xc_DPQ / Zc_DPQ
            Y_norm = Yc_DPQ / Zc_DPQ
            J_intrinsic_unweighted = np.zeros((n_points_i_DPQ, 2, 4))
            J_intrinsic_unweighted[:, 0, 0] = X_norm
            J_intrinsic_unweighted[:, 1, 0] = 0
            J_intrinsic_unweighted[:, 0, 1] = 0
            J_intrinsic_unweighted[:, 1, 1] = Y_norm
            J_intrinsic_unweighted[:, 0, 2] = 1
            J_intrinsic_unweighted[:, 1, 2] = 0
            J_intrinsic_unweighted[:, 0, 3] = 0
            J_intrinsic_unweighted[:, 1, 3] = 1
            # Fill the intrinsic parameter Jacobian
            for j in range(n_points_i_DPQ):
                jac_DPQ[num_jac_all_DPQ + j*2: num_jac_all_DPQ + j*2 + 2, -5:-1] = J_intrinsic_unweighted[j]


            q0 = points2D_back1[i-1, :, :][valid_DPQ] 
            q1 = points2D_uv[i, :, :][valid_DPQ] 
            q2 = points2D_forward1[i+1, :, :][valid_DPQ] 

            # Compute the t values and their derivatives w.r.t. gamma
            t0 = gamma * (q0[:, 1] / H)
            t1 = gamma * (q1[:, 1] / H) + 1
            t2 = gamma * (q2[:, 1] / H) + 2
            
            dt0_dgamma = q0[:, 1] / H
            dt1_dgamma = q1[:, 1] / H
            dt2_dgamma = q2[:, 1] / H
            
            # Compute the Lagrange basis functions
            L0 = (t_target - t1) * (t_target - t2) / ((t0 - t1) * (t0 - t2))
            L1 = (t_target - t0) * (t_target - t2) / ((t1 - t0) * (t1 - t2))
            L2 = (t_target - t0) * (t_target - t1) / ((t2 - t0) * (t2 - t1))
            
            D0 = (t0 - t1) * (t0 - t2)
            D1 = (t1 - t0) * (t1 - t2)
            D2 = (t2 - t0) * (t2 - t1)
            
            # Compute the derivatives of the numerators w.r.t. gamma
            dN0_dgamma = -(t_target - t2) * dt1_dgamma - (t_target - t1) * dt2_dgamma
            dN1_dgamma = -(t_target - t2) * dt0_dgamma - (t_target - t0) * dt2_dgamma
            dN2_dgamma = -(t_target - t1) * dt0_dgamma - (t_target - t0) * dt1_dgamma
            
            # Compute the derivatives of the denominators w.r.t. gamma
            dD0_dgamma = (dt0_dgamma - dt1_dgamma) * (t0 - t2) + (t0 - t1) * (dt0_dgamma - dt2_dgamma)
            dD1_dgamma = (dt1_dgamma - dt0_dgamma) * (t1 - t2) + (t1 - t0) * (dt1_dgamma - dt2_dgamma)
            dD2_dgamma = (dt2_dgamma - dt0_dgamma) * (t2 - t1) + (t2 - t0) * (dt2_dgamma - dt1_dgamma)
            
            # Compute the derivatives of the basis functions w.r.t. gamma
            dL0_dgamma = (dN0_dgamma * D0 - L0 * D0 * dD0_dgamma) / (D0 * D0 + 1e-12)
            dL1_dgamma = (dN1_dgamma * D1 - L1 * D1 * dD1_dgamma) / (D1 * D1 + 1e-12)
            dL2_dgamma = (dN2_dgamma * D2 - L2 * D2 * dD2_dgamma) / (D2 * D2 + 1e-12)
            
            # Expand dimensions for broadcasting
            dL0_dgamma = dL0_dgamma[:, None]
            dL1_dgamma = dL1_dgamma[:, None]
            dL2_dgamma = dL2_dgamma[:, None]
            
            # Compute the derivative of q_target w.r.t. gamma
            dq_dgamma = q0 * dL0_dgamma + q1 * dL1_dgamma + q2 * dL2_dgamma
            J_gamma = -dq_dgamma
            # Fill the gamma Jacobian into the last column
            for j in range(n_points_i_DPQ):
                jac_DPQ[num_jac_all_DPQ + j*2: num_jac_all_DPQ + j*2 + 2, -1] = J_gamma[j]


            num_jac_all_DPQ += n_points_i_DPQ * 2
            residual_start_idx_DPQ += n_points_i_DPQ * 2

    scale = 0.5
    all_residuals = np.concatenate([all_residuals_TE, all_residuals_DPQ*scale])
    jac = np.concatenate((jac_TE, jac_DPQ*scale))

    return all_residuals, jac





def optimize_RS_masked_manifold_DPQ(x_init, x_if, meta, H, W, points2D_uv, points2D_back1, points2D_forward1, mask_TE, mask_DPQ, t_target, max_nfev=50):
    """
    Optimize with EasyLM using the analytic Jacobian
    """
    import numpy as np
    
    # 1. Prepare the objective function (returns residuals and Jacobian)
    def opt_func_with_jac(x_flat):
        x = x_flat.flatten()
        residuals, jac = rs_residuals_jac_DPQ(x, meta, H, W, points2D_uv, points2D_back1, points2D_forward1, mask_TE, mask_DPQ, t_target)
        return residuals.reshape(-1, 1), jac
    
    # 2. Prepare the update function
    def update_func(x_current, dx):
        x = x_current.flatten().copy()
        delta = dx.flatten()
        n_params = len(x)
        
        i = 0
        while i < n_params:
            # Check whether this is a pose parameter
            is_pose = False
            for start, end in meta['pose']:
                if start <= i < end:
                    is_pose = True
                    pose_start, pose_end = start, end
                    break
            
            if is_pose:
                if i == pose_start:  # handle each pose
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
    
    # 3. Fixed parameter setup
    fix_param = []
    for i, should_opt in enumerate(x_if):
        if not should_opt:
            fix_param.append(i)
    
    # 4. Optimization
    lm = EasyLM(
        opt_func=opt_func_with_jac,
        update_func=update_func,
        MaxIteration=max_nfev,
        miu=0.01,
        tolX=1e-8,
        tolFun=1e-8,
        tolOpt=1e-10,
        SpecifyObjectiveGradient=True,  # use the analytic Jacobian
        CheckGradient = False,
        FixParameter=fix_param if fix_param else None,
        Debug=1
    )
    
    # 5. Run
    x_opt = lm.solve(x_init.reshape(1, -1))
    
    # 6. Return results
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

