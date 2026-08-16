import numpy as np
from transformtion import *
from tool import *
from manifold_LM import EasyLM


def rs_residuals_jac_CEQ(all_x, meta, H, W, points2D_uv, points2D_back1,points2D_forward1,mask, t_target):
    """
    Compute the residuals and the Jacobian matrix
    """
    all_pose, points3D_xyz, K, gamma = unflatten_params(all_x, meta)

    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    num_pose = len(all_pose)

    num_points = len(points3D_xyz)
    num_images = points2D_uv.shape[0]


    points2D_uv_gs = []
    for idx in range(2,num_images-1):
        q0 = points2D_back1[idx-1, :, :]
        q1 = points2D_uv[idx, :, :]
        q2 = points2D_forward1[idx+1, :, :]
        t0 = gamma*(q0[:, 1]/H)
        t1 = gamma*(q1[:, 1]/H) + 1
        t2 = gamma*(q2[:, 1]/H) + 2

        q1_gs = lagrange_interp_batch(q0, q1, q2, t0, t1, t2, t_target)

        if idx == 2:
            points2D_uv_gs.append(q1_gs)
            points2D_uv_gs.append(q1_gs)
        points2D_uv_gs.append(q1_gs)
    points2D_uv_gs.append(q1_gs)


    # 1. Compute the total number of residuals
    total_residuals = 0
    for i in range(num_pose):
        valid = mask[i+2]
        total_residuals += 2 * np.sum(valid)
    
    # 2. Total number of parameters
    total_params = meta['gamma'] + 1
    # 3. Initialize the Jacobian matrix
    jac = np.zeros((total_residuals, total_params))
    num_jac_all = 0


    for i in range(num_pose):
        valid = mask[i+2]                     # bool
        points3D_xyz_i = points3D_xyz[valid]          # → (n, 3)
        points2D_uv_i = points2D_uv_gs[i+2][valid]         # → (n, 2)
        Tt = all_pose[i,:,:]

        n_points_i = points3D_xyz_i.shape[0]
        # Homogeneous coordinates
        Pw_h = np.hstack([points3D_xyz_i, np.ones((n_points_i, 1))])  # (n,4)

        # World points -> camera coordinates
        Pc_h = np.einsum('ij,nj->ni', Tt, Pw_h)
        Xc = Pc_h[:, 0]
        Yc = Pc_h[:, 1]
        Zc = Pc_h[:, 2]

        # Projection
        u_proj = fx * (Xc / Zc) + cx
        v_proj = fy * (Yc / Zc) + cy

        # Residuals
        du = u_proj - points2D_uv_i[:, 0]
        dv = v_proj - points2D_uv_i[:, 1]

        # Concatenate the x, y residuals into a single 1D vector for LM optimization
        reproj_residuals_i = np.empty(2 * n_points_i, dtype=np.float64)
        reproj_residuals_i[0::2] = du
        reproj_residuals_i[1::2] = dv
        # Append the residuals of the current image to the total residual vector
        if i == 0:
            all_residuals = reproj_residuals_i
        else:
            all_residuals = np.hstack([all_residuals, reproj_residuals_i])



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

        # Fill the pose Jacobian
        J_Xi = Jp.reshape(n_points_i*2, 6)
        jac[num_jac_all: num_jac_all+n_points_i*2, i*6 :i*6+6] = J_Xi


        # Jacobian w.r.t. 3D points (unweighted)
        Rt = Tt[:3, :3]
        de_P_unweighted = Jp[:, :, 3:6] @ Rt  # (n, 2, 3)
        # Fill the Jacobian for each valid 3D point
        valid_indices = np.where(valid)[0] 
        num_poses = num_pose * 6 
        point_start_idx = num_poses
        for j in range(n_points_i):
            point_idx = valid_indices[j]
            param_start = point_start_idx + point_idx * 3
            param_end = param_start + 3
            # Fill the Jacobian matrix
            jac[num_jac_all + j*2: num_jac_all + j*2 + 2, param_start:param_end] = de_P_unweighted[j]



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
        # Fill the intrinsic parameter Jacobian
        for j in range(n_points_i):
            jac[num_jac_all + j*2: num_jac_all + j*2 + 2, -5:-1] = J_intrinsic_unweighted[j]



        q0 = points2D_back1[i+1, :, :][valid]
        q1 = points2D_uv[i+2, :, :][valid]
        q2 = points2D_forward1[i+3, :, :][valid]

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
        
        # Compute the denominators
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
        
        dL0_dgamma = dL0_dgamma[:, None]
        dL1_dgamma = dL1_dgamma[:, None]
        dL2_dgamma = dL2_dgamma[:, None]
        
        # Compute the derivative of q_target w.r.t. gamma
        dq_dgamma = q0 * dL0_dgamma + q1 * dL1_dgamma + q2 * dL2_dgamma
        J_gamma = -dq_dgamma
        
        # Fill the gamma Jacobian into the last column
        for j in range(n_points_i):
            jac[num_jac_all + j*2: num_jac_all + j*2 + 2, -1] = J_gamma[j]


        num_jac_all+=n_points_i*2

    return all_residuals, jac




def optimize_RS_masked_manifold_CEQ(x_init, x_if, meta, H, W, points2D_uv,points2D_back1,points2D_forward1, mask, t_target, max_nfev=50):
    """
    Optimize with EasyLM using the analytic Jacobian
    """
    import numpy as np
    
    # 1. Prepare the objective function (returns residuals and Jacobian)
    def opt_func_with_jac(x_flat):
        x = x_flat.flatten()
        residuals, jac = rs_residuals_jac_CEQ(x, meta, H, W, points2D_uv,points2D_back1,points2D_forward1, mask, t_target)
        return residuals.reshape(-1, 1), jac
    
    # 2. Prepare the update function
    def update_func(x_current, dx):
        x = x_current.flatten().copy()
        delta = dx.flatten()
        n_params = len(x)
        
        i = 0
        while i < n_params:
            is_pose = False
            for start, end in meta['pose']:
                if start <= i < end:
                    is_pose = True
                    pose_start, pose_end = start, end
                    break
            
            if is_pose:
                if i == pose_start: 
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

