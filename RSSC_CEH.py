import numpy as np
from transformtion import *
from tool import *
from manifold_LM import EasyLM


def rs_residuals_jac_CEH(all_x, meta, H, W, points2D_uv, points2D_back2, points2D_back1, points2D_forward1, mask, t_target):
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
        q0 = points2D_back2[idx-2, :, :]
        q1 = points2D_back1[idx-1, :, :]
        q2 = points2D_uv[idx, :, :]
        q3 = points2D_forward1[idx+1, :, :]
        t0 = gamma*(q0[:, 1]/H)
        t1 = gamma*(q1[:, 1]/H) + 1
        t2 = gamma*(q2[:, 1]/H) + 2
        t3 = gamma*(q3[:, 1]/H) + 3

        t_target = 2
        q1_gs = hermite_interp_batch(q0, q1, q2, q3,t0, t1, t2, t3, t_target)
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
        valid = mask[i+2]                     
        points3D_xyz_i = points3D_xyz[valid]        
        points2D_uv_i = points2D_uv_gs[i+2][valid]     
        Tt = all_pose[i,:,:]

        n_points_i = points3D_xyz_i.shape[0]
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

        reproj_residuals_i = np.empty(2 * n_points_i, dtype=np.float64)
        reproj_residuals_i[0::2] = du
        reproj_residuals_i[1::2] = dv

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


        # Jacobian w.r.t. 3D points
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


        q0 = points2D_back2[i, :, :][valid]      #
        q1 = points2D_back1[i+1, :, :][valid]
        q2 = points2D_uv[i+2, :, :][valid]
        q3 = points2D_forward1[i+3, :, :][valid]
        # Compute the t values and their derivatives w.r.t. gamma
        t0 = gamma * (q0[:, 1] / H)
        t1 = gamma * (q1[:, 1] / H) + 1
        t2 = gamma * (q2[:, 1] / H) + 2
        t3 = gamma * (q3[:, 1] / H) + 3

        dt0_dgamma = q0[:, 1] / H
        dt1_dgamma = q1[:, 1] / H
        dt2_dgamma = q2[:, 1] / H
        dt3_dgamma = q3[:, 1] / H

        # Interval length and its derivative w.r.t. gamma
        dt = t2 - t1                          # (n,)
        ddt_dgamma = dt2_dgamma - dt1_dgamma  # (n,)

        # Normalized parameter s and its derivative w.r.t. gamma
        s = (t_target - t1) / dt               # (n,)
        ds_dgamma = (-dt1_dgamma * dt - (t_target - t1) * ddt_dgamma) / (dt * dt + 1e-12)

        # Expand dimensions for broadcasting
        s = s[:, None]
        ds_dgamma = ds_dgamma[:, None]
        dt = dt[:, None]
        ddt_dgamma = ddt_dgamma[:, None]

        # Hermite basis functions
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2

        # Derivatives of the Hermite basis functions w.r.t. s
        dh00_ds = 6*s**2 - 6*s
        dh10_ds = 3*s**2 - 4*s + 1
        dh01_ds = -6*s**2 + 6*s
        dh11_ds = 3*s**2 - 2*s

        # Derivatives of the basis functions w.r.t. gamma
        dh00_dgamma = dh00_ds * ds_dgamma
        dh10_dgamma = dh10_ds * ds_dgamma
        dh01_dgamma = dh01_ds * ds_dgamma
        dh11_dgamma = dh11_ds * ds_dgamma

        # Denominators for the derivative computation (central differences)
        denom1 = (t2 - t0)[:, None]
        denom2 = (t3 - t1)[:, None]
        # Compute the derivatives
        m1 = (q2 - q0) / denom1
        m2 = (q3 - q1) / denom2

        # Derivatives of the denominators w.r.t. gamma
        ddenom1_dgamma = (dt2_dgamma - dt0_dgamma)[:, None]
        ddenom2_dgamma = (dt3_dgamma - dt1_dgamma)[:, None]

        # Derivatives of m1 and m2 w.r.t. gamma
        dm1_dgamma = -(q2 - q0) * ddenom1_dgamma / (denom1 * denom1 + 1e-12)
        dm2_dgamma = -(q3 - q1) * ddenom2_dgamma / (denom2 * denom2 + 1e-12)

        # Compute the derivative of q_target w.r.t. gamma
        term1 = q1 * dh00_dgamma
        term2 = dh10_dgamma * dt * m1 + h10 * ddt_dgamma * m1 + h10 * dt * dm1_dgamma
        term3 = q2 * dh01_dgamma
        term4 = dh11_dgamma * dt * m2 + h11 * ddt_dgamma * m2 + h11 * dt * dm2_dgamma

        dq_dgamma = term1 + term2 + term3 + term4
        J_gamma = -dq_dgamma

        # Fill the gamma Jacobian into the last column
        for j in range(n_points_i):
            jac[num_jac_all + j*2: num_jac_all + j*2 + 2, -1] = J_gamma[j]


        num_jac_all+=n_points_i*2

    return all_residuals, jac




def optimize_RS_masked_manifold_CEH(x_init, x_if, meta, H, W, points2D_uv, points2D_back2, points2D_back1,points2D_forward1, mask, t_target, max_nfev=50):
    """
    Optimize with EasyLM using the analytic Jacobian
    """
    import numpy as np
    
    # 1. Prepare the objective function (returns residuals and Jacobian)
    def opt_func_with_jac(x_flat):
        x = x_flat.flatten()
        residuals, jac = rs_residuals_jac_CEH(x, meta, H, W, points2D_uv, points2D_back2, points2D_back1,points2D_forward1, mask, t_target)
        return residuals.reshape(-1, 1), jac
    
    # 2. Prepare the update function (same as above)
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

