import numpy as np
from se3_utils import SE3

def flatten_params(all_pose, points3D_xyz, K, gamma):
    all_x_list = []
    meta = {'pose': [], 'points3D_xyz': None, 'K': (), 'gamma': None}
    
    idx = 0
    N_camera = len(all_pose)
    N_points = points3D_xyz.shape[0]
    
    # ----- Flatten Pose (Lie algebra form) -----
    for i in range(N_camera):
        T_mat = np.asarray(all_pose[i])  # shape: (4, 4)

        # Compute the Lie algebra xi = [phi, rho]
        xi = SE3.log(T_mat)  # shape: (6,)
        
        start = idx
        # Lie algebra params: first 3 are rotation (phi), last 3 are translation (rho)
        all_x_list.extend(xi.tolist())
        end = idx + 6
        
        meta['pose'].append((start, end))
        idx = end
    
    # ----- Flatten 3D points -----
    points3D_xyz = np.asarray(points3D_xyz)
    start = idx
    all_x_list.extend(points3D_xyz.flatten().tolist())  # N*3
    end = idx + N_points * 3
    meta['points3D_xyz'] = (start, end)
    idx = end
    
    # ----- Flatten K -----
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    start = idx
    all_x_list.extend([fx, fy, cx, cy])
    end = idx + 4
    meta['K'] = (start, end)
    idx = end
    
    # ----- Flatten gamma -----
    meta['gamma'] = idx
    all_x_list.append(gamma)
    idx += 1

    all_x = np.array(all_x_list, dtype=np.float64)
    
    return all_x, meta


def unflatten_params(all_x, meta):
    all_pose = []
    
    # ----- Pose (recover from Lie algebra) -----
    N_pose = len(meta['pose'])
    for i in range(N_pose):
        start, end = meta['pose'][i]
        xi = all_x[start:end]  # shape: (6,)
        
        # Recover the SE(3) matrix via the exponential map
        T_mat = SE3.exp(xi)  # shape: (4, 4)

        all_pose.append(T_mat)

    all_pose = np.array(all_pose)
    
    # ----- 3D Points -----
    start, end = meta['points3D_xyz']
    pts = all_x[start:end]
    N_points = (end - start) // 3
    points3D_xyz = pts.reshape(N_points, 3)
    
    # ----- K -----
    start, end = meta['K']
    fx, fy, cx, cy = all_x[start:end]
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # ----- gamma -----
    gamma = all_x[meta['gamma']]
    
    return all_pose, points3D_xyz, K, gamma