import numpy as np
from se3_utils import SE3
import cv2

def compute_relative_lie_algebras(all_pose):
    """
    Compute the relative Lie algebras between consecutive poses
    """

    all_pose = np.asarray(all_pose)
    N = all_pose.shape[0]

    if N < 2:
        raise ValueError("At least two poses are required to compute relative poses")
    inv_pose = np.linalg.inv(all_pose[:-1])   # (N-1,4,4)
    relative_poses = inv_pose @ all_pose[1:]   # (N-1,4,4)
    omegas = SE3.log(relative_poses)   # (N-1,6)
    # =====================================
    # Optional verification (for debugging)
    # =====================================
    recon = all_pose[:-1] @ SE3.exp(omegas)
    error = np.max(np.abs(all_pose[1:] - recon))
    if error > 1e-10:
        print(f"Warning: relative transformation reconstruction error is large: {error:.2e}")

    return np.array(omegas), relative_poses



def cumulative_basis_vectorized(t):
    """
    t: (n,) numpy array, 0 <= t <= 1
    Returns: (n, 4) numpy array, each row is the cumulative basis function vector
    """
    C = np.array([
        [6, 0, 0, 0],
        [5, 3, -3, 1],
        [1, 3, 3, -2],
        [0, 0, 0, 1]
    ], dtype=np.float64) / 6.0

    # u_vec: (n, 4) = [1, u, u^2, u^3], one point per row
    u_vec = np.stack([np.ones_like(t), t, t**2, t**3], axis=1)  # (n,4)
    result = u_vec @ C.T  # (n,4)

    return result


def find_corresponding_points(points1, mask1, image1, image2):
    """
    points1: (n,2) float, may contain NaN
    mask1: (n,) bool
    image1, image2: BGR or Gray

    return:
        points2: (n,2) float, may contain NaN
    """

    # ----------------------------
    # Step1: Convert to grayscale
    # ----------------------------
    if image1.ndim == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1.copy()
        gray2 = image2.copy()

    # ----------------------------
    # Step2: SIFT features + matching
    # ----------------------------
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        raise RuntimeError("Too few feature matches to estimate homography")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    # ----------------------------
    # Step3: Estimate homography
    # ----------------------------
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

    if H is None:
        raise RuntimeError("Homography estimation failed")

    # ----------------------------
    # Step4: Optical flow tracking
    # ----------------------------
    valid_pts1 = points1[mask1].astype(np.float32)

    valid_pts1 = valid_pts1.reshape(-1, 1, 2)

    lk_params = dict(
        winSize=(51, 51),
        maxLevel=7,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
    )

    tracked_pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, valid_pts1, None, **lk_params
    )

    tracked_pts2 = tracked_pts2.reshape(-1, 2)
    status = status.reshape(-1)

    # ----------------------------
    # Step5: Homography projection as fallback
    # ----------------------------
    proj_pts = cv2.perspectiveTransform(valid_pts1, H)
    proj_pts = proj_pts.reshape(-1, 2)

    final_valid_pts2 = np.zeros_like(tracked_pts2)

    for i in range(len(tracked_pts2)):
        if status[i] == 1:
            final_valid_pts2[i] = tracked_pts2[i]
        else:
            final_valid_pts2[i] = proj_pts[i]

    # ----------------------------
    # Step6: Assemble full points2
    # ----------------------------
    points2 = np.full_like(points1, np.nan)

    points2[mask1] = final_valid_pts2

    return points2



def filter_points_by_distance(points2D_uv, points2D_uv_rs_to_gs, mask, threshold=1.0):
    """
    Compare the distances between two sets of corresponding points and filter out bad points according to the threshold.
    """

    diff = points2D_uv - points2D_uv_rs_to_gs
    distances = np.linalg.norm(diff, axis=2)  
    mask_new = mask.copy()
    mask_new[mask & (distances > threshold)] = False
    return mask_new


def find_corresponding_points_seq(images, points2D_uv, mask, k_back=(1,), k_forward=(1,), threshold=0.1):
    """
    Find point correspondences between neighboring frames and verify them by
    tracking each point back to its original frame (round trip).
    """

    num_image = len(images)
    extra2 = {(-1, 1): 1, (-2, 2): 0, (1, 1): 3}

    def track_and_verify(offset, k):
        # --- initial tracking: frame id -> frame id + offset ---
        points = []
        for id in range(2, num_image - 1):
            points.append(find_corresponding_points(
                points2D_uv[id], mask[id],
                images[id], images[id + offset]))
            if id == 2:
                points.extend([points[-1]] * extra2[(offset, k)])
        points.extend([points[-1]] * (3 - extra2[(offset, k)]))
        tracked = np.array(points)

        # --- round trip: frame id + offset -> frame id ---
        points_back = []
        for id in range(2, num_image - 1):
            points_back.append(find_corresponding_points(
                tracked[id + offset], mask[id],
                images[id + offset], images[id]))
            if id == 2:
                points_back.extend([points_back[-1]] * 2)
        points_back.append(points_back[-1])

        mask_valid = filter_points_by_distance(
            points2D_uv, np.array(points_back), mask, threshold=threshold)
        return tracked, mask_valid

    tracked_back = {}
    for k in k_back:
        if (-k, k) not in extra2:
            raise ValueError(
                f"Unsupported backward offset k={k}; no padding configuration "
                f"in `extra2` for offset -{k}")
        tracked_back[k] = track_and_verify(-k, k)

    tracked_forward = {}
    for k in k_forward:
        if (k, k) not in extra2:
            raise ValueError(
                f"Unsupported forward offset k={k}; no padding configuration "
                f"in `extra2` for offset +{k}")
        tracked_forward[k] = track_and_verify(k, k)

    mask_combined = mask.copy()
    for _, mask_valid in list(tracked_back.values()) + list(tracked_forward.values()):
        mask_combined = mask_combined & mask_valid

    return tracked_back, tracked_forward, mask_combined


def lagrange_interp_batch(q0, q1, q2, t0, t1, t2, t_target):
    """
    q0, q1, q2 : (n,2)
    t0, t1, t2 : (n,)
    t_target   : scalar

    return: (n,2)
    """
    L0 = (t_target - t1) * (t_target - t2) / ((t0 - t1) * (t0 - t2))
    L1 = (t_target - t0) * (t_target - t2) / ((t1 - t0) * (t1 - t2))
    L2 = (t_target - t0) * (t_target - t1) / ((t2 - t0) * (t2 - t1))
    L0 = L0[:, None]
    L1 = L1[:, None]
    L2 = L2[:, None]
    q_target = q0 * L0 + q1 * L1 + q2 * L2

    return q_target


def hermite_interp_batch(q0, q1, q2, q3,
                         t0, t1, t2, t3,
                         t_target):
    """
    q0,q1,q2,q3 : (n,2)
    t0,t1,t2,t3 : (n,)
    t_target    : scalar
    return      : (n,2)
    """

    dt = (t2 - t1)                # (n,)
    s = (t_target - t1) / dt      # (n,)

    s = s[:, None]
    dt = dt[:, None]

    h00 = 2*s**3 - 3*s**2 + 1
    h10 = s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 = s**3 - s**2

    m1 = (q2 - q0) / (t2 - t0)[:, None]
    m2 = (q3 - q1) / (t3 - t1)[:, None]
    q_target = h00*q1 + h10*dt*m1 + h01*q2 + h11*dt*m2

    return q_target