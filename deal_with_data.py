import os
import cv2
import numpy as np
from se3_utils import SE3

def load_images_from_folder(image_dir):
    all_images = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    all_files = sorted(os.listdir(image_dir))
    
    for filename in all_files:
        filepath = os.path.join(image_dir, filename)
        
        if os.path.isfile(filepath):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                try:
                    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        print(f"Failed to load {filename}")
                        continue

                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    all_images.append(img)
                
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
    
    print(f"Loaded {len(all_images)} images in total")
    return all_images


def extrapolate_SE3(all_pose, a, b):
    """
    Extrapolate poses using the SE(3) Lie algebra

    Args:
        all_pose: (N,4,4)
        a: number of frames to extrapolate forward
        b: number of frames to extrapolate backward

    Returns:
        all_pose_new: (N+a+b,4,4)
    """

    all_pose = np.asarray(all_pose)
    N = all_pose.shape[0]

    if N < 2:
        raise ValueError("At least two poses are required for extrapolation.")
    xis = SE3.log(all_pose)   # (N,6)
    v0 = xis[1] - xis[0]
    v1 = xis[-1] - xis[-2]
    xis_pre = [xis[0] - i * v0 for i in range(1, a+1)]
    xis_post = [xis[-1] + i * v1 for i in range(1, b+1)]
    new_xis = np.concatenate([
        np.array(xis_pre) if a > 0 else np.empty((0,6)),
        xis,
        np.array(xis_post) if b > 0 else np.empty((0,6))
    ], axis=0)
    all_pose_new = SE3.exp(new_xis)   # (N+a+b,4,4)
    return all_pose_new