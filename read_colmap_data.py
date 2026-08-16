import numpy as np
from pathlib import Path
import cv2
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from scipy.spatial.transform import Rotation as R

class ColmapModel:
    def __init__(self, model_path, image_dir):
        """
        model_path: directory containing the COLMAP binary model (sparse/)
        image_dir: directory of the image folder (images/)
        """
        self.model_path = Path(model_path)
        self.image_dir = Path(image_dir)
        self.cameras = None
        self.images = None
        self.points3D = None
        self.all_image = None  

    def read_camera(self):
        """Read camera intrinsics (shared intrinsics assumed) and return the K matrix"""
        if self.cameras is None:
            self.cameras = read_cameras_binary(self.model_path / "cameras.bin")
        cam = list(self.cameras.values())[0]
        fx = fy = cam.params[0]  # fx = fy
        cx = cam.params[1]
        cy = cam.params[2]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        return K

    def read_images(self):
        """Read image poses, return all_R and all_T"""
        if self.images is None:
            self.images = read_images_binary(self.model_path / "images.bin")
        
        # Sort by image_id
        image_ids = sorted(self.images.keys())
        all_R = []
        all_T = []
        for img_id in image_ids:
            img = self.images[img_id]
            # Convert quaternion to rotation matrix
            R_mat = R.from_quat([img.qvec[1], img.qvec[2], img.qvec[3], img.qvec[0]]).as_matrix()
            t_vec = img.tvec
            all_R.append(R_mat)
            all_T.append(t_vec)
        all_R = np.array(all_R)  # shape (m, 3, 3)
        all_T = np.array(all_T)  # shape (m, 3)
        return all_R, all_T

    def read_points_and_2d(self):
        """Read 3D points, the 2D points of each image, and the mask"""
        if self.images is None:
            self.images = read_images_binary(self.model_path / "images.bin")
        if self.points3D is None:
            self.points3D = read_points3D_binary(self.model_path / "points3D.bin")

        image_ids = sorted(self.images.keys())
        point3D_ids = sorted(self.points3D.keys())
        n_points = len(point3D_ids)
        n_images = len(image_ids)

        # 3D point coordinates
        points3D_xyz = np.array([self.points3D[p].xyz for p in point3D_ids], dtype=np.float64)

        # Initialize the 2D point array and mask
        xys = np.full((n_images, n_points, 2), np.nan, dtype=np.float64)
        mask = np.zeros((n_images, n_points), dtype=bool)

        # Build a mapping from point3D_id to index
        point3D_id2idx = {pid: idx for idx, pid in enumerate(point3D_ids)}

        for i, img_id in enumerate(image_ids):
            img = self.images[img_id]
            for j, pid in enumerate(img.point3D_ids):
                if pid == -1:
                    continue
                if pid in point3D_id2idx:
                    idx = point3D_id2idx[pid]
                    xys[i, idx] = img.xys[j]
                    mask[i, idx] = True

        return points3D_xyz, xys, mask

    def read_all_images(self):
        """Read all images in the image folder and return a list of numpy arrays in the same order as images.bin"""
        if self.images is None:
            self.images = read_images_binary(self.model_path / "images.bin")

        image_ids = sorted(self.images.keys())
        all_image = []

        for img_id in image_ids:
            img_name = self.images[img_id].name
            img_path = self.image_dir / img_name
            img_cv = cv2.imread(str(img_path))  # BGR format
            if img_cv is None:
                raise FileNotFoundError(f"Image {img_path} not found!")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  
            all_image.append(img_rgb)

        self.all_image = all_image  
        return all_image
