import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix


def project_points(K, C, R, X):
    """
    Project 3D points to 2D image coordinates.
    
    Parameters
    ----------
    K : numpy.ndarray
        Camera intrinsic matrix (3 x 3)
    C : numpy.ndarray
        Camera center (3,)
    R : numpy.ndarray
        Camera rotation matrix (3 x 3)
    X : numpy.ndarray
        3D points (N x 3)
    
    Returns
    -------
    x_proj : numpy.ndarray
        Projected 2D points (N x 2)
    """
    # Convert to homogeneous coordinates
    X_hom = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Camera pose: P = K * [R | -R*C]
    P = K @ np.hstack([R, -R @ C.reshape(3, 1)])
    
    # Project points
    x_hom = (P @ X_hom.T).T
    
    # Convert back to 2D
    x_proj = x_hom[:, :2] / (x_hom[:, 2:3] + 1e-8)
    
    return x_proj


def bundle_adjustment_residuals(params, n_cameras, n_points, camera_indices, point_indices, 
                                points_2d, K, n_cam_params=6):
    """
    Compute residuals for bundle adjustment.
    
    Parameters
    ----------
    params : numpy.ndarray
        Flattened array of camera parameters and 3D points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_indices : numpy.ndarray
        Camera index for each observation
    point_indices : numpy.ndarray
        Point index for each observation
    points_2d : numpy.ndarray
        Observed 2D points (N x 2)
    K : numpy.ndarray
        Camera intrinsic matrix
    n_cam_params : int
        Number of parameters per camera (6 for rotation + translation)
    
    Returns
    -------
    residuals : numpy.ndarray
        Reprojection errors (2*N,)
    """
    # Extract camera parameters
    camera_params = params[:n_cameras * n_cam_params].reshape(n_cameras, n_cam_params)
    
    # Extract 3D points
    points_3d = params[n_cameras * n_cam_params:].reshape(n_points, 3)
    
    # Reconstruct camera poses
    C_list = []
    R_list = []
    for i in range(n_cameras):
        # Camera parameters: [rotation_vector (3), translation (3)]
        rvec = camera_params[i, :3]
        tvec = camera_params[i, 3:6]
        
        # Convert rotation vector to rotation matrix
        R = Rotation.from_rotvec(rvec).as_matrix()
        C = -R.T @ tvec
        
        C_list.append(C)
        R_list.append(R)
    
    # Compute reprojection errors
    residuals = []
    for i in range(len(camera_indices)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        C = C_list[cam_idx]
        R = R_list[cam_idx]
        X = points_3d[pt_idx]
        
        # Project point
        x_proj = project_points(K, C, R, X.reshape(1, 3))[0]
        
        # Compute error
        error = points_2d[i] - x_proj
        residuals.extend([error[0], error[1]])
    
    return np.array(residuals)


def perform_bundle_adjustment(all_world_coords, filtered_world_coords, feature_x, feature_y,
                              filtered_feature_flags, R_set, C_set, K, cam_index):
    """
    Perform bundle adjustment to optimize camera poses and 3D points.
    
    Parameters
    ----------
    all_world_coords : numpy.ndarray
        All 3D points (n_features x 3)
    filtered_world_coords : numpy.ndarray
        Flag array indicating which points are valid (n_features x 1)
    feature_x : numpy.ndarray
        X coordinates of features (n_features x n_cameras)
    feature_y : numpy.ndarray
        Y coordinates of features (n_features x n_cameras)
    filtered_feature_flags : numpy.ndarray
        Flag matrix indicating which features are visible in which cameras (n_features x n_cameras)
    R_set : list
        List of rotation matrices for each camera
    C_set : list
        List of camera centers for each camera
    K : numpy.ndarray
        Camera intrinsic matrix (3 x 3)
    cam_index : int
        Current camera index (0-based)
    
    Returns
    -------
    R_set_opt : list
        Optimized rotation matrices
    C_set_opt : list
        Optimized camera centers
    all_world_coords_opt : numpy.ndarray
        Optimized 3D points
    """
    # Find valid points (points that have been triangulated)
    valid_points_mask = (filtered_world_coords.flatten() == 1)
    valid_point_indices = np.where(valid_points_mask)[0]
    
    if len(valid_point_indices) == 0:
        return R_set, C_set, all_world_coords
    
    # Number of cameras to optimize (up to cam_index + 1)
    n_cameras = len(R_set)
    n_points = len(valid_point_indices)
    
    # Collect observations (camera-point pairs with valid features)
    camera_indices = []
    point_indices = []
    points_2d = []
    
    for pt_idx in valid_point_indices:
        for cam_idx in range(n_cameras):
            if filtered_feature_flags[pt_idx, cam_idx] == 1:
                camera_indices.append(cam_idx)
                point_indices.append(np.where(valid_point_indices == pt_idx)[0][0])
                points_2d.append([feature_x[pt_idx, cam_idx], feature_y[pt_idx, cam_idx]])
    
    if len(camera_indices) == 0:
        return R_set, C_set, all_world_coords
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    # Initialize parameters
    # Camera parameters: rotation vector (3) + translation (3) = 6 per camera
    initial_params = []
    
    # Convert camera poses to parameter vectors
    for i in range(n_cameras):
        R = np.array(R_set[i])
        C = np.array(C_set[i])
        
        # Convert rotation matrix to rotation vector
        rvec = Rotation.from_matrix(R).as_rotvec()
        
        # Translation in camera frame: t = -R * C
        tvec = -R @ C
        
        initial_params.extend([rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]])
    
    # Add 3D points
    for pt_idx in valid_point_indices:
        initial_params.extend(all_world_coords[pt_idx])
    
    initial_params = np.array(initial_params)
    
    # Perform optimization
    print(f"  Bundle adjustment: {n_cameras} cameras, {n_points} points, {len(camera_indices)} observations")
    
    try:
        result = least_squares(
            bundle_adjustment_residuals,
            initial_params,
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K),
            method='lm',  # Levenberg-Marquardt
            verbose=0,
            max_nfev=100  # Limit iterations for speed
        )
        
        optimized_params = result.x
        
        # Extract optimized camera poses
        R_set_opt = []
        C_set_opt = []
        
        for i in range(n_cameras):
            rvec = optimized_params[i*6:(i*6)+3]
            tvec = optimized_params[(i*6)+3:(i*6)+6]
            
            R_opt = Rotation.from_rotvec(rvec).as_matrix()
            C_opt = -R_opt.T @ tvec
            
            R_set_opt.append(R_opt)
            C_set_opt.append(C_opt)
        
        # Extract optimized 3D points
        all_world_coords_opt = all_world_coords.copy()
        param_offset = n_cameras * 6
        for i, pt_idx in enumerate(valid_point_indices):
            all_world_coords_opt[pt_idx] = optimized_params[param_offset + i*3:param_offset + i*3 + 3]
        
        print(f"  Bundle adjustment completed. Final cost: {result.cost:.6f}")
        
        return R_set_opt, C_set_opt, all_world_coords_opt
        
    except Exception as e:
        print(f"  Bundle adjustment failed: {e}")
        return R_set, C_set, all_world_coords




    