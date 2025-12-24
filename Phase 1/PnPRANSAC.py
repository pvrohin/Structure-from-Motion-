import random
import numpy as np
from LinearPnP import LinearPnP
    

def PnPRANSAC(X, x, K, threshold=200, n_max=1000):
    """
    Estimates the 6-DoF camera pose with respect to a 3D object using the Perspective-n-Point (PnP) algorithm
    with Random Sample Consensus (RANSAC) to handle outliers.

    Parameters
    ----------
    X : numpy.ndarray
        a set of 3D points in the world (N x 3)
    x : numpy.ndarray
        the 2D projections of the 3D points in the image (N x 2)
    K : numpy.ndarray
        the camera intrinsic matrix (3 x 3)
    threshold : float
        threshold for inlier detection in pixels (default: 200)
    n_max : int
        maximum number of RANSAC iterations (default: 1000)

    Results
    -------
    Cnew : numpy.ndarray
        the estimated center of camera (3,)
    Rnew : numpy.ndarray
        the estimated rotation matrix (3 x 3)
    """
    # Convert to numpy arrays
    X = np.array(X)
    x = np.array(x)
    K = np.array(K)
    
    n_points = len(X)
    
    if n_points < 4:
        raise ValueError("At least 4 point correspondences are required for PnP")
    
    # Initialize best results
    best_inlier_count = 0
    best_C = None
    best_R = None
    
    # RANSAC iterations
    for iteration in range(n_max):
        # Randomly sample 4 points (minimum for PnP)
        sample_indices = random.sample(range(n_points), min(4, n_points))
        X_sample = X[sample_indices]
        x_sample = x[sample_indices]
        
        try:
            # Estimate camera pose from sample using linear PnP
            C, R = LinearPnP(X_sample, x_sample, K)
            
            # Compute reprojection error for all points
            # Project 3D points to 2D
            X_hom = np.hstack([X, np.ones((n_points, 1))])
            P = K @ np.hstack([R, -R @ C.reshape(3, 1)])
            x_proj_hom = (P @ X_hom.T).T
            x_proj = x_proj_hom[:, :2] / (x_proj_hom[:, 2:3] + 1e-8)
            
            # Compute Euclidean distance (reprojection error)
            errors = np.sqrt(np.sum((x - x_proj)**2, axis=1))
            
            # Find inliers (points with error below threshold)
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            
            # Update best result if this iteration has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_C = C.copy()
                best_R = R.copy()
                
        except Exception as e:
            # If estimation fails, skip this iteration
            continue
    
    # If no good pose was found, try with all points
    if best_C is None or best_inlier_count < 4:
        try:
            best_C, best_R = LinearPnP(X, x, K)
            print("Warning: PnP RANSAC failed, using linear PnP on all points")
        except Exception as e:
            raise ValueError(f"PnP estimation failed: {e}")
    
    return best_C, best_R
    