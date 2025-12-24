import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

def NonLinearPnPLoss(X0, X, x, K):
    """
    The loss function for optimization in Non-Linear PnP. 

    Parameters
    ----------
    X0 : numpy.ndarray
        initial values of parameters for optimization (6 parameters: rotation vector (3) + translation (3))
    X : numpy.ndarray
        a set of 3D points (N x 3)
    x : numpy.ndarray
        a set of projections of these 3D points (N x 2)
    K : numpy.ndarray
        camera intrinsic matrix (3 x 3)

    Results
    -------
    error : numpy.ndarray
        the error for optimization in Non-Linear PnP (2*N array of reprojection errors)
    """
    # Extract rotation vector and translation from X0
    rvec = X0[:3]
    tvec = X0[3:6]
    
    # Convert rotation vector to rotation matrix
    R = Rotation.from_rotvec(rvec).as_matrix()
    
    # Compute camera center: C = -R^T * t
    C = -R.T @ tvec
    
    # Project 3D points to 2D
    X_hom = np.hstack([X, np.ones((X.shape[0], 1))])
    P = K @ np.hstack([R, -R @ C.reshape(3, 1)])
    x_proj_hom = (P @ X_hom.T).T
    x_proj = x_proj_hom[:, :2] / (x_proj_hom[:, 2:3] + 1e-8)
    
    # Compute reprojection error
    error = (x - x_proj).flatten()
    
    return error


def NonLinearPnP(X, x, K, C, R):
    """
    Non-linear Perspective-n-Point (PnP) algorithm for solving the pose estimation problem 
    using a set of 3D object points and their corresponding 2D image points.
    Refines camera pose using non-linear optimization.

    Parameters
    ----------
    X : numpy.ndarray
        a set of 3D points (N x 3)
    x : numpy.ndarray
        a set of projections of these 3D points (N x 2)
    K : numpy.ndarray
        camera intrinsic matrix (3 x 3)
    C : numpy.ndarray
        the center of camera (3,) - initial estimate
    R : numpy.ndarray
        the rotation matrix (3 x 3) - initial estimate
    
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
    C = np.array(C)
    R = np.array(R)
    
    n_points = len(X)
    
    if n_points < 4:
        # Not enough points, return initial estimate
        return C, R
    
    # Convert initial rotation matrix to rotation vector
    rvec = Rotation.from_matrix(R).as_rotvec()
    
    # Convert camera center to translation in camera frame: t = -R * C
    tvec = -R @ C
    
    # Initial parameters: [rotation_vector (3), translation (3)]
    X0 = np.hstack([rvec, tvec])
    
    # Perform non-linear optimization
    try:
        result = least_squares(
            NonLinearPnPLoss,
            X0,
            args=(X, x, K),
            method='lm',  # Levenberg-Marquardt
            verbose=0,
            max_nfev=100  # Limit iterations
        )
        
        optimized_params = result.x
        
        # Extract optimized rotation and translation
        rvec_opt = optimized_params[:3]
        tvec_opt = optimized_params[3:6]
        
        # Convert to rotation matrix
        Rnew = Rotation.from_rotvec(rvec_opt).as_matrix()
        
        # Compute camera center: C = -R^T * t
        Cnew = -Rnew.T @ tvec_opt
        
        return Cnew, Rnew
        
    except Exception as e:
        # If optimization fails, return initial estimate
        print(f"Non-linear PnP optimization failed: {e}")
        return C, R


# Alias for lowercase function name (for compatibility with wrapper)
def nonlinear_PnP(K, C, R, x, X):
    """
    Alias for NonLinearPnP with different parameter order to match wrapper usage.
    
    Parameters
    ----------
    K : numpy.ndarray
        camera intrinsic matrix
    C : numpy.ndarray
        the center of camera (initial estimate)
    R : numpy.ndarray
        the rotation matrix (initial estimate)
    x : numpy.ndarray
        a set of projections of 3D points
    X : numpy.ndarray
        a set of 3D points
    
    Returns
    -------
    Cnew : numpy.ndarray
        the estimated center of camera
    Rnew : numpy.ndarray
        the estimated rotation matrix
    """
    return NonLinearPnP(X, x, K, C, R)
    