import numpy as np
from scipy.optimize import least_squares


def Loss(X, x1, x2, P1, P2):
    """
    Loss function for optimization in non-linear triangulation.
    Computes reprojection error for a single 3D point.

    Parameters
    ----------
    X : array-like
        3D point (3,)
    x1 : array-like
        projection of a point in first image (2,)
    x2 : array-like
        projection of a point in the second image (2,)
    P1 : array-like
        first projection matrix (3 x 4)
    P2 : array-like
        second projection matrix (3 x 4)

    Results
    -------
    error : numpy.ndarray
        the error array calculated for optimization (4,) - reprojection errors
    """
    # Convert 3D point to homogeneous coordinates
    X_hom = np.array([X[0], X[1], X[2], 1])
    
    # Project to first image
    x1_proj_hom = P1 @ X_hom
    if abs(x1_proj_hom[2]) < 1e-8:
        x1_proj = np.array([x1[0], x1[1]])  # Fallback
    else:
        x1_proj = x1_proj_hom[:2] / x1_proj_hom[2]
    
    # Project to second image
    x2_proj_hom = P2 @ X_hom
    if abs(x2_proj_hom[2]) < 1e-8:
        x2_proj = np.array([x2[0], x2[1]])  # Fallback
    else:
        x2_proj = x2_proj_hom[:2] / x2_proj_hom[2]
    
    # Compute reprojection errors
    error1 = x1 - x1_proj
    error2 = x2 - x2_proj
    
    # Return flattened error array
    return np.hstack([error1, error2])


def NonLinearTriangulation(K, C1, R1, C2, R2, x1, x2, x0):
    """
    Computes the 3D position of a set of points given its projections in two images using
    non-linear triangulation. Refines 3D points using non-linear optimization.

    Parameters
    ----------
    K : array-like
        camera intrinsic matrix (3 x 3)
    C1 : array-like
        center of first camera (3,)
    R1 : array-like
        rotation matrix of first camera (3 x 3)
    C2 : array-like
        center of second camera (3,)
    R2 : array-like
        rotation matrix of second camera (3 x 3)
    x1 : array-like
        projections of a set of points in first image (N x 2)
    x2 : array-like
        projections of a set of points in second image (N x 2)
    x0 : array-like
        initial estimate of 3D points from linear triangulation (N x 3)
    
    Results
    -------
    X : array-like
        refined 3D points (N x 3)
    """
    # Convert to numpy arrays
    K = np.array(K)
    C1 = np.array(C1)
    R1 = np.array(R1)
    C2 = np.array(C2)
    R2 = np.array(R2)
    x1 = np.array(x1)
    x2 = np.array(x2)
    x0 = np.array(x0)
    
    n_points = len(x1)
    
    # Build projection matrices
    # P = K * [R | -R*C]
    t1 = -R1 @ C1.reshape(3, 1)
    P1 = K @ np.hstack([R1, t1])
    
    t2 = -R2 @ C2.reshape(3, 1)
    P2 = K @ np.hstack([R2, t2])
    
    # Refine each 3D point using non-linear optimization
    X_refined = []
    
    for i in range(n_points):
        X_init = x0[i]  # Initial estimate from linear triangulation
        x1_i = x1[i]
        x2_i = x2[i]
        
        # Define loss function for this point
        def point_loss(X_3d):
            return Loss(X_3d, x1_i, x2_i, P1, P2)
        
        # Perform non-linear optimization
        try:
            result = least_squares(
                point_loss,
                X_init,
                method='lm',  # Levenberg-Marquardt
                verbose=0,
                max_nfev=50  # Limit iterations for speed
            )
            
            X_refined.append(result.x)
            
        except Exception as e:
            # If optimization fails, use initial estimate
            X_refined.append(X_init)
    
    return np.array(X_refined)


# Alias for lowercase function name (for compatibility with wrapper)
def nonlinear_triangulation(K, C1, R1, C2, R2, x1, x2, x0):
    """
    Alias for NonLinearTriangulation with lowercase name.
    """
    return NonLinearTriangulation(K, C1, R1, C2, R2, x1, x2, x0)
    