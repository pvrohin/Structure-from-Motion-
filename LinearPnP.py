import numpy as np

def LinearPnP(Xset, xset, K):
    """
    Estimates camera pose using linear least squares method on the 3D points and corresponding 2D projection on
    the image. Uses Direct Linear Transform (DLT) algorithm.

    Parameters
    ----------
    Xset : numpy.ndarray
        set of 3D points (N x 3)
    xset : numpy.ndarray
        set of 2D projections of 3D points on the image (N x 2)
    K : numpy.ndarray
        camera intrinsic matrix (3 x 3)
    
    Results
    -------
    C : numpy.ndarray
        the estimated center of camera (3,)
    R : numpy.ndarray
        the estimated rotation matrix (3 x 3)
    """
    # Convert to numpy arrays
    Xset = np.array(Xset)
    xset = np.array(xset)
    K = np.array(K)
    
    n_points = len(Xset)
    
    if n_points < 4:
        raise ValueError("At least 4 point correspondences are required for PnP")
    
    # Normalize 2D points: remove camera intrinsics
    # x_normalized = K^-1 * x_homogeneous
    ones = np.ones((n_points, 1))
    x_hom = np.hstack([xset, ones])
    x_normalized = (np.linalg.inv(K) @ x_hom.T).T
    
    # Build the constraint matrix A for DLT
    # For each point: [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x] * p = 0
    #                 [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y] * p = 0
    # where p is the flattened projection matrix P (12 parameters)
    
    A = np.zeros((2 * n_points, 12))
    for i in range(n_points):
        X, Y, Z = Xset[i, 0], Xset[i, 1], Xset[i, 2]
        x_norm, y_norm = x_normalized[i, 0], x_normalized[i, 1]
        
        # First equation
        A[2*i, :] = [X, Y, Z, 1, 0, 0, 0, 0, -x_norm*X, -x_norm*Y, -x_norm*Z, -x_norm]
        # Second equation
        A[2*i+1, :] = [0, 0, 0, 0, X, Y, Z, 1, -y_norm*X, -y_norm*Y, -y_norm*Z, -y_norm]
    
    # Solve for projection matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    P_vec = Vt[-1, :]  # Last row of V^T (corresponds to smallest singular value)
    P = P_vec.reshape(3, 4)
    
    # Extract camera matrix M and translation t from P = [M | t]
    M = P[:, :3]
    t = P[:, 3]
    
    # Ensure M has positive determinant (for proper orientation)
    if np.linalg.det(M) < 0:
        M = -M
        t = -t
    
    # Extract rotation and scale from M using RQ decomposition
    # M = K_camera * R, but we want R, so we use QR decomposition of M^T
    # M^T = Q * R_qr, so M = R_qr^T * Q^T
    # We want R = Q^T (orthogonal) and scale = R_qr^T (upper triangular)
    Q, R_qr = np.linalg.qr(M.T)
    R = Q.T
    scale_matrix = R_qr.T
    
    # Extract scale (should be positive)
    scale = np.mean(np.diag(scale_matrix))
    if scale < 0:
        R = -R
        scale = -scale
    
    # Normalize rotation matrix to ensure it's orthogonal
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Ensure det(R) = 1 (proper rotation)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Compute camera center: C = -R^T * t / scale
    t_normalized = t / scale
    C = -R.T @ t_normalized
    
    return C, R
    