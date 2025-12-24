import numpy as np

def EstimateFundamentalMatrix(points1, points2):
    """
    Estimates the fundamental matrix from the eight randomly selected feature matches.
    Uses the 8-point algorithm with normalization (Hartley normalization).

    Parameters
    ----------
    points1 : array-like
        points for matching from image 1 (N x 2 or 8 x 2)
    points2 : array-like
        points for matching from image 2 (N x 2 or 8 x 2)

    Results
    -------
    F : array-like
         the resulting fundamental matrix (3 x 3)
    """
    # Convert to numpy arrays and ensure correct shape
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Ensure points are in 2D format
    if points1.shape[1] == 2:
        n_points = points1.shape[0]
        
        # Normalize points (Hartley normalization)
        # Compute centroids
        mean1 = np.mean(points1, axis=0)
        mean2 = np.mean(points2, axis=0)
        
        # Center the points
        centered1 = points1 - mean1
        centered2 = points2 - mean2
        
        # Compute scale (average distance from origin)
        scale1 = np.sqrt(2) / (np.mean(np.sqrt(np.sum(centered1**2, axis=1))) + 1e-8)
        scale2 = np.sqrt(2) / (np.mean(np.sqrt(np.sum(centered2**2, axis=1))) + 1e-8)
        
        # Normalization matrices
        T1 = np.array([[scale1, 0, -scale1 * mean1[0]],
                       [0, scale1, -scale1 * mean1[1]],
                       [0, 0, 1]])
        T2 = np.array([[scale2, 0, -scale2 * mean2[0]],
                       [0, scale2, -scale2 * mean2[1]],
                       [0, 0, 1]])
        
        # Normalize points
        ones = np.ones((n_points, 1))
        points1_hom = np.hstack([points1, ones])
        points2_hom = np.hstack([points2, ones])
        
        normalized1 = (T1 @ points1_hom.T).T
        normalized2 = (T2 @ points2_hom.T).T
        
        # Build the constraint matrix A for the 8-point algorithm
        A = np.zeros((n_points, 9))
        for i in range(n_points):
            x1, y1 = normalized1[i, 0], normalized1[i, 1]
            x2, y2 = normalized2[i, 0], normalized2[i, 1]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        
        # Solve for F using SVD
        U, S, Vt = np.linalg.svd(A)
        F_vec = Vt[-1, :]  # Last row of V^T (corresponds to smallest singular value)
        F = F_vec.reshape(3, 3)
        
        # Enforce rank-2 constraint (fundamental matrix must have rank 2)
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0  # Set smallest singular value to 0
        F = U @ np.diag(S) @ Vt
        
        # Denormalize
        F = T2.T @ F @ T1
        
        # Normalize F
        F = F / F[2, 2]
        
    else:
        raise ValueError("Points must be in 2D format (N x 2)")
    
    return F
    