import numpy as np

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Computes the 3D position of a set of points given its projections in two images using
    linear triangulation. Uses Direct Linear Transform (DLT) algorithm.

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

    Results
    -------
    X : array-like
        set of vectors representing the 3D positions of points in space (N x 3)
    """
    # Convert to numpy arrays
    K = np.array(K)
    C1 = np.array(C1)
    R1 = np.array(R1)
    C2 = np.array(C2)
    R2 = np.array(R2)
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    n_points = len(x1)
    
    # Build projection matrices
    # P = K * [R | -R*C]
    # For camera 1
    t1 = -R1 @ C1.reshape(3, 1)
    P1 = K @ np.hstack([R1, t1])
    
    # For camera 2
    t2 = -R2 @ C2.reshape(3, 1)
    P2 = K @ np.hstack([R2, t2])
    
    # Triangulate each point
    X = []
    
    for i in range(n_points):
        # Get 2D points
        x1_i = x1[i]
        x2_i = x2[i]
        
        # Convert to homogeneous coordinates
        x1_hom = np.array([x1_i[0], x1_i[1], 1])
        x2_hom = np.array([x2_i[0], x2_i[1], 1])
        
        # Build the constraint matrix A
        # For each point: x × (P * X) = 0
        # This gives us 2 independent equations per image (4 total)
        # [x[1]*P[2] - P[1]] * X = 0
        # [P[0] - x[0]*P[2]] * X = 0
        
        A = np.zeros((4, 4))
        
        # From first image: x1 × (P1 * X) = 0
        A[0, :] = x1_hom[1] * P1[2, :] - P1[1, :]  # y1 * P1[2] - P1[1]
        A[1, :] = P1[0, :] - x1_hom[0] * P1[2, :]  # P1[0] - x1 * P1[2]
        
        # From second image: x2 × (P2 * X) = 0
        A[2, :] = x2_hom[1] * P2[2, :] - P2[1, :]  # y2 * P2[2] - P2[1]
        A[3, :] = P2[0, :] - x2_hom[0] * P2[2, :]  # P2[0] - x2 * P2[2]
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        X_hom = Vt[-1, :]  # Last row of V^T (corresponds to smallest singular value)
        
        # Convert from homogeneous to 3D coordinates
        if abs(X_hom[3]) > 1e-8:
            X_3d = X_hom[:3] / X_hom[3]
        else:
            # Handle points at infinity
            X_3d = X_hom[:3]
        
        X.append(X_3d)
    
    return np.array(X)

# Alias for lowercase function name (for compatibility)
def linear_triangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Alias for LinearTriangulation with lowercase name.
    """
    return LinearTriangulation(K, C1, R1, C2, R2, x1, x2)
    


