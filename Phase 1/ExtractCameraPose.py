import numpy as np

def ExtractCameraPose(E):
    """
    Estimate the camera pose given the essential matrix.
    Extracts 4 possible camera pose configurations from the essential matrix.

    Parameters
    ----------
    E : array-like
        The essential matrix (3 x 3)
    
    Results
    -------
    Cset : array-like
        the set of camera centers (4 x 3 array, each row is a camera center)
    Rset : array-like
        the set of rotation matrices (4 x 3 x 3 array)
    """
    # Convert to numpy array
    E = np.array(E)
    
    # Perform SVD on essential matrix
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper SVD (det(U) * det(V^T) = 1)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        U[:, -1] *= -1
    
    # Define W matrix (skew-symmetric matrix)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Two possible translations (up to scale, third column of U)
    t = U[:, 2]
    
    # Normalize translation
    t = t / np.linalg.norm(t)
    
    # Four possible camera pose configurations
    # For each configuration, camera center C = -R^T * t
    Cset = []
    Rset = []
    
    # Configuration 1: R1, t
    C1 = -R1.T @ t
    Cset.append(C1)
    Rset.append(R1)
    
    # Configuration 2: R1, -t
    C2 = -R1.T @ (-t)
    Cset.append(C2)
    Rset.append(R1)
    
    # Configuration 3: R2, t
    C3 = -R2.T @ t
    Cset.append(C3)
    Rset.append(R2)
    
    # Configuration 4: R2, -t
    C4 = -R2.T @ (-t)
    Cset.append(C4)
    Rset.append(R2)
    
    # Convert to numpy arrays
    Cset = np.array(Cset)
    Rset = np.array(Rset)
    
    return Cset, Rset