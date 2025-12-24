import numpy as np

def DisambiguateCameraPose(Cset, Rset, Xset):
    """
    Find unique camera pose by checking the cheirality condition.
    
    The cheirality condition states that 3D points must be in front of both cameras
    (positive Z coordinate in each camera's coordinate frame).

    Parameters
    ----------
    Cset : array-like
        the set of camera centers (4 x 3 array, each row is a camera center)
    Rset : array-like
        the set of rotation matrices (4 x 3 x 3 array)
    Xset : array-like or list
        set of vectors representing the 3D positions of points in space
        Can be a list of 4 arrays (one for each configuration) or a 4D array
        Each element has shape (N x 3) where N is the number of points
    
    Results
    -------
    best_C : array-like
        the corrected camera center (3,)
    best_R : array-like
        the corrected rotation matrix (3 x 3)
    best_X : array-like
        the corrected set of vectors representing the 3D positions of points in space (N x 3)
    """
    # Convert to numpy arrays
    Cset = np.array(Cset)
    Rset = np.array(Rset)
    
    # Handle Xset - it can be a list of arrays or a 3D array
    if isinstance(Xset, list):
        Xset = [np.array(X) for X in Xset]
    else:
        Xset = [np.array(Xset[i]) for i in range(len(Xset))]
    
    n_configs = len(Cset)
    
    # Assume first camera is at origin with identity rotation
    C1 = np.zeros(3)
    R1 = np.eye(3)
    
    best_count = -1
    best_idx = 0
    best_C = None
    best_R = None
    best_X = None
    
    # Check each configuration
    for i in range(n_configs):
        C2 = Cset[i]
        R2 = Rset[i]
        X = Xset[i]  # 3D points triangulated with this configuration
        
        if len(X) == 0:
            continue
        
        # Transform points to camera coordinate frames
        # Camera 1: points are already in world frame, check Z > 0
        # For camera 1 at origin with identity, Z coordinate is just X[:, 2]
        z_cam1 = X[:, 2]
        
        # Camera 2: transform points to camera 2's coordinate frame
        # Transform: X_cam2 = R2 @ (X - C2)
        X_relative = X - C2.reshape(1, 3)  # Points relative to camera 2 center
        X_cam2 = (R2 @ X_relative.T).T  # Transform to camera 2 frame
        z_cam2 = X_cam2[:, 2]
        
        # Count points that are in front of both cameras (positive Z in both frames)
        in_front_mask = (z_cam1 > 0) & (z_cam2 > 0)
        count = np.sum(in_front_mask)
        
        # Update best configuration if this one has more points in front
        if count > best_count:
            best_count = count
            best_idx = i
            best_C = C2.copy()
            best_R = R2.copy()
            best_X = X.copy()
    
    # If no valid configuration found, return the first one
    if best_C is None:
        best_C = Cset[0]
        best_R = Rset[0]
        best_X = Xset[0]
        print("Warning: No valid camera pose found, using first configuration")
    else:
        print(f"Selected camera pose configuration {best_idx} with {best_count} points in front of both cameras")
    
    return best_C, best_R, best_X