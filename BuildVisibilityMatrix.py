import numpy as np

def getIndexAndVisibilityMatrix(X_f, inlier_feature_flag, camIndex):
    """
    Get the indices of inliers and the visibility matrix.
    
    This function finds 3D points that are both:
    1. Already triangulated (X_f is True)
    2. Visible in the camera being registered (inlier_feature_flag[:, camIndex] is True)

    Parameters
    ----------
    X_f : numpy.ndarray
        the array of found 3D points (boolean array or flag array, shape: (n_features,))
        True/1 indicates the 3D point has been triangulated
    inlier_feature_flag : numpy.ndarray
        the flag matrix representing the inliers (shape: (n_features, n_cameras))
        Value is 1 if feature is an inlier in that camera, 0 otherwise
    camIndex : int
        the number of the image being registered (0-based index)
    
    Results
    -------
    X_index : numpy.ndarray
        indices of inliers (1D array of feature indices where both conditions are met)
    visibility_matrix : numpy.ndarray
        visibility matrix (shape: (n_features, n_cameras))
        Value is 1 if 3D point exists and is visible in that camera, 0 otherwise
    """
    # Convert to numpy arrays
    X_f = np.array(X_f)
    inlier_feature_flag = np.array(inlier_feature_flag)
    
    # Ensure X_f is 1D boolean array
    if X_f.ndim > 1:
        # If it's a 2D array, take the first column or flatten
        if X_f.shape[1] == 1:
            X_f = X_f.flatten()
        else:
            # If it's a 3D coordinate array, check if points are non-zero
            X_f = np.any(X_f != 0, axis=1)
    
    # Convert to boolean
    X_f = X_f.astype(bool)
    
    # Get number of features and cameras
    n_features = len(X_f)
    n_cameras = inlier_feature_flag.shape[1]
    
    # Find indices where:
    # 1. 3D point exists (X_f is True)
    # 2. Feature is visible in the camera being registered (inlier_feature_flag[:, camIndex] is True)
    X_index = np.where(X_f & (inlier_feature_flag[:, camIndex] == 1))[0]
    
    # Build visibility matrix
    # visibility_matrix[i, j] = 1 if:
    #   - 3D point i exists (X_f[i] is True)
    #   - Feature i is visible in camera j (inlier_feature_flag[i, j] == 1)
    visibility_matrix = np.zeros((n_features, n_cameras), dtype=int)
    
    # For each feature that has a 3D point
    for i in range(n_features):
        if X_f[i]:
            # Mark visibility for all cameras where this feature is an inlier
            visibility_matrix[i, :] = inlier_feature_flag[i, :]
    
    return X_index, visibility_matrix