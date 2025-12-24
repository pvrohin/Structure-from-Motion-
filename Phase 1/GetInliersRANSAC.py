import random
import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInliersRANSAC(points1, points2, index, threshold=0.06, n_max=1000):
    """
    Rejects the outliers from a set of feature matches and returns the inliner indices.

    Parameters
    ----------
    points1 : numpy.ndarray
        feature points for matching in first image
    points2 : numpy.ndarray
        feature points for matching in second image
    index : numpy.ndarray
        index of all feature matches
    threshold : float
        threshold for inlier detection (default: 0.06)
    n_max : int
        maximum number of RANSAC iterations (default: 1000)

    Results
    -------
    inlier_index : numpy.ndarray
        index of all inlier feature matches
    outlier_index : numpy.ndarray
        index of all outlier feature matches for visulaization
    F_best : numpy.ndarray
        the best fundamental matrix
    """
    # Ensure inputs are numpy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)
    index = np.array(index)
    
    n_points = len(points1)
    
    if n_points < 8:
        # Not enough points for fundamental matrix estimation
        return np.array([]), index, None
    
    # Initialize best results
    best_inlier_count = 0
    best_inlier_indices = np.array([])
    F_best = None
    
    # Convert points to homogeneous coordinates
    ones = np.ones((n_points, 1))
    points1_hom = np.hstack([points1, ones])
    points2_hom = np.hstack([points2, ones])
    
    # RANSAC iterations
    for iteration in range(n_max):
        # Randomly sample 8 points
        sample_indices = random.sample(range(n_points), min(8, n_points))
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]
        
        try:
            # Estimate fundamental matrix from sample
            F = EstimateFundamentalMatrix(sample_points1, sample_points2)
            
            # Compute epipolar constraint error for all points
            # Use symmetric epipolar distance: |x2^T * F * x1| / (||Fx1|| + ||F^T x2||)
            
            # Compute F * x1 for all points
            Fx1 = (F @ points1_hom.T).T  # Shape: (n_points, 3)
            # Compute F^T * x2 for all points
            FTx2 = (F.T @ points2_hom.T).T  # Shape: (n_points, 3)
            
            # Compute x2^T * F * x1 (epipolar constraint)
            epipolar_constraint = np.sum(points2_hom * Fx1, axis=1)  # Shape: (n_points,)
            
            # Compute symmetric epipolar distance
            # Distance from point to epipolar line
            dist1 = np.abs(epipolar_constraint) / (np.sqrt(Fx1[:, 0]**2 + Fx1[:, 1]**2) + 1e-8)
            dist2 = np.abs(epipolar_constraint) / (np.sqrt(FTx2[:, 0]**2 + FTx2[:, 1]**2) + 1e-8)
            errors = (dist1 + dist2) / 2
            
            # Find inliers (points with error below threshold)
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            
            # Update best result if this iteration has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_indices = np.where(inlier_mask)[0]
                F_best = F.copy()
                
        except Exception as e:
            # If estimation fails, skip this iteration
            continue
    
    # If no good F was found, return empty results
    if F_best is None or best_inlier_count == 0:
        return np.array([]), index, None
    
    # best_inlier_indices are indices into the points array (0 to n_points-1)
    # These correspond to indices in the index array
    inlier_index = best_inlier_indices
    
    # Get outlier indices (all indices not in inlier_index)
    outlier_mask = ~np.isin(np.arange(n_points), best_inlier_indices)
    outlier_indices = index[outlier_mask]
    
    return inlier_index, outlier_indices, F_best

# Alias for lowercase function name (for compatibility)
def get_inliers_ransac(points1, points2, index, threshold=0.06, n_max=1000):
    """
    Alias for GetInliersRANSAC with lowercase name.
    Returns (F_best, inlier_index) instead of (inlier_index, outlier_index, F_best)
    for compatibility with wrapper code.
    """
    inlier_index, outlier_indices, F_best = GetInliersRANSAC(points1, points2, index, threshold, n_max)
    # Map inlier indices back to original index array
    if len(inlier_index) > 0:
        inlier_idx = index[inlier_index]
    else:
        inlier_idx = np.array([])
    return F_best, inlier_idx
    