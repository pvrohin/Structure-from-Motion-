import numpy as np
import random

def find_homography(image1_coords, image2_coords):
    """
    Find the homography matrix between two images using Direct Linear Transform (DLT).
    Requires at least 4 point correspondences.

    :param image1_coords: Image 1 coordinates (N x 2 array).
    :type image1_coords: numpy.ndarray
    :param image2_coords: Image 2 coordinates (N x 2 array).
    :type image2_coords: numpy.ndarray
    :return: Homography matrix (3 x 3).
    :rtype: numpy.ndarray
    """
    # Convert to numpy arrays
    image1_coords = np.array(image1_coords)
    image2_coords = np.array(image2_coords)
    
    n_points = len(image1_coords)
    
    if n_points < 4:
        raise ValueError("At least 4 point correspondences are required for homography estimation")
    
    # Normalize points (similar to Hartley normalization for fundamental matrix)
    # Compute centroids
    mean1 = np.mean(image1_coords, axis=0)
    mean2 = np.mean(image2_coords, axis=0)
    
    # Center the points
    centered1 = image1_coords - mean1
    centered2 = image2_coords - mean2
    
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
    
    # Convert to homogeneous coordinates
    ones = np.ones((n_points, 1))
    points1_hom = np.hstack([image1_coords, ones])
    points2_hom = np.hstack([image2_coords, ones])
    
    # Normalize points
    normalized1 = (T1 @ points1_hom.T).T
    normalized2 = (T2 @ points2_hom.T).T
    
    # Build the constraint matrix A for DLT algorithm
    # For each point correspondence (x1, y1) <-> (x2, y2), we have:
    # [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2] * h = 0
    # [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2] * h = 0
    # where h is the flattened homography matrix
    
    A = np.zeros((2 * n_points, 9))
    for i in range(n_points):
        x1, y1 = normalized1[i, 0], normalized1[i, 1]
        x2, y2 = normalized2[i, 0], normalized2[i, 1]
        
        # First equation
        A[2*i, :] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
        # Second equation
        A[2*i+1, :] = [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
    
    # Solve for homography using SVD
    U, S, Vt = np.linalg.svd(A)
    H_vec = Vt[-1, :]  # Last row of V^T (corresponds to smallest singular value)
    H = H_vec.reshape(3, 3)
    
    # Denormalize
    H = np.linalg.inv(T2) @ H @ T1
    
    # Normalize H
    H = H / H[2, 2]
    
    return H


def get_homography_inliers(image1_coords_org, image2_coords_org, idx, threshold=30, n_max=1000):
    """
    Get the inliers for the homography matrix using RANSAC.

    :param image1_coords_org: Coordinates of the points in the first image (N x 2).
    :type image1_coords_org: numpy.ndarray
    :param image2_coords_org: Coordinates of the points in the second image (N x 2).
    :type image2_coords_org: numpy.ndarray
    :param idx: Indices of the points.
    :type idx: numpy.ndarray
    :param threshold: Error threshold for inlier detection (default: 30 pixels).
    :type threshold: float
    :param n_max: Maximum number of RANSAC iterations (default: 1000).
    :type n_max: int, optional
    :return: Homography matrix, inlier indices
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    # Ensure inputs are numpy arrays
    image1_coords_org = np.array(image1_coords_org)
    image2_coords_org = np.array(image2_coords_org)
    idx = np.array(idx)
    
    n_points = len(image1_coords_org)
    
    if n_points < 4:
        # Not enough points for homography estimation
        return None, np.array([])
    
    # Initialize best results
    best_inlier_count = 0
    best_inlier_indices = np.array([])
    H_best = None
    
    # Convert points to homogeneous coordinates
    ones = np.ones((n_points, 1))
    points1_hom = np.hstack([image1_coords_org, ones])
    points2_hom = np.hstack([image2_coords_org, ones])
    
    # RANSAC iterations
    for iteration in range(n_max):
        # Randomly sample 4 points (minimum required for homography)
        sample_indices = random.sample(range(n_points), min(4, n_points))
        sample_points1 = image1_coords_org[sample_indices]
        sample_points2 = image2_coords_org[sample_indices]
        
        try:
            # Estimate homography from sample
            H = find_homography(sample_points1, sample_points2)
            
            # Compute reprojection error for all points
            # Transform points from image1 to image2 using homography
            transformed_points = (H @ points1_hom.T).T  # Shape: (n_points, 3)
            
            # Convert back to 2D (normalize by z coordinate)
            transformed_points_2d = transformed_points[:, :2] / (transformed_points[:, 2:3] + 1e-8)
            
            # Compute Euclidean distance between transformed points and actual points
            errors = np.sqrt(np.sum((transformed_points_2d - image2_coords_org)**2, axis=1))
            
            # Find inliers (points with error below threshold)
            inlier_mask = errors < threshold
            inlier_count = np.sum(inlier_mask)
            
            # Update best result if this iteration has more inliers
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inlier_indices = np.where(inlier_mask)[0]
                H_best = H.copy()
                
        except Exception as e:
            # If estimation fails, skip this iteration
            continue
    
    # If no good H was found, return None
    if H_best is None or best_inlier_count == 0:
        return None, np.array([])
    
    # Map inlier indices back to original index array
    inlier_idx = idx[best_inlier_indices]
    
    return H_best, inlier_idx
