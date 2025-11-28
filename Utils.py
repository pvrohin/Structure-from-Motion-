import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DisambiguateCameraPose import DisambiguateCameraPose

def get_data(data_path, no_of_images):
    """
    Read data from matching files and extract features.

    :param data_path: Path to the directory containing matching files.
    :type data_path: str
    :param no_of_images: Number of images.
    :type no_of_images: int
    :return: x_features, y_features, feature_flags
    :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
    """   
    x_features = []
    y_features = []
    feature_flags = []

    for n in range(1, no_of_images):
        matching_file = open(data_path + "/matching" + str(n) + ".txt", "r")

        for i, row in enumerate(matching_file):
            if i == 0: continue # skip 1st line

            x_row = np.zeros((1, no_of_images))
            y_row = np.zeros((1, no_of_images))
            flag_row = np.zeros((1, no_of_images), dtype=int)
            row_elements = row.split()
            cols = [float(x) for x in row_elements]
            cols = np.asarray(cols)

            no_of_matches = cols[0]
            current_x = cols[4]
            current_y = cols[5]

            x_row[0, n-1] = current_x
            y_row[0, n-1] = current_y
            flag_row[0, n-1] = 1

            m = 1
            while no_of_matches > 1:
                image_id = int(cols[5+m])
                image_id_x = int(cols[6+m])
                image_id_y = int(cols[7+m])
                m += 3
                no_of_matches = no_of_matches - 1

                x_row[0, image_id-1] = image_id_x
                y_row[0, image_id-1] = image_id_y
                flag_row[0, image_id-1] = 1

            x_features.append(x_row)
            y_features.append(y_row)
            feature_flags.append(flag_row)

    x_features = np.asarray(x_features).reshape(-1, no_of_images)
    y_features = np.asarray(y_features).reshape(-1, no_of_images)
    feature_flags = np.asarray(feature_flags).reshape(-1, no_of_images)

    return x_features, y_features, feature_flags


def draw_features(image, coords, color=(255, 0, 0)):
    """
    Draw features on an image.
    
    :param image: Input image (BGR format)
    :type image: numpy.ndarray
    :param coords: Feature coordinates as (N, 2) array with [x, y] pairs
    :type coords: numpy.ndarray
    :param color: Color for drawing features (B, G, R)
    :type color: tuple
    :return: Image with features drawn
    :rtype: numpy.ndarray
    """
    # Convert coordinates to KeyPoint objects
    keypoints = [cv2.KeyPoint(float(coord[0]), float(coord[1]), 1) for coord in coords]
    image_ = cv2.drawKeypoints(image, keypoints, None, color=color, flags=0)
    return image_

def draw_feature_matches(image1, image2, image1_coords, image2_coords, save_path=None, color=(0, 255, 0)):
    """
    Draw feature matches between two images.

    :param image1: Path to image 1.
    :type image1: str
    :param image2: Path to image 2.
    :type image2: str
    :param image1_coords: Coordinates of features in image 1.
    :type image1_coords: numpy.ndarray
    :param image2_coords: Coordinates of features in image 2.
    :type image2_coords: numpy.ndarray
    :param save_path: Path to save the output image.
    :type save_path: str, optional
    :param color: Color of the matches.
    :type color: tuple[int, int, int], optional
    """

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    image1_ = draw_features(image1, image1_coords, color=(255, 0, 0))
    image2_ = draw_features(image2, image2_coords, color=(255, 0, 0))

    image1_coords_ = [cv2.KeyPoint(i[0], i[1], 1) for i in image1_coords]
    image2_coords_ = [cv2.KeyPoint(i[0], i[1], 1) for i in image2_coords]

    matches = [cv2.DMatch(i, i, 0) for i in range(len(image1_coords))]

    output_img = cv2.drawMatches(image1_, image1_coords_, image2_, image2_coords_, matches, None, matchColor=color, flags=2)

    # Display the output image
    if save_path:
        cv2.imwrite(save_path, output_img)
    else:
        cv2.imshow('Matches', output_img)
        cv2.waitKey(0)

# Global figure for hold functionality
_plot_fig = None
_plot_ax = None

def plot_world_coords(world_coords_list, save_path=None, color='b', hold=False):
    """
    Plot 3D world coordinates.
    
    :param world_coords_list: List of 3D point arrays (each N x 3)
    :type world_coords_list: list
    :param save_path: Path to save the plot
    :type save_path: str, optional
    :param color: Color for the points ('r', 'g', 'b', etc.)
    :type color: str, optional
    :param hold: If True, add to existing plot; if False, create new plot
    :type hold: bool, optional
    """
    global _plot_fig, _plot_ax
    
    if not hold or _plot_fig is None:
        _plot_fig = plt.figure(figsize=(10, 8))
        _plot_ax = _plot_fig.add_subplot(111, projection='3d')
    
    # Plot each set of coordinates
    for world_coords in world_coords_list:
        world_coords = np.array(world_coords)
        if len(world_coords) > 0:
            _plot_ax.scatter(world_coords[:, 0], world_coords[:, 1], world_coords[:, 2], 
                           c=color, s=1, alpha=0.6)
    
    _plot_ax.set_xlabel('X')
    _plot_ax.set_ylabel('Y')
    _plot_ax.set_zlabel('Z')
    _plot_ax.set_title('3D World Coordinates')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not hold:
            plt.close(_plot_fig)
            _plot_fig = None
            _plot_ax = None
    elif not hold:
        plt.show()
        _plot_fig = None
        _plot_ax = None

def plot_camera_pose(C, R, camera_id, save_path=None, hold=False):
    """
    Plot camera pose (position and orientation) in 3D space.
    
    :param C: Camera center (3,)
    :type C: numpy.ndarray
    :param R: Camera rotation matrix (3 x 3)
    :type R: numpy.ndarray
    :param camera_id: Camera ID for labeling
    :type camera_id: int
    :param save_path: Path to save the plot
    :type save_path: str, optional
    :param hold: If True, add to existing plot; if False, create new plot
    :type hold: bool, optional
    """
    global _plot_fig, _plot_ax
    
    if not hold or _plot_fig is None:
        _plot_fig = plt.figure(figsize=(10, 8))
        _plot_ax = _plot_fig.add_subplot(111, projection='3d')
    
    C = np.array(C)
    R = np.array(R)
    
    # Camera center
    _plot_ax.scatter([C[0]], [C[1]], [C[2]], c='red', s=100, marker='^')
    _plot_ax.text(C[0], C[1], C[2], f'Cam {camera_id}', fontsize=10)
    
    # Camera coordinate frame (axes)
    scale = 0.1  # Scale for visualization
    # X axis (red)
    x_axis = C + scale * R[:, 0]
    _plot_ax.plot([C[0], x_axis[0]], [C[1], x_axis[1]], [C[2], x_axis[2]], 'r-', linewidth=2)
    # Y axis (green)
    y_axis = C + scale * R[:, 1]
    _plot_ax.plot([C[0], y_axis[0]], [C[1], y_axis[1]], [C[2], y_axis[2]], 'g-', linewidth=2)
    # Z axis (blue)
    z_axis = C + scale * R[:, 2]
    _plot_ax.plot([C[0], z_axis[0]], [C[1], z_axis[1]], [C[2], z_axis[2]], 'b-', linewidth=2)
    
    _plot_ax.set_xlabel('X')
    _plot_ax.set_ylabel('Y')
    _plot_ax.set_zlabel('Z')
    _plot_ax.set_title('Camera Poses')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if not hold:
            plt.close(_plot_fig)
            _plot_fig = None
            _plot_ax = None
    elif not hold:
        plt.show()
        _plot_fig = None
        _plot_ax = None

def draw_reprojections(image_path1, image_path2, K, C1, R1, C2, R2, world_coords, 
                      image1_coords, image2_coords, save_path=None):
    """
    Draw reprojections of 3D points onto images.
    
    :param image_path1: Path to first image
    :type image_path1: str
    :param image_path2: Path to second image
    :type image_path2: str
    :param K: Camera intrinsic matrix (3 x 3)
    :type K: numpy.ndarray
    :param C1: Camera 1 center (3,)
    :type C1: numpy.ndarray
    :param R1: Camera 1 rotation matrix (3 x 3)
    :type R1: numpy.ndarray
    :param C2: Camera 2 center (3,)
    :type C2: numpy.ndarray
    :param R2: Camera 2 rotation matrix (3 x 3)
    :type R2: numpy.ndarray
    :param world_coords: 3D points (N x 3)
    :type world_coords: numpy.ndarray
    :param image1_coords: Observed 2D coordinates in image 1 (N x 2)
    :type image1_coords: numpy.ndarray
    :param image2_coords: Observed 2D coordinates in image 2 (N x 2)
    :type image2_coords: numpy.ndarray
    :param save_path: Path to save the output image
    :type save_path: str, optional
    """
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images")
        return
    
    # Convert BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Build projection matrices
    t1 = -R1 @ C1.reshape(3, 1)
    P1 = K @ np.hstack([R1, t1])
    
    t2 = -R2 @ C2.reshape(3, 1)
    P2 = K @ np.hstack([R2, t2])
    
    # Project 3D points to 2D
    world_coords = np.array(world_coords)
    n_points = len(world_coords)
    ones = np.ones((n_points, 1))
    X_hom = np.hstack([world_coords, ones])
    
    # Project to image 1
    x1_proj_hom = (P1 @ X_hom.T).T
    x1_proj = x1_proj_hom[:, :2] / (x1_proj_hom[:, 2:3] + 1e-8)
    
    # Project to image 2
    x2_proj_hom = (P2 @ X_hom.T).T
    x2_proj = x2_proj_hom[:, :2] / (x2_proj_hom[:, 2:3] + 1e-8)
    
    # Get image dimensions
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    
    # Create side-by-side visualization
    max_h = max(h1, h2)
    combined_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1_rgb
    combined_img[:h2, w1:w1+w2] = img2_rgb
    
    # Draw observed points (green) and reprojected points (red)
    image1_coords = np.array(image1_coords).astype(int)
    image2_coords = np.array(image2_coords).astype(int)
    x1_proj = x1_proj.astype(int)
    x2_proj = x2_proj.astype(int)
    
    for i in range(n_points):
        # Draw observed points (green circles) - using RGB since combined_img is RGB
        if 0 <= image1_coords[i, 0] < w1 and 0 <= image1_coords[i, 1] < h1:
            cv2.circle(combined_img, tuple(image1_coords[i]), 3, (0, 255, 0), -1)
        if 0 <= image2_coords[i, 0] < w2 and 0 <= image2_coords[i, 1] < h2:
            cv2.circle(combined_img, (image2_coords[i, 0] + w1, image2_coords[i, 1]), 3, (0, 255, 0), -1)
        
        # Draw reprojected points (red circles) - using RGB since combined_img is RGB
        if 0 <= x1_proj[i, 0] < w1 and 0 <= x1_proj[i, 1] < h1:
            cv2.circle(combined_img, tuple(x1_proj[i]), 3, (255, 0, 0), -1)
        if 0 <= x2_proj[i, 0] < w2 and 0 <= x2_proj[i, 1] < h2:
            cv2.circle(combined_img, (x2_proj[i, 0] + w1, x2_proj[i, 1]), 3, (255, 0, 0), -1)
        
        # Draw lines connecting observed and reprojected (yellow) - using RGB since combined_img is RGB
        if (0 <= image1_coords[i, 0] < w1 and 0 <= image1_coords[i, 1] < h1 and
            0 <= x1_proj[i, 0] < w1 and 0 <= x1_proj[i, 1] < h1):
            cv2.line(combined_img, tuple(image1_coords[i]), tuple(x1_proj[i]), (255, 255, 0), 1)
        
        if (0 <= image2_coords[i, 0] < w2 and 0 <= image2_coords[i, 1] < h2 and
            0 <= x2_proj[i, 0] < w2 and 0 <= x2_proj[i, 1] < h2):
            cv2.line(combined_img, 
                    (image2_coords[i, 0] + w1, image2_coords[i, 1]),
                    (x2_proj[i, 0] + w1, x2_proj[i, 1]), 
                    (255, 255, 0), 1)
    
    # Save or display
    if save_path:
        combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, combined_img_bgr)
    else:
        plt.figure(figsize=(15, 8))
        plt.imshow(combined_img)
        plt.axis('off')
        plt.title('Reprojections: Green=Observed, Red=Reprojected, Yellow=Error')
        plt.show()

def get_correct_camera_pose(Cset, Rset, Xset):
    """
    Find the correct camera pose from multiple possible poses using cheirality condition.
    
    This is a wrapper around DisambiguateCameraPose that returns only the camera pose
    (C, R) without the 3D points.
    
    :param Cset: Set of camera centers (4 x 3 array, each row is a camera center)
    :type Cset: numpy.ndarray
    :param Rset: Set of rotation matrices (4 x 3 x 3 array)
    :type Rset: numpy.ndarray
    :param Xset: Set of 3D point arrays (list of 4 arrays, each N x 3)
    :type Xset: list or numpy.ndarray
    :return: Corrected camera center and rotation matrix (C, R)
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    C_corr, R_corr, _ = DisambiguateCameraPose(Cset, Rset, Xset)
    return C_corr, R_corr