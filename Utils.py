import numpy as np
import cv2
import os

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

def draw_feature_matches(image_path1, image_path2, feature_x, feature_y, feature_flag, 
                         img_idx1, img_idx2, save_path=None, max_matches=100):
    """
    Draw feature matches between two images.
    
    :param image_path1: Path to first image
    :type image_path1: str
    :param image_path2: Path to second image
    :type image_path2: str
    :param feature_x: Array of x-coordinates for features (num_features, num_images)
    :type feature_x: numpy.ndarray
    :param feature_y: Array of y-coordinates for features (num_features, num_images)
    :type feature_y: numpy.ndarray
    :param feature_flag: Array of flags indicating feature presence (num_features, num_images)
    :type feature_flag: numpy.ndarray
    :param img_idx1: Index of first image (0-based)
    :type img_idx1: int
    :param img_idx2: Index of second image (0-based)
    :type img_idx2: int
    :param save_path: Optional path to save the visualization
    :type save_path: str or None
    :param max_matches: Maximum number of matches to display (for clarity)
    :type max_matches: int
    :return: Combined image with matches drawn
    :rtype: numpy.ndarray
    """
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None:
        raise ValueError(f"Could not load image: {image_path1}")
    if img2 is None:
        raise ValueError(f"Could not load image: {image_path2}")
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]
    
    # Find features that exist in both images
    valid_mask = (feature_flag[:, img_idx1] == 1) & (feature_flag[:, img_idx2] == 1)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"No matching features found between image {img_idx1+1} and image {img_idx2+1}")
        return None
    
    # Limit number of matches for visualization
    if len(valid_indices) > max_matches:
        np.random.seed(42)  # For reproducibility
        valid_indices = np.random.choice(valid_indices, max_matches, replace=False)
    
    # Extract matching points
    pts1 = np.column_stack([feature_x[valid_indices, img_idx1], 
                           feature_y[valid_indices, img_idx1]]).astype(int)
    pts2 = np.column_stack([feature_x[valid_indices, img_idx2], 
                           feature_y[valid_indices, img_idx2]]).astype(int)
    
    # Create side-by-side visualization
    max_h = max(h1, h2)
    combined_img = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1_rgb
    combined_img[:h2, w1:w1+w2] = img2_rgb
    
    # Draw matches
    for pt1, pt2 in zip(pts1, pts2):
        # Generate random color for each match
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw points
        cv2.circle(combined_img, tuple(pt1), 3, color, -1)
        cv2.circle(combined_img, (pt2[0] + w1, pt2[1]), 3, color, -1)
        
        # Draw line connecting matches
        cv2.line(combined_img, tuple(pt1), (pt2[0] + w1, pt2[1]), color, 1)
    
    # Save if path provided
    if save_path:
        # Convert back to BGR for saving with OpenCV
        combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, combined_img_bgr)
        print(f"Saved feature matches visualization to: {save_path}")
    
    return combined_img

def draw_all_feature_matches(image_paths, feature_x, feature_y, feature_flag, 
                             image_pairs, results_dir, max_matches=100):
    """
    Draw feature matches for all specified image pairs.
    
    :param image_paths: List of paths to images
    :type image_paths: list
    :param feature_x: Array of x-coordinates for features (num_features, num_images)
    :type feature_x: numpy.ndarray
    :param feature_y: Array of y-coordinates for features (num_features, num_images)
    :type feature_y: numpy.ndarray
    :param feature_flag: Array of flags indicating feature presence (num_features, num_images)
    :type feature_flag: numpy.ndarray
    :param image_pairs: List of tuples (img_idx1, img_idx2) where indices are 1-based
    :type image_pairs: list
    :param results_dir: Directory to save the visualizations
    :type results_dir: str
    :param max_matches: Maximum number of matches to display per pair
    :type max_matches: int
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for img_idx1, img_idx2 in image_pairs:
        # Convert from 1-based to 0-based indexing
        idx1 = img_idx1 - 1
        idx2 = img_idx2 - 1
        
        if idx1 >= len(image_paths) or idx2 >= len(image_paths):
            print(f"Skipping pair ({img_idx1}, {img_idx2}): index out of range")
            continue
        
        save_path = os.path.join(results_dir, f"matches_{img_idx1}_{img_idx2}.png")
        
        try:
            combined_img = draw_feature_matches(
                image_paths[idx1], 
                image_paths[idx2],
                feature_x, 
                feature_y, 
                feature_flag,
                idx1, 
                idx2,
                save_path=save_path,
                max_matches=max_matches
            )
            
            if combined_img is not None:
                num_matches = np.sum((feature_flag[:, idx1] == 1) & (feature_flag[:, idx2] == 1))
                print(f"Image pair ({img_idx1}, {img_idx2}): {num_matches} matches found")
        except Exception as e:
            print(f"Error processing pair ({img_idx1}, {img_idx2}): {e}")
