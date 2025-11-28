import os
import argparse
import re
from itertools import combinations
from Utils import *
from GetHomographyInliers import *
from GetInliersRANSAC import get_inliers_ransac

def natural_sort_key(text):
    """Convert text to a tuple for natural sorting (e.g., '2.png' before '10.png')"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

def sfm_wrapper(data_path, results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    image_ids = []
    image_paths = []
    for file in sorted(os.listdir(data_path), key=natural_sort_key):
        if file.endswith(".png"):
            image_paths.append(data_path + file)
            image_ids.append(int(os.path.splitext(file)[0]))

    print(image_ids)
    print(image_paths)

    feature_x, feature_y, feature_flag = get_data(data_path, len(image_ids))
    print(feature_x.shape)
    print(feature_y.shape)
    print(feature_flag.shape)

    print(feature_x)
    print(feature_y)
    print(feature_flag)

    #Create all combinations of feature matches from 1 to 5 
    feature_combinations = list(combinations(range(1, 6), 2))
    print(feature_combinations)

    # Create directory for original matches
    original_matches_dir = os.path.join(results_dir, "original_matches")
    if not os.path.exists(original_matches_dir):
        os.makedirs(original_matches_dir)
    
    # Create directory for homography matches
    homography_matches_dir = os.path.join(results_dir, "homography_matches")
    if not os.path.exists(homography_matches_dir):
        os.makedirs(homography_matches_dir)
    
    print("\nSaving original matches for all image combinations...")
    for combination in feature_combinations:
        image1_id, image2_id = combination
        combination_key = str(image1_id) + "_" + str(image2_id)

        print(f"Processing combination: {combination_key}")

        _idx = np.where(feature_flag[:, image1_id-1] & feature_flag[:, image2_id-1])
        image1_coords_org = np.hstack((feature_x[_idx, image1_id-1].reshape((-1,1)), feature_y[_idx, image1_id-1].reshape((-1,1))))
        image2_coords_org = np.hstack((feature_x[_idx, image2_id-1].reshape((-1,1)), feature_y[_idx, image2_id-1].reshape((-1,1))))
        idx = np.array(_idx).reshape(-1)

        # Save original matches for this combination
        save_path = os.path.join(original_matches_dir, f'original_matches_{combination_key}.png')
        draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords_org, image2_coords_org,
                                 color=(0, 0, 255), save_path=save_path)
        print(f"  Saved: {save_path} ({len(idx)} matches)")
    
        # Get homography inliers
        H, h_inlier_idx = get_homography_inliers(image1_coords_org, image2_coords_org, idx, threshold=30, n_max=1000)
        
        if H is not None and len(h_inlier_idx) > 0:
            # Extract inlier coordinates
            image1_coords = np.hstack((feature_x[h_inlier_idx, image1_id-1].reshape((-1,1)), feature_y[h_inlier_idx, image1_id-1].reshape((-1,1))))
            image2_coords = np.hstack((feature_x[h_inlier_idx, image2_id-1].reshape((-1,1)), feature_y[h_inlier_idx, image2_id-1].reshape((-1,1))))

            print(f'  Number of homography inliers: {len(image1_coords)}')
            
            # Save homography matches for this combination
            save_path = os.path.join(homography_matches_dir, f'homography_matches_{combination_key}.png')
            draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords, image2_coords,
                                     color=(0, 255, 255), save_path=save_path)
            print(f"  Saved: {save_path} ({len(h_inlier_idx)} inliers)")
        else:
            print(f"  No homography inliers found for combination {combination_key}")

        F, f_inlier_idx = get_inliers_ransac(image1_coords, image2_coords, h_inlier_idx, threshold=0.06, n_max=1000)

        if combination_key == '1_2': F_12 = F

        image1_coords_inliers = np.hstack((feature_x[f_inlier_idx, image1_id-1].reshape((-1,1)), feature_y[f_inlier_idx, image1_id-1].reshape((-1,1))))
        image2_coords_inliers = np.hstack((feature_x[f_inlier_idx, image2_id-1].reshape((-1,1)), feature_y[f_inlier_idx, image2_id-1].reshape((-1,1))))

        print('Number of matches RANSAC: ', len(image1_coords_inliers))
        draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords_inliers, image2_coords_inliers,
                                 color=(0, 255, 0), save_path=results_dir+f'/ransac_matches_{combination_key}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="P3Data/", help="Path of input images and feature matches text files")
    parser.add_argument("--results_dir", type=str, default="P3Data/Results", help="Directory to save results")
    args = parser.parse_args()

    sfm_wrapper(args.data_path, args.results_dir)