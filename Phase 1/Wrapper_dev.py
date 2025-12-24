import os
import argparse
import re
from itertools import combinations
from Utils import *
from GetHomographyInliers import *
from GetInliersRANSAC import get_inliers_ransac
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from NonLinearTriangulation import nonlinear_triangulation
from LinearTriangulation import linear_triangulation
from NonlinearPnP import nonlinear_PnP
from PnPRANSAC import PnPRANSAC
from BundleAdjustment import perform_bundle_adjustment

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
    filtered_feature_flags = np.zeros_like(feature_flag)
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
    
    # Create directory for RANSAC matches
    ransac_matches_dir = os.path.join(results_dir, "ransac_matches")
    if not os.path.exists(ransac_matches_dir):
        os.makedirs(ransac_matches_dir)

    inliers = {} # Dictionary of global inliers between all combinations

    camera_poses = {} # Dictionary of camera poses for all combinations

    structure = {} # Dictionary of world coordinates for all combinations
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
            continue  # Skip RANSAC if no homography inliers

        F, f_inlier_idx = get_inliers_ransac(image1_coords, image2_coords, h_inlier_idx, threshold=0.06, n_max=1000)
        
        if F is None or len(f_inlier_idx) == 0:
            print(f"  No RANSAC inliers found for combination {combination_key}")
            continue

        if combination_key == '1_2': F_12 = F

        image1_coords_inliers = np.hstack((feature_x[f_inlier_idx, image1_id-1].reshape((-1,1)), feature_y[f_inlier_idx, image1_id-1].reshape((-1,1))))
        image2_coords_inliers = np.hstack((feature_x[f_inlier_idx, image2_id-1].reshape((-1,1)), feature_y[f_inlier_idx, image2_id-1].reshape((-1,1))))

        print('Number of matches RANSAC: ', len(image1_coords_inliers))
        save_path = os.path.join(ransac_matches_dir, f'ransac_matches_{combination_key}.png')
        draw_feature_matches(image_paths[image1_id-1], image_paths[image2_id-1], image1_coords_inliers, image2_coords_inliers,
                                 color=(0, 255, 0), save_path=save_path)
        print(f"  Saved: {save_path} ({len(f_inlier_idx)} inliers)")
        inliers[combination_key] = [image1_coords_inliers, image2_coords_inliers]
        filtered_feature_flags[f_inlier_idx, image1_id-1] = 1
        filtered_feature_flags[f_inlier_idx, image2_id-1] = 1

    print(f"\nAll original matches saved to: {original_matches_dir}")
    print(f"All homography matches saved to: {homography_matches_dir}")
    print(f"All RANSAC matches saved to: {ransac_matches_dir}")
    
    #Consider only first 2 image pairs for now
    image1_id, image2_id = feature_combinations[0]
    combination_key = str(image1_id) + "_" + str(image2_id)
    result_dir = os.path.join(results_dir, combination_key)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print(f'Processing image pair: {combination_key}')

    _idx = np.where(filtered_feature_flags[:, image1_id-1] & filtered_feature_flags[:, image2_id-1])
    image1_inliers = np.hstack((feature_x[_idx, image1_id-1].reshape((-1, 1)), feature_y[_idx, image1_id-1].reshape((-1, 1))))
    image2_inliers = np.hstack((feature_x[_idx, image2_id-1].reshape((-1, 1)), feature_y[_idx, image2_id-1].reshape((-1, 1))))

    K = np.array([[531.122155322710, 0 ,407.192550839899],[0, 531.541737503901, 313.308715048366],[0,0,1]])
    
    # Get F from the first combination (1_2)
    if 'F_12' not in locals():
        # If F_12 wasn't set, get it from the inliers dictionary or recompute
        print("Warning: F_12 not found, using F from combination 1_2")
        # Try to get from stored inliers or recompute
        F_12 = None
    
    if F_12 is None:
        print("Error: Fundamental matrix F not available")
        return
    
    E = EssentialMatrixFromFundamentalMatrix(F_12, K)

    Cset, Rset = ExtractCameraPose(E)

    C0 = np.zeros(3)
    R0 = np.eye(3)

    camera_poses[image1_id] = [C0, R0]

    possible_world_coords = []

    for c, r in zip(Cset, Rset):
        _world_coords = linear_triangulation(K, C0, R0, c, r, image1_inliers, image2_inliers)
        possible_world_coords.append(_world_coords)

    # no_log flag - set to True to skip plotting
    no_log = False
    if not no_log:
        plot_world_coords(possible_world_coords, save_path=os.path.join(result_dir, 'possible_world_coords.png'))

    C_corr, R_corr = get_correct_camera_pose(Cset, Rset, possible_world_coords)
    camera_poses[image2_id] = [C_corr, R_corr]

    corr_world_coords = linear_triangulation(K, C0, R0, C_corr, R_corr, image1_inliers, image2_inliers)

    
    plot_world_coords([corr_world_coords], save_path=result_dir+'/corrected_world_coords.png', color='r', hold=True)

    draw_reprojections(image_paths[image1_id-1], image_paths[image2_id-1], K, C0, R0, C_corr, R_corr, corr_world_coords,
                           image1_inliers, image2_inliers, save_path=result_dir+'/corrected_reprojections.png')

    refined_world_coords = nonlinear_triangulation(K, C0, R0, C_corr, R_corr, image1_inliers, image2_inliers, corr_world_coords)

    
    plot_world_coords([refined_world_coords], save_path=result_dir+'/refined_world_coords.png', hold=True)

    plot_camera_pose(C0, R0, 1, save_path=result_dir+'/with_camera_pose.png', hold=True)
    plot_camera_pose(C_corr, R_corr, 2, save_path=result_dir+'/with_camera_pose.png', hold=True)

    draw_reprojections(image_paths[image1_id-1], image_paths[image2_id-1], K, C0, R0, C_corr, R_corr, refined_world_coords,
                           image1_inliers, image2_inliers, save_path=result_dir+'/refined_reprojections.png')

    print('Number of world points added: ', len(refined_world_coords))
    structure[combination_key] = refined_world_coords

    all_world_coords = np.zeros((feature_x.shape[0], 3))
    all_world_coords_2 = np.zeros((feature_x.shape[0], 3)) # plot
    cam_indices = np.zeros((feature_x.shape[0], 1), dtype=int)
    filtered_world_coords = np.zeros((feature_x.shape[0], 1), dtype=int)

    all_world_coords[_idx] = refined_world_coords
    all_world_coords_2[_idx] = refined_world_coords # plot
    filtered_world_coords[_idx] = 1
    cam_indices[_idx] = 1
    # Filter out points with negative Z coordinates (behind camera)
    filtered_world_coords[np.where(all_world_coords[:, 2] < 0)[0]] = 0

    C_set = []
    R_set = []
    C_set.append(C0)
    R_set.append(R0)
    C_set.append(C_corr)
    R_set.append(R_corr)

    for img_id in image_ids:
        if img_id == image1_id or img_id == image2_id:
            continue

        combination_key = str(image1_id) + "_" + str(img_id)
        if not no_log: 
            result_dir = os.path.join(results_dir, combination_key)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        print('Processing image pair: ', combination_key)

        feature_idx_i = np.where(filtered_world_coords[:, 0] & filtered_feature_flags[:, img_id-1])
        if len(feature_idx_i[0]) < 8:
            print("Got ", len(feature_idx_i), "common points between X and ", img_id, "image")
            continue

        inliers_1 = np.hstack((feature_x[feature_idx_i, image1_id-1].reshape(-1, 1), feature_y[feature_idx_i, image1_id-1].reshape(-1, 1)))
        inliers_id = np.hstack((feature_x[feature_idx_i, img_id-1].reshape(-1, 1), feature_y[feature_idx_i, img_id-1].reshape(-1, 1)))
        world_coords = all_world_coords[feature_idx_i, :].reshape(-1, 3)

        C_new, R_new = PnPRANSAC(world_coords, inliers_id, K, threshold=200, n_max=1000)

        init_world_coords_new = linear_triangulation(K, C0, R0, C_new, R_new, inliers_1, inliers_id)
        corr_world_coords_new = nonlinear_triangulation(K, C0, R0, C_new, R_new, inliers_1, inliers_id, init_world_coords_new)
        # if not no_log:
        #     plot_world_coords([corr_world_coords_new], save_path=result_dir+'/world_coords_new.png', hold=True)

        C_new_corr, R_new_corr = nonlinear_PnP(K, C_new, R_new, inliers_id, corr_world_coords_new)
        C_new_corr, R_new_corr = nonlinear_PnP(K, C_new, R_new, inliers_id, world_coords)
        camera_poses[img_id] = [C_new_corr, R_new_corr]

        init_world_coords_new_corr = linear_triangulation(K, C0, R0, C_new_corr, R_new_corr, inliers_1, inliers_id)
        refined_world_coords_new = nonlinear_triangulation(K, C0, R0, C_new_corr, R_new_corr, inliers_1, inliers_id, init_world_coords_new_corr)
        if not no_log:
            plot_world_coords([refined_world_coords_new], save_path=result_dir+'/refined_world_coords_new.png', hold=True)

            plot_camera_pose(C_new_corr, R_new_corr, img_id, save_path=result_dir+'/with_camera_pose.png')

        R_set.append(R_new_corr)
        C_set.append(C_new_corr)

        for _img in range(1, img_id):
            world_coords_idx = np.where(filtered_feature_flags[:, _img-1] & filtered_feature_flags[:, img_id-1])
            world_coords_idx = np.squeeze(np.asarray(world_coords_idx))

            _key = str(_img) + "_" + str(img_id)
            if not no_log: 
                result_dir = os.path.join(results_dir, _key)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
            [C_img, R_img] = camera_poses[_img]

            if len(world_coords_idx) < 8:
                print("Got ", len(world_coords_idx), "common points between X and ", img_id, "image")
                continue

            _img1_inliers = np.hstack((feature_x[world_coords_idx, _img-1].reshape(-1, 1), feature_y[world_coords_idx, _img-1].reshape(-1, 1)))
            _img2_inliers = np.hstack((feature_x[world_coords_idx, img_id-1].reshape(-1, 1), feature_y[world_coords_idx, img_id-1].reshape(-1, 1)))

            _init_world_coords = linear_triangulation(K, C_img, R_img, C_new_corr, R_new_corr, _img1_inliers, _img2_inliers)
            _refined_world_coords = nonlinear_triangulation(K, C_img, R_img, C_new_corr, R_new_corr, _img1_inliers, _img2_inliers, _init_world_coords)
            if not no_log:
                plot_world_coords([_refined_world_coords], save_path=result_dir+'/refined_world_coords_.png', hold=True)

                plot_camera_pose(C_new_corr, R_new_corr, img_id, save_path=result_dir+'/with_camera_pose_.png', hold=True)

            print('Number of world points added: ', len(_refined_world_coords))

            all_world_coords[world_coords_idx] = _refined_world_coords
            all_world_coords_2[world_coords_idx] = _refined_world_coords
            filtered_world_coords[world_coords_idx] = 1

            print( 'Performing Bundle Adjustment  for image : ', img_id)
            R_set, C_set, all_world_coords = perform_bundle_adjustment(all_world_coords, filtered_world_coords, feature_x, feature_y,
                                                                        filtered_feature_flags, R_set, C_set, K, img_id-1)

            if not no_log:
                plot_world_coords([all_world_coords], save_path=result_dir+'/BA.png', hold=True)

                for i in range(img_id):
                    plot_camera_pose(C_set[i], R_set[i], i, save_path=result_dir+'/BA_with_camera_pose.png', hold=True)


    plot_world_coords([all_world_coords_2], save_path=os.path.join(results_dir, 'before_BA.png'), color='r', hold=True)
    plot_world_coords([all_world_coords], save_path=os.path.join(results_dir, 'BA.png'), color='b', hold=True)

    for i in range(len(C_set)):
        plot_camera_pose(C_set[i], R_set[i], i+1, save_path=os.path.join(results_dir, 'BA_with_camera_pose.png'), hold=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="P3Data/", help="Path of input images and feature matches text files")
    parser.add_argument("--results_dir", type=str, default="P3Data/Results", help="Directory to save results")
    args = parser.parse_args()

    sfm_wrapper(args.data_path, args.results_dir)