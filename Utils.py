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