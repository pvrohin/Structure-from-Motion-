import numpy as np

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
