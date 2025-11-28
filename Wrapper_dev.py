import os
import argparse
import re
from Utils import *

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
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="P3Data/", help="Path of input images and feature matches text files")
    parser.add_argument("--results_dir", type=str, default="P3Data/Results", help="Directory to save results")
    args = parser.parse_args()

    sfm_wrapper(args.data_path, args.results_dir)