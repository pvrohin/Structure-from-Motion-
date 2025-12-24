import numpy as np
from itertools import combinations
from Utils import get_data

# Load data
data_path = "./P3Data"
no_of_images = 5
feature_x, feature_y, feature_flag = get_data(data_path, no_of_images)

print("=" * 80)
print("SAMPLE OUTPUT FOR ONE FEATURE COMBINATION")
print("=" * 80)
print(f"\nFeature data shapes:")
print(f"  feature_x.shape: {feature_x.shape}")
print(f"  feature_y.shape: {feature_y.shape}")
print(f"  feature_flag.shape: {feature_flag.shape}")

# Create all combinations
feature_combinations = list(combinations(range(1, 6), 2))
print(f"\nAll combinations: {feature_combinations}")

# Process first combination only
combination = feature_combinations[0]  # (1, 2)
image1_id, image2_id = combination
combination_key = str(image1_id) + "_" + str(image2_id)

print("\n" + "=" * 80)
print(f"PROCESSING COMBINATION: {combination_key}")
print("=" * 80)
print(f"Image 1 ID: {image1_id} (0-based index: {image1_id-1})")
print(f"Image 2 ID: {image2_id} (0-based index: {image2_id-1})")

# Find matching features
_idx = np.where(feature_flag[:, image1_id-1] & feature_flag[:, image2_id-1])
print(f"\nStep 1: Finding matching features")
print(f"  feature_flag[:, {image1_id-1}].shape: {feature_flag[:, image1_id-1].shape}")
print(f"  feature_flag[:, {image2_id-1}].shape: {feature_flag[:, image2_id-1].shape}")
print(f"  _idx type: {type(_idx)}")
print(f"  _idx: {_idx}")
print(f"  Number of matching features: {len(_idx[0])}")

# Extract coordinates for image 1
print(f"\nStep 2: Extracting coordinates for Image {image1_id}")
print(f"  feature_x[_idx, {image1_id-1}]:")
x1_raw = feature_x[_idx, image1_id-1]
print(f"    Shape: {x1_raw.shape}")
print(f"    First 10 values: {x1_raw.flatten()[:10]}")
print(f"  feature_y[_idx, {image1_id-1}]:")
y1_raw = feature_y[_idx, image1_id-1]
print(f"    Shape: {y1_raw.shape}")
print(f"    First 10 values: {y1_raw.flatten()[:10]}")

image1_coords_org = np.hstack((feature_x[_idx, image1_id-1].reshape((-1,1)), feature_y[_idx, image1_id-1].reshape((-1,1))))
print(f"\n  image1_coords_org:")
print(f"    Shape: {image1_coords_org.shape}")
print(f"    First 5 rows:")
print(f"    {image1_coords_org[:5]}")

# Extract coordinates for image 2
print(f"\nStep 3: Extracting coordinates for Image {image2_id}")
image2_coords_org = np.hstack((feature_x[_idx, image2_id-1].reshape((-1,1)), feature_y[_idx, image2_id-1].reshape((-1,1))))
print(f"  image2_coords_org:")
print(f"    Shape: {image2_coords_org.shape}")
print(f"    First 5 rows:")
print(f"    {image2_coords_org[:5]}")

# Get feature indices
idx = np.array(_idx).reshape(-1)
print(f"\nStep 4: Feature indices")
print(f"  idx:")
print(f"    Shape: {idx.shape}")
print(f"    First 20 indices: {idx[:20]}")
print(f"    Last 10 indices: {idx[-10:]}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Combination: {combination_key}")
print(f"Total matching features: {len(idx)}")
print(f"image1_coords_org.shape: {image1_coords_org.shape}")
print(f"image2_coords_org.shape: {image2_coords_org.shape}")
print(f"idx.shape: {idx.shape}")
print(f"\nFirst 3 matching feature pairs:")
for i in range(min(3, len(idx))):
    print(f"  Feature {idx[i]}: Image1({image1_coords_org[i,0]:.2f}, {image1_coords_org[i,1]:.2f}) <-> Image2({image2_coords_org[i,0]:.2f}, {image2_coords_org[i,1]:.2f})")

