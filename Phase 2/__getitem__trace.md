# Trace of `__getitem__` Function for One Sample Image

## Input
- `idx = 0` (first image in the dataset)
- Assume image dimensions: H=800, W=800 (typical NeRF synthetic dataset)

## Step-by-Step Execution

### Step 1: Get label for the frame
```python
label = self.labels['frames'][idx]
```
**Output:**
```python
label = {
    'file_path': './train/r_0',
    'rotation': 0.012566370614359171,
    'transform_matrix': [
        [-0.04659058526158333, 0.17529651522636414, -0.9834126234054565, -3.9642629623413086],
        [-0.9989140629768372, -0.008176046423614025, 0.04586757719516754, 0.18489810824394226],
        [-9.313225746154785e-10, 0.984481692314148, 0.17548708617687225, 0.7074110507965088],
        [0.0, 0.0, 0.0, 1.0]
    ]
}
```

### Step 2: Extract filename
```python
file_name = os.path.basename(label['file_path']) + '.png'
```
**Output:**
```python
file_name = 'r_0.png'
```

### Step 3: Construct image path
```python
img_path = os.path.join(self.path_to_images, file_name)
```
**Output:**
```python
img_path = '/path/to/Phase 2/nerf_synthetic/ship/train/r_0.png'
```

### Step 4: Load and transform image
```python
image = Image.open(img_path).convert("RGB")  # PIL Image, shape: (800, 800, 3)
image = self.transform(image)  # transforms.ToTensor()
```
**Output:**
```python
image.shape = torch.Size([3, 800, 800])  # [C, H, W]
# Values: float32 tensor with RGB values in range [0, 1]
```

### Step 5: Set constants
```python
N_rays = 4096
H, W = image.shape[1], image.shape[2]  # H=800, W=800
```
**Output:**
```python
N_rays = 4096
H = 800
W = 800
```

### Step 6: Calculate focal length
```python
if self.camera_angle_x is not None:
    focal = W / (2 * np.tan(self.camera_angle_x / 2))
else:
    focal = W / 2
```
**Given:** `camera_angle_x = 0.6911112070083618`
**Output:**
```python
focal = 800 / (2 * tan(0.6911112070083618 / 2))
focal ≈ 577.35  # (typical NeRF focal length for 800x800 images)
```

### Step 7: Sample random pixel coordinates
```python
i = torch.randint(0, W, (N_rays,))  # Random x-coordinates
j = torch.randint(0, H, (N_rays,))  # Random y-coordinates
```
**Output:**
```python
i.shape = torch.Size([4096])  # Random integers in range [0, 799]
j.shape = torch.Size([4096])  # Random integers in range [0, 799]
# Example values: i = [234, 567, 123, ...], j = [456, 789, 234, ...]
```

### Step 8: Extract RGB ground truth
```python
rgb_gt = image[:, j, i].permute(1, 0)  # [N_rays, 3]
```
**Output:**
```python
rgb_gt.shape = torch.Size([4096, 3])
# Each row contains RGB values for one sampled pixel
# Values in range [0, 1] (float32)
# Example: rgb_gt[0] = [0.85, 0.72, 0.61]  # RGB for pixel (i[0], j[0])
```

### Step 9: Convert pixel coordinates to normalized camera coordinates
```python
x = (i.float() - W * 0.5) / focal
y = (j.float() - H * 0.5) / focal
z = -torch.ones_like(x)
```
**Output:**
```python
x.shape = torch.Size([4096])  # Normalized x coordinates (centered at 0)
y.shape = torch.Size([4096])  # Normalized y coordinates (centered at 0)
z.shape = torch.Size([4096])  # All values = -1.0
# Example: x[0] ≈ (234 - 400) / 577.35 ≈ -0.287
#          y[0] ≈ (456 - 400) / 577.35 ≈ 0.097
#          z[0] = -1.0
```

### Step 10: Create ray directions in camera space
```python
dirs = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]
```
**Output:**
```python
dirs.shape = torch.Size([4096, 3])
# Each row is a ray direction vector in camera space
# Example: dirs[0] = [-0.287, 0.097, -1.0]
```

### Step 11: Get camera-to-world transformation matrix
```python
c2w = torch.tensor(label['transform_matrix'], dtype=torch.float32)  # [4, 4]
```
**Output:**
```python
c2w.shape = torch.Size([4, 4])
c2w = [
    [-0.0466,  0.1753, -0.9834, -3.9643],
    [-0.9989, -0.0082,  0.0459,  0.1849],
    [ 0.0000,  0.9845,  0.1755,  0.7074],
    [ 0.0000,  0.0000,  0.0000,  1.0000]
]
```

### Step 12: Transform ray directions to world space
```python
rays_d = (dirs @ c2w[:3, :3].T).float()  # Rotate ray directions
```
**Output:**
```python
rays_d.shape = torch.Size([4096, 3])
# Ray directions in world coordinate system
# Each row is a normalized direction vector (approximately)
# Example: rays_d[0] = [some rotated direction vector]
```

### Step 13: Get ray origins (camera position in world space)
```python
rays_o = c2w[:3, 3].expand(rays_d.shape)  # [N_rays, 3]
```
**Output:**
```python
rays_o.shape = torch.Size([4096, 3])
# All rows are the same: camera position in world space
rays_o = [[-3.9643, 0.1849, 0.7074],  # Same for all 4096 rays
          [-3.9643, 0.1849, 0.7074],
          ...
          [-3.9643, 0.1849, 0.7074]]
```

### Step 14: Define near and far planes
```python
near, far = 2.0, 6.0
```
**Output:**
```python
near = 2.0
far = 6.0
```

### Step 15: Create depth sampling points
```python
t_vals = torch.linspace(0., 1., steps=64)
z_vals = near * (1. - t_vals) + far * t_vals  # [64]
z_vals = z_vals.expand(N_rays, -1)  # [N_rays, 64]
```
**Output:**
```python
t_vals.shape = torch.Size([64])
# t_vals = [0.0, 0.0159, 0.0317, ..., 1.0]

z_vals.shape = torch.Size([4096, 64])
# z_vals[0] = [2.0, 2.0635, 2.1270, ..., 6.0]  # Depth values along ray
# All 4096 rays have the same depth sampling initially
```

### Step 16: Hierarchical sampling (add randomness)
```python
mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
upper = torch.cat([mids, z_vals[..., -1:]], -1)
lower = torch.cat([z_vals[..., :1], mids], -1)
t_rand = torch.rand_like(z_vals)
z_vals = lower + (upper - lower) * t_rand
```
**Output:**
```python
z_vals.shape = torch.Size([4096, 64])
# Now each ray has slightly different depth sampling (hierarchical sampling)
# Values still in range [near=2.0, far=6.0]
# Example: z_vals[0, 0] ≈ 2.0, z_vals[0, 31] ≈ 4.0, z_vals[0, 63] ≈ 6.0
```

### Step 17: Sample 3D points along rays
```python
points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [N_rays, 64, 3]
```
**Output:**
```python
points.shape = torch.Size([4096, 64, 3])
# 3D points sampled along each ray
# points[i, j, :] = 3D coordinates of j-th sample point along i-th ray
# Example: 
#   points[0, 0, :] = rays_o[0] + rays_d[0] * z_vals[0, 0]  # Point near camera
#   points[0, 63, :] = rays_o[0] + rays_d[0] * z_vals[0, 63]  # Point far from camera
```

## Final Return Dictionary

```python
return {
    'points': points,      # Shape: [4096, 64, 3] - 3D points along rays
    'rays_d': rays_d,      # Shape: [4096, 3] - Ray directions in world space
    'rgb_gt': rgb_gt,      # Shape: [4096, 3] - Ground truth RGB for sampled pixels
    'z_vals': z_vals       # Shape: [4096, 64] - Depth values along each ray
}
```

## Summary

For **one image** (idx=0):
- **4096 rays** are randomly sampled from the image
- Each ray has **64 depth samples** (points along the ray)
- Total **3D points**: 4096 × 64 = 262,144 points
- Each point will be fed to the NeRF network to predict RGB and density
- The predicted RGB is compared with `rgb_gt` for training

## Key Variables Summary Table

| Variable | Shape | Description |
|----------|-------|-------------|
| `image` | [3, 800, 800] | Original image tensor |
| `i`, `j` | [4096] | Random pixel coordinates |
| `rgb_gt` | [4096, 3] | Ground truth RGB for sampled pixels |
| `rays_o` | [4096, 3] | Ray origins (camera position) |
| `rays_d` | [4096, 3] | Ray directions in world space |
| `z_vals` | [4096, 64] | Depth sampling values |
| `points` | [4096, 64, 3] | 3D points along rays |
| `c2w` | [4, 4] | Camera-to-world transformation |

