# Data Processing Pipeline: Dataloader → Forward Pass

This document explains how training data flows from `self.dataloader` through preprocessing steps before the NeRF model's forward pass.

---

## 1. Data from Dataloader

When you iterate through `self.dataloader`, each batch returns a dictionary with the following structure:

```python
batch = {
    'points': torch.Tensor,      # Shape: [batch_size, N_rays, 64, 3]
    'rays_d': torch.Tensor,       # Shape: [batch_size, N_rays, 3]
    'rgb_gt': torch.Tensor,       # Shape: [batch_size, N_rays, 3]
    'z_vals': torch.Tensor        # Shape: [batch_size, N_rays, 64]
}
```

**Example with `batch_size=1`:**
- `points`: `[1, 4096, 64, 3]` - 3D points along rays
- `rays_d`: `[1, 4096, 3]` - Ray directions in world space
- `rgb_gt`: `[1, 4096, 3]` - Ground truth RGB values
- `z_vals`: `[1, 4096, 64]` - Depth values along rays

---

## 2. Data Extraction and Reshaping

Before the forward pass, the data needs to be extracted and reshaped:

```python
# Extract from batch dictionary
points = batch['points']        # [batch_size, N_rays, 64, 3]
rays_d = batch['rays_d']       # [batch_size, N_rays, 3]
rgb_gt = batch['rgb_gt']       # [batch_size, N_rays, 3]
z_vals = batch['z_vals']        # [batch_size, N_rays, 64]

# Reshape for processing: flatten rays and samples
batch_size, N_rays, N_samples, _ = points.shape

# Flatten to process all points at once
points_flat = points.view(-1, 3)           # [batch_size * N_rays * N_samples, 3]
rays_d_flat = rays_d.unsqueeze(2).expand(-1, -1, N_samples, -1)  # [batch_size, N_rays, N_samples, 3]
rays_d_flat = rays_d_flat.view(-1, 3)      # [batch_size * N_rays * N_samples, 3]
```

**Key transformations:**
- **Points**: Flatten from `[batch_size, N_rays, 64, 3]` → `[batch_size * N_rays * 64, 3]`
- **Ray directions**: Expand and flatten to match points shape
- Each point along a ray needs its corresponding ray direction for view-dependent rendering

---

## 3. Positional Encoding

The NeRF model uses positional encoding to help the network learn high-frequency details. This happens in the `NeRFModel.forward()` method:

### 3.1 Position Encoding (for 3D points)

```python
# Input: points_flat [N_total_points, 3]
# Output: points_encoded [N_total_points, 63] (with pos_freqs=10)

points_encoded = self.pos_encoder(points_flat)
```

**Positional encoding formula:**
For each coordinate $(x, y, z)$ in a point:
$$
\text{encoded} = [x, y, z, \sin(\pi x), \cos(\pi x), \sin(2\pi x), \cos(2\pi x), \ldots, \sin(2^9\pi x), \cos(2^9\pi x), \ldots]
$$

With `pos_freqs=10` and `include_input=True`:
- Input dimensions: 3 (x, y, z)
- Frequencies: $2^0\pi, 2^1\pi, \ldots, 2^9\pi$ (10 frequencies)
- Each frequency has sin and cos: 2 × 10 = 20 components per coordinate
- Total: 3 × (1 + 20) = **63 dimensions**

**Code implementation:**
```python
# For each frequency level
for freq in [π, 2π, 4π, ..., 512π]:
    out.append(sin(x * freq))
    out.append(cos(x * freq))
    # Same for y and z
```

### 3.2 Direction Encoding (for ray directions)

```python
# Input: rays_d_flat [N_total_points, 3]
# Output: dirs_encoded [N_total_points, 27] (with dir_freqs=4)

dirs_encoded = self.dir_encoder(rays_d_flat)
```

**Direction encoding formula:**
For each direction component $(d_x, d_y, d_z)$:
$$
\text{encoded} = [d_x, d_y, d_z, \sin(\pi d_x), \cos(\pi d_x), \sin(2\pi d_x), \cos(2\pi d_x), \ldots]
$$

With `dir_freqs=4` and `include_input=True`:
- Input dimensions: 3 (dx, dy, dz)
- Frequencies: $2^0\pi, 2^1\pi, 2^2\pi, 2^3\pi$ (4 frequencies)
- Each frequency has sin and cos: 2 × 4 = 8 components per coordinate
- Total: 3 × (1 + 8) = **27 dimensions**

**Why normalize directions?**
Ray directions should be normalized before encoding:
```python
# Normalize ray directions to unit vectors
rays_d_normalized = F.normalize(rays_d_flat, p=2, dim=-1)
dirs_encoded = self.dir_encoder(rays_d_normalized)
```

---

## 4. Forward Pass Through NeRF Network

After encoding, the data flows through the NeRF network:

```python
# Forward pass
output = model(points_encoded, dirs_encoded)
# Output shape: [N_total_points, 4]
# Output contains: [RGB (3), sigma (1)]
```

### 4.1 Network Architecture Flow

```
Input: points_encoded [N, 63], dirs_encoded [N, 27]

1. Position Network (processes 3D location):
   - Layer 1-2: [63] → [256] → [256]
   - Layer 3: [256 + 63] → [256]  (residual connection)
   - Layer 4: [256] → [256]
   - Layer 5: [256 + 63] → [256]  (residual connection)
   - Layer 6: [256] → [256]
   - Layer 7: [256 + 63] → [256]  (residual connection)
   - Layer 8: [256] → [256]
   - Sigma: [256] → [1]  (density prediction)

2. Direction Network (processes view direction):
   - Concatenate: [256 (from position) + 27 (encoded dir)] → [283]
   - Layer 1: [283] → [128]
   - Layer 2: [128] → [128]
   - RGB: [128] → [3]  (color prediction, sigmoid activated)

Output: [RGB (3), sigma (1)] = [4]
```

### 4.2 Output Interpretation

For each 3D point along each ray:
- **RGB**: Predicted color `[r, g, b]` in range [0, 1] (sigmoid output)
- **Sigma (σ)**: Predicted density/opacity (can be any real value)

---

## 5. Reshaping Output for Volume Rendering

After the forward pass, the output needs to be reshaped back to ray structure:

```python
# Output from model: [batch_size * N_rays * N_samples, 4]
output_flat = model(points_encoded, dirs_encoded)  # [N_total, 4]

# Split RGB and sigma
rgb_pred = output_flat[:, :3]    # [N_total, 3]
sigma = output_flat[:, 3:4]     # [N_total, 1]

# Reshape back to ray structure
rgb_pred = rgb_pred.view(batch_size, N_rays, N_samples, 3)  # [B, R, S, 3]
sigma = sigma.view(batch_size, N_rays, N_samples, 1)        # [B, R, S, 1]
```

---

## 6. Complete Processing Pipeline

Here's the complete flow in code form:

```python
# Step 1: Get batch from dataloader
for batch in dataloader:
    points = batch['points']      # [B, R, S, 3]
    rays_d = batch['rays_d']       # [B, R, 3]
    rgb_gt = batch['rgb_gt']       # [B, R, 3]
    z_vals = batch['z_vals']       # [B, R, S]
    
    B, R, S, _ = points.shape
    
    # Step 2: Flatten for batch processing
    points_flat = points.view(-1, 3)                    # [B*R*S, 3]
    
    # Step 3: Expand and flatten ray directions
    rays_d_expanded = rays_d.unsqueeze(2).expand(-1, -1, S, -1)  # [B, R, S, 3]
    rays_d_flat = rays_d_expanded.view(-1, 3)           # [B*R*S, 3]
    
    # Step 4: Normalize ray directions (important!)
    rays_d_normalized = F.normalize(rays_d_flat, p=2, dim=-1)
    
    # Step 5: Positional encoding
    points_encoded = model.pos_encoder(points_flat)      # [B*R*S, 63]
    dirs_encoded = model.dir_encoder(rays_d_normalized) # [B*R*S, 27]
    
    # Step 6: Forward pass through NeRF
    output = model.nerf(points_encoded, dirs_encoded)  # [B*R*S, 4]
    
    # Step 7: Split and reshape
    rgb_pred = output[:, :3].view(B, R, S, 3)           # [B, R, S, 3]
    sigma = output[:, 3:4].view(B, R, S, 1)             # [B, R, S, 1]
    
    # Step 8: Volume rendering (happens next, not in forward pass)
    # This would use rgb_pred, sigma, and z_vals to compute final pixel colors
```

---

## 7. Key Points to Remember

1. **Batch Processing**: All points from all rays are processed in parallel for efficiency
2. **Positional Encoding**: Expands 3D coordinates from 3D → 63D to help learn high-frequency details
3. **View Dependence**: Ray directions are encoded separately (3D → 27D) and concatenated later for view-dependent color
4. **Residual Connections**: The position network uses skip connections (concatenating input at layers 3, 5, 7)
5. **Output Shape**: Each point gets [RGB, sigma] = [4] values
6. **Volume Rendering**: The actual rendering (combining samples along rays) happens after the forward pass using the predicted RGB and sigma values

---

## 8. Shape Summary Table

| Stage | Variable | Shape | Description |
|-------|----------|-------|-------------|
| **Dataloader Output** | `points` | `[B, R, S, 3]` | 3D points along rays |
| | `rays_d` | `[B, R, 3]` | Ray directions |
| | `rgb_gt` | `[B, R, 3]` | Ground truth colors |
| **After Flattening** | `points_flat` | `[B*R*S, 3]` | Flattened points |
| | `rays_d_flat` | `[B*R*S, 3]` | Flattened directions |
| **After Encoding** | `points_encoded` | `[B*R*S, 63]` | Position encoded |
| | `dirs_encoded` | `[B*R*S, 27]` | Direction encoded |
| **Model Output** | `output` | `[B*R*S, 4]` | [RGB, sigma] |
| **After Reshape** | `rgb_pred` | `[B, R, S, 3]` | Predicted RGB |
| | `sigma` | `[B, R, S, 1]` | Predicted density |

Where:
- `B` = batch_size (typically 1)
- `R` = N_rays (4096)
- `S` = N_samples (64)

---

## 9. Example with Concrete Numbers

Assuming `batch_size=1`:

1. **Dataloader**: Returns 1 image with 4096 rays, 64 samples each
2. **Flatten**: 1 × 4096 × 64 = **262,144 points** to process
3. **Encoding**: Each point expands from 3D → 63D (position) and 3D → 27D (direction)
4. **Forward Pass**: Process all 262,144 points through the network in parallel
5. **Output**: Get RGB and sigma predictions for all points
6. **Volume Rendering**: Combine 64 samples per ray to get final pixel colors (262,144 → 4,096)

This parallel processing is what makes NeRF training efficient!
