# Mathematical Equations in `__getitem__` Function

This document contains all the mathematical equations used to compute the final return values in the `__getitem__` function.

---

## 1. Focal Length Calculation

The focal length is computed from the camera's field of view (FOV):

\[
f = \frac{W}{2 \tan(\theta/2)}
\]

where:
- \( f \) = focal length
- \( W \) = image width (in pixels)
- \( \theta \) = `camera_angle_x` (horizontal field of view in radians)

**Fallback case** (if `camera_angle_x` is not available):
\[
f = \frac{W}{2}
\]

**Code equivalent:**
```python
focal = W / (2 * np.tan(self.camera_angle_x / 2))
```

---

## 2. Random Pixel Coordinate Sampling

Random pixel coordinates are sampled uniformly from the image:

\[
i \sim \text{Uniform}(0, W-1) \quad \text{for } N_{\text{rays}} \text{ samples}
\]
\[
j \sim \text{Uniform}(0, H-1) \quad \text{for } N_{\text{rays}} \text{ samples}
\]

where:
- \( i \) = x-coordinate (column) in pixel space
- \( j \) = y-coordinate (row) in pixel space
- \( H \) = image height
- \( W \) = image width
- \( N_{\text{rays}} = 4096 \)

**Code equivalent:**
```python
i = torch.randint(0, W, (N_rays,))
j = torch.randint(0, H, (N_rays,))
```

---

## 3. Ground Truth RGB Extraction

RGB values are extracted from the image at sampled pixel locations:

\[
\text{rgb}_{\text{gt}}[k, c] = I[c, j[k], i[k]]
\]

where:
- \( I \) = image tensor of shape \([C, H, W]\) (C=3 for RGB)
- \( k \in \{0, 1, \ldots, N_{\text{rays}}-1\} \) = ray index
- \( c \in \{0, 1, 2\} \) = color channel (R, G, B)

**Output shape:** \([N_{\text{rays}}, 3]\)

**Code equivalent:**
```python
rgb_gt = image[:, j, i].permute(1, 0)  # [N_rays, 3]
```

---

## 4. Normalized Camera Coordinates

Pixel coordinates are converted to normalized camera coordinates (centered at origin):

\[
x = \frac{i - W/2}{f}
\]
\[
y = \frac{j - H/2}{f}
\]
\[
z = -1
\]

where:
- \( (x, y, z) \) = normalized camera coordinates
- The negative z-direction points into the scene (camera looks along -z)

**Code equivalent:**
```python
x = (i.float() - W * 0.5) / focal
y = (j.float() - H * 0.5) / focal
z = -torch.ones_like(x)
```

---

## 5. Ray Direction in Camera Space

Ray directions in camera coordinate system:

\[
\mathbf{d}_{\text{cam}} = \begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} \frac{i - W/2}{f} \\ \frac{j - H/2}{f} \\ -1 \end{bmatrix}
\]

**Output shape:** \([N_{\text{rays}}, 3]\)

**Code equivalent:**
```python
dirs = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]
```

---

## 6. Camera-to-World Transformation Matrix

The camera-to-world transformation matrix is extracted from the label:

\[
\mathbf{T}_{\text{c2w}} = \begin{bmatrix}
R_{00} & R_{01} & R_{02} & t_x \\
R_{10} & R_{11} & R_{12} & t_y \\
R_{20} & R_{21} & R_{22} & t_z \\
0 & 0 & 0 & 1
\end{bmatrix} = \begin{bmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
\]

where:
- \( \mathbf{R} \in \mathbb{R}^{3 \times 3} \) = rotation matrix
- \( \mathbf{t} \in \mathbb{R}^{3} \) = translation vector (camera position in world space)

**Code equivalent:**
```python
c2w = torch.tensor(label['transform_matrix'], dtype=torch.float32)  # [4, 4]
```

---

## 7. Ray Direction Transformation to World Space

Ray directions are transformed from camera space to world space using the rotation matrix:

\[
\mathbf{d}_{\text{world}} = \mathbf{R}^T \mathbf{d}_{\text{cam}}
\]

In matrix form:
\[
\begin{bmatrix} d_{\text{world},x} \\ d_{\text{world},y} \\ d_{\text{world},z} \end{bmatrix} = \begin{bmatrix}
R_{00} & R_{10} & R_{20} \\
R_{01} & R_{11} & R_{21} \\
R_{02} & R_{12} & R_{22}
\end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix}
\]

**Output shape:** \([N_{\text{rays}}, 3]\)

**Code equivalent:**
```python
rays_d = (dirs @ c2w[:3, :3].T).float()
```

---

## 8. Ray Origin (Camera Position in World Space)

The ray origin is the camera position in world coordinates:

\[
\mathbf{o} = \mathbf{t} = \begin{bmatrix} t_x \\ t_y \\ t_z \end{bmatrix}
\]

This is the same for all rays from the same camera:
\[
\mathbf{o}_k = \mathbf{t} \quad \forall k \in \{0, 1, \ldots, N_{\text{rays}}-1\}
\]

**Output shape:** \([N_{\text{rays}}, 3]\) (same value repeated)

**Code equivalent:**
```python
rays_o = c2w[:3, 3].expand(rays_d.shape)  # [N_rays, 3]
```

---

## 9. Linear Depth Sampling

Depth values are sampled linearly between near and far planes:

\[
t \in [0, 1] \quad \text{(uniformly spaced)}
\]
\[
z = z_{\text{near}} (1 - t) + z_{\text{far}} \cdot t
\]

where:
- \( z_{\text{near}} = 2.0 \)
- \( z_{\text{far}} = 6.0 \)
- \( t \) = 64 uniformly spaced values from 0 to 1

**Explicit form:**
\[
z = z_{\text{near}} + (z_{\text{far}} - z_{\text{near}}) \cdot t
\]

**Output shape:** \([N_{\text{rays}}, 64]\) (same depth values for all rays initially)

**Code equivalent:**
```python
t_vals = torch.linspace(0., 1., steps=64)
z_vals = near * (1. - t_vals) + far * t_vals  # [64]
z_vals = z_vals.expand(N_rays, -1)  # [N_rays, 64]
```

---

## 10. Hierarchical Sampling (Stratified Sampling)

To improve sampling efficiency, depth values are perturbed using hierarchical/stratified sampling:

### Step 10.1: Compute Midpoints

\[
m_i = \frac{z_i + z_{i+1}}{2} \quad \text{for } i \in \{0, 1, \ldots, N_{\text{samples}}-2\}
\]

where \( N_{\text{samples}} = 64 \)

### Step 10.2: Define Upper and Lower Bounds

\[
\text{lower}_i = \begin{cases}
z_0 & \text{if } i = 0 \\
m_{i-1} & \text{if } i > 0
\end{cases}
\]

\[
\text{upper}_i = \begin{cases}
m_i & \text{if } i < N_{\text{samples}}-1 \\
z_{N_{\text{samples}}-1} & \text{if } i = N_{\text{samples}}-1
\end{cases}
\]

### Step 10.3: Random Perturbation

\[
u \sim \text{Uniform}(0, 1) \quad \text{(random value for each sample)}
\]

\[
z_{\text{new},i} = \text{lower}_i + (\text{upper}_i - \text{lower}_i) \cdot u_i
\]

This ensures that each sample is randomly distributed within its bin, improving coverage.

**Output shape:** \([N_{\text{rays}}, 64]\) (now different for each ray)

**Code equivalent:**
```python
mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
upper = torch.cat([mids, z_vals[..., -1:]], -1)
lower = torch.cat([z_vals[..., :1], mids], -1)
t_rand = torch.rand_like(z_vals)
z_vals = lower + (upper - lower) * t_rand
```

---

## 11. 3D Point Sampling Along Rays

3D points are sampled along each ray using the parametric ray equation:

\[
\mathbf{p}_{k,i} = \mathbf{o}_k + z_{k,i} \cdot \mathbf{d}_{\text{world},k}
\]

In component form:
\[
\begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix}_{k,i} = \begin{bmatrix} o_x \\ o_y \\ o_z \end{bmatrix}_k + z_{k,i} \begin{bmatrix} d_x \\ d_y \\ d_z \end{bmatrix}_{\text{world},k}
\]

where:
- \( k \in \{0, 1, \ldots, N_{\text{rays}}-1\} \) = ray index
- \( i \in \{0, 1, \ldots, 63\} \) = sample point index along the ray
- \( \mathbf{p}_{k,i} \) = 3D point coordinates in world space
- \( z_{k,i} \) = depth value along the ray

**Output shape:** \([N_{\text{rays}}, 64, 3]\)

**Code equivalent:**
```python
points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
```

---

## Summary: Complete Pipeline

The complete mathematical pipeline can be written as:

1. **Sample pixels:** \( (i, j) \sim \text{Uniform}(0, W-1) \times \text{Uniform}(0, H-1) \)

2. **Normalize coordinates:** 
   \[
   (x, y, z) = \left( \frac{i - W/2}{f}, \frac{j - H/2}{f}, -1 \right)
   \]

3. **Transform to world space:**
   \[
   \mathbf{d}_{\text{world}} = \mathbf{R}^T \begin{bmatrix} x \\ y \\ z \end{bmatrix}
   \]
   \[
   \mathbf{o} = \mathbf{t}
   \]

4. **Sample depths:**
   \[
   z_i = z_{\text{near}} + (z_{\text{far}} - z_{\text{near}}) \cdot t_i + \text{perturbation}
   \]

5. **Compute 3D points:**
   \[
   \mathbf{p}_i = \mathbf{o} + z_i \mathbf{d}_{\text{world}}
   \]

---

## Final Output Dictionary

\[
\text{return} = \begin{cases}
\text{points} & \in \mathbb{R}^{N_{\text{rays}} \times 64 \times 3} \\
\text{rays\_d} & \in \mathbb{R}^{N_{\text{rays}} \times 3} \\
\text{rgb}_{\text{gt}} & \in [0, 1]^{N_{\text{rays}} \times 3} \\
\text{z\_vals} & \in \mathbb{R}^{N_{\text{rays}} \times 64}
\end{cases}
\]

where \( N_{\text{rays}} = 4096 \) and depth samples = 64.

