# Technical Architecture: Independent Incremental 3D Scene Reconstruction Studio Using Gaussian Splatting

## 1. Gaussian Parameter Representation and Memory Layout

In 3D Gaussian Splatting, each 3D Gaussian $G_i$ is parameterized by:
- **Mean Position $\boldsymbol{\mu} \in \mathbb{R}^3$**: 3D world-space centroid $(x, y, z)$.
- **Scale $\boldsymbol{s} \in \mathbb{R}^3$**: Log-space scale vector $(\ln s_x, \ln s_y, \ln s_z) \in \mathbb{R}^3$, enforcing positive scaling through $s_j = \exp(\ln s_j)$.
- **Rotation $\boldsymbol{q} \in \mathbb{R}^4$**: Unit quaternion $(q_r, q_x, q_y, q_z)$ or $(w, x, y, z)$ normalized to $\|\boldsymbol{q}\| = 1$. The corresponding rotation matrix $R \in \mathrm{SO}(3)$ transforms local axes to world space.
- **Opacity $\sigma \in \mathbb{R}$**: Unconstrained logit-space parameter transformed via the logistic sigmoid $\alpha = \frac{1}{1 + e^{-\sigma}} \in [0, 1]$.
- **Spherical Harmonics (SH) $\boldsymbol{c}_{sh}$**: View-dependent color representation. For degree $D=0$, there are $1$ coefficient per color channel ($3$ floats total: base RGB). For degree $D=3$, there are $(D+1)^2 = 16$ coefficients per channel ($16 \times 3 = 48$ floats total).

### Memory Organization: Structure-of-Arrays (SoA) vs Array-of-Structures (AoS)

To satisfy WebGPU memory alignment, compute shader coalesced memory access, and granular optimizer memory access:
- **AoS (Array of Structures)** introduces awkward padding (e.g., vec3 requires 16-byte alignment in WGSL struct definitions) and causes substantial memory bandwidth waste during rendering passes that only require position and conic parameters.
- **SoA (Structure of Arrays)** is selected for the primary GPU storage buffers:
  1. `positions`: `array<f32>` (3 floats per Gaussian: $x, y, z$, packed consecutively).
  2. `scales`: `array<f32>` (3 floats per Gaussian: $\ln s_x, \ln s_y, \ln s_z$).
  3. `rotations`: `array<vec4<f32>>` (4 floats per Gaussian: $w, x, y, z$).
  4. `opacities`: `array<f32>` (1 float per Gaussian: $\sigma$).
  5. `sh_coeffs`: `array<f32>` ($3 \times 16 = 48$ floats per Gaussian for $D=3$, or degree 0 with 3 floats).

For rasterization runtime performance, an intermediate compacted render buffer is generated during the projection pass:
```rust
struct ProjectedGaussian {
    screen_pos: vec2<f32>,     // 8 bytes (x, y in pixels)
    depth: f32,                // 4 bytes (camera space z)
    radius: f32,               // 4 bytes (bounding radius in pixels)
    conic: vec3<f32>,          // 12 bytes (inv_cov_xx, inv_cov_xy, inv_cov_yy)
    color: vec4<f32>,          // 16 bytes (r, g, b, alpha)
    flags: u32,                // 4 bytes
}
```

---

## 2. GPU Buffer Architecture

WebGPU enforces rigorous buffer binding rules and limits (`maxStorageBufferBindingSize` default 128 MB, configurable up to 1 GB+ on modern desktop/Apple Silicon). The buffer architecture is divided into three tiers:

```
+-----------------------------------------------------------------------------------+
|                              GPU BUFFER HIERARCHY                                 |
+-----------------------------------------------------------------------------------+
|  1. MODEL STATE BUFFERS (Global / Training / Storage)                            |
|     - Positions Buffer (vec3<f32>)                                                |
|     - Scales Buffer (vec3<f32>)                                                   |
|     - Rotations Buffer (vec4<f32>)                                                |
|     - Opacities Buffer (f32)                                                      |
|     - SH Coefficients Buffer (48 x f32)                                           |
|     - Optimizer State: First Moment (m) & Second Moment (v) Buffers               |
|     - Gradient Accumulation Buffers                                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v (Projection Compute Shader)
+-----------------------------------------------------------------------------------+
|  2. INTERMEDIATE RENDER BUFFERS (Per-Frame / Screen Resolution Dependent)         |
|     - Projected Splat Buffer (compacted, 48 bytes / visible Gaussian)             |
|     - Radii / Visible Indices Buffer                                              |
|     - Tile Overlap Counts & Offsets Buffer                                        |
|     - Sort Keys (`tile_id[16] | depth[16]`) and Value Indices (u32)               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v (Radix Sort + Tile Binning)
+-----------------------------------------------------------------------------------+
|  3. TILE DISPATCH & COMPOSITING BUFFERS                                           |
|     - Tile Ranges Buffer (`array<vec2<u32>, num_tiles>`: start_idx, end_idx)      |
|     - Output Target Texture (Storage / Render Attachment: RGBA8Unorm / RGBA16Float)|
+-----------------------------------------------------------------------------------+
```

### Buffer Dynamic Reallocation Strategy
When adaptive density control (cloning, splitting, pruning) alters the number of active Gaussians $N$:
1. A geometric allocation factor ($1.5\times$ growth factor) is maintained to avoid continuous reallocations.
2. WebGPU storage buffers are resized only when capacity is exceeded or when active count drops below $50\%$ capacity after compaction.
3. Compaction utilizes an atomic prefix sum stream compaction compute shader on the GPU, avoiding CPU round-trips.

---

## 3. WebGPU Projection Pipeline

The projection pipeline transforms 3D world-space Gaussians into 2D screen-space elliptic splats:

### Mathematical Formulation
1. **World to View Transformation**:
   $$\boldsymbol{\mu}_c = W \boldsymbol{\mu}_w + \boldsymbol{t}$$
   where $W \in \mathrm{SO}(3)$ is the camera rotation matrix and $\boldsymbol{t} \in \mathbb{R}^3$ is translation.
   Frustum culling: if $\boldsymbol{\mu}_{c, z} \le z_{\text{near}}$ or falls outside frustum planes expanded by bounding radius, discard.

2. **3D Covariance Matrix $\Sigma \in \mathbb{R}^{3 \times 3}$**:
   From rotation quaternion $\boldsymbol{q} \to R$ and scaling $\boldsymbol{s} = \exp(\boldsymbol{s}_{\log})$, diagonal scale matrix $S = \operatorname{diag}(\boldsymbol{s})$:
   $$\Sigma = R S S^T R^T$$

3. **Affine Approximation of Projective Transformation**:
   The projective mapping $\boldsymbol{x} = \pi(\boldsymbol{p}) = \left( \frac{f_x p_x}{p_z} + c_x, \frac{f_y p_y}{p_z} + c_y \right)^T$.
   Its Jacobian evaluated at $\boldsymbol{\mu}_c = (t_x, t_y, t_z)^T$:
   $$J = \begin{bmatrix}
   \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\
   0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2}
   \end{bmatrix}$$

4. **2D Projected Covariance $\Sigma' \in \mathbb{R}^{2 \times 2}$**:
   $$\Sigma' = J W \Sigma W^T J^T$$
   To prevent aliasing and mathematical singular matrices when projected Gaussians are smaller than a single pixel, an anti-aliasing low-pass convolution filter is applied:
   $$\Sigma'_{00} \leftarrow \Sigma'_{00} + 0.3, \quad \Sigma'_{11} \leftarrow \Sigma'_{11} + 0.3$$

5. **Conic Computation & Eigenvalue Clamping**:
   $$\det(\Sigma') = \Sigma'_{00} \Sigma'_{11} - (\Sigma'_{01})^2$$
   If $\det(\Sigma') \le 10^{-7}$, cull splat.
   $$\Sigma'^{-1} = \frac{1}{\det(\Sigma')} \begin{bmatrix} \Sigma'_{11} & -\Sigma'_{01} \\ -\Sigma'_{01} & \Sigma'_{00} \end{bmatrix} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$$
   The eigenvalues of $\Sigma'$ determine the major and minor axes:
   $$\lambda_{1, 2} = \frac{(\Sigma'_{00} + \Sigma'_{11}) \pm \sqrt{(\Sigma'_{00} - \Sigma'_{11})^2 + 4(\Sigma'_{01})^2}}{2}$$
   Screen-space bounding radius: $r = \lceil 3.0 \sqrt{\max(\lambda_1, \lambda_2)} \rceil$.

---

## 4. Tile-Based Rasterization

The viewport of width $W$ and height $H$ is divided into non-overlapping tiles of $16 \times 16$ pixels.
- Number of tiles horizontally: $T_x = \lceil W / 16 \rceil$.
- Number of tiles vertically: $T_y = \lceil H / 16 \rceil$.
- Total tiles: $T = T_x \times T_y$.

```
Screen Viewport (W x H)
+----------+----------+----------+ ... +----------+
| Tile 0,0 | Tile 1,0 | Tile 2,0 |     |          |
+----------+----------+----------+ ... +----------+
| Tile 0,1 |  [Gaussian Splat]   |     |          |
|          |    intersects       |     |          |
|          |  multiple tiles     |     |          |
+----------+----------+----------+ ... +----------+
|   ...    |   ...    |   ...    |     |          |
+----------+----------+----------+ ... +----------+
```

### Tile Assignment Workflow
1. For each visible Gaussian $i$, determine its bounding box $[\boldsymbol{\mu}_{\text{screen}} - r, \boldsymbol{\mu}_{\text{screen}} + r]$.
2. Compute the intersected tile coordinates:
   $$\text{min\_tile\_x} = \max(0, \lfloor (\mu_x - r) / 16 \rfloor), \quad \text{max\_tile\_x} = \min(T_x - 1, \lfloor (\mu_x + r) / 16 \rfloor)$$
   $$\text{min\_tile\_y} = \max(0, \lfloor (\mu_y - r) / 16 \rfloor), \quad \text{max\_tile\_y} = \min(T_y - 1, \lfloor (\mu_y + r) / 16 \rfloor)$$
3. Count tile overlaps. A prefix sum over overlap counts determines storage offsets in the sort key-value list.
4. Duplicate keys into the sort buffer:
   $$\text{key} = (\text{tile\_id} \ll 32) \mid \text{as\_u32}(\text{depth})$$
   (represented in 32-bit WebGPU WGSL as packed pairs or two-level keys).

---

## 5. Parallel Depth Sorting

Sorting splats by depth within each tile is required for correct front-to-back alpha compositing.

### Sorting Implementation Strategy:
1. **Milestone 1**: Parallel CPU sorting / Hybrid Radix Sort dispatched to GPU using bitonic sorting for moderate counts to establish immediate baseline correctness and verification.
2. **Milestone 2**: Pure GPU Workgroup-level 8-bit Radix Sort:
   - Pass 1: Histogram generation per workgroup over 8-bit radix digits.
   - Pass 2: Global prefix sum over histogram buckets.
   - Pass 3: Scatter to reordered key/index buffers.
   - Repeat for 4 passes (32 bits).
3. **Tile Boundary Identification**:
   A lightweight compute pass evaluates sorted keys: whenever $\text{key}[k].\text{tile\_id} \ne \text{key}[k+1].\text{tile\_id}$, update $\text{tile\_ranges}[\text{tile\_id}].\text{end} = k+1$ and $\text{tile\_ranges}[\text{key}[k+1].\text{tile\_id}].\text{start} = k+1$.

---

## 6. Front-to-Back Alpha Compositing

Within each tile ($16 \times 16$ pixels), a compute workgroup of 256 threads ($16 \times 16$) processes pixels in parallel.

### Transmittance and Blending Formulation
Let sorted Gaussians in the tile be $g_1, g_2, \ldots, g_M$:
For each pixel $\boldsymbol{p} = (u, v)^T$:
1. Initial accumulated color $\boldsymbol{C} = \mathbf{0}$, initial transmittance $T = 1.0$.
2. Load batches of Gaussians into workgroup shared memory (`var<workgroup>`).
3. For each Gaussian $i$:
   - Distance from Gaussian center: $\Delta = \boldsymbol{p} - \boldsymbol{\mu}_{\text{screen}, i}$.
   - Mahalanobis distance squared:
     $$d^2 = \frac{1}{2} (a \Delta_x^2 + 2b \Delta_x \Delta_y + c \Delta_y^2)$$
   - If $d^2 < 0$, skip.
   - Splat alpha:
     $$\alpha_i = \min(0.99, \alpha_{\text{base}, i} \cdot \exp(-d^2))$$
   - If $\alpha_i < \frac{1}{255}$, skip.
   - Color accumulation:
     $$\boldsymbol{C} \leftarrow \boldsymbol{C} + \boldsymbol{c}_i \cdot \alpha_i \cdot T$$
     $$T \leftarrow T \cdot (1 - \alpha_i)$$
   - Early saturation stop: If $T < 0.0001$, terminate compositing for this pixel.
4. Background blending: $\boldsymbol{C}_{\text{final}} = \boldsymbol{C} + T \cdot \boldsymbol{C}_{\text{bg}}$.

---

## 7. Differentiable Backward Pass

Training requires calculating exact analytical gradients of the loss function $\mathcal{L}$ with respect to all parameters:
$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_w}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{s}}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{q}}, \quad \frac{\partial \mathcal{L}}{\partial \sigma}, \quad \frac{\partial \mathcal{L}}{\partial \boldsymbol{c}_{sh}}$$

### Backpropagation Chain Rule
1. **From Pixel Loss to Accumulated Splats**:
   $$\frac{\partial \mathcal{L}}{\partial \boldsymbol{c}_i} = \sum_{\boldsymbol{p} \in \text{splat}_i} \frac{\partial \mathcal{L}}{\partial \boldsymbol{C}(\boldsymbol{p})} \alpha_i(\boldsymbol{p}) T_i(\boldsymbol{p})$$
   $$\frac{\partial \mathcal{L}}{\partial \alpha_i(\boldsymbol{p})} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{C}(\boldsymbol{p})} \cdot \left( \boldsymbol{c}_i T_i(\boldsymbol{p}) - \frac{\sum_{j=i+1}^M \boldsymbol{c}_j \alpha_j(\boldsymbol{p}) T_j(\boldsymbol{p})}{1 - \alpha_i(\boldsymbol{p})} \right)$$
2. **From Alpha to Conic & 2D Position**:
   From $\alpha_i = \alpha_{\text{base}} \exp(-d^2)$:
   $$\frac{\partial \alpha_i}{\partial d^2} = -\alpha_i, \quad \frac{\partial \alpha_i}{\partial \sigma} = \alpha_i (1 - \alpha_i) \exp(-d^2)$$
   $$\frac{\partial d^2}{\partial \boldsymbol{\mu}_{\text{screen}}} = -\begin{bmatrix} a \Delta_x + b \Delta_y \\ b \Delta_x + c \Delta_y \end{bmatrix}, \quad \frac{\partial d^2}{\partial a} = \frac{1}{2}\Delta_x^2, \quad \frac{\partial d^2}{\partial b} = \Delta_x \Delta_y, \quad \frac{\partial d^2}{\partial c} = \frac{1}{2}\Delta_y^2$$
3. **From 2D Covariance to 3D Covariance and World Position**:
   Using $\Sigma' = J W \Sigma W^T J^T$ and $\boldsymbol{\mu}_c = W \boldsymbol{\mu}_w + \boldsymbol{t}$, reverse differentiation propagates through $J$, $W$, and $R S S^T R^T$.
4. **Finite-Difference Verification**:
   All analytical gradients will be unit-tested against numerical approximations $\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$ in Rust unit tests with $\epsilon = 10^{-4}$ and tolerance $\le 10^{-3}$.

---

## 8. Adam Optimizer

Parameters are updated using the Adam optimizer with decoupled parameter-specific learning rates:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Learning Rate Schedule
- $\boldsymbol{\mu}$ (position): Exponential decay from $\eta_{\text{init}} = 1.6 \times 10^{-4}$ to $\eta_{\text{final}} = 1.6 \times 10^{-6}$.
- $\boldsymbol{s}$ (scale): Constant $\eta = 5.0 \times 10^{-3}$.
- $\boldsymbol{q}$ (rotation): Constant $\eta = 1.0 \times 10^{-3}$, re-normalized to unit norm after each step.
- $\sigma$ (opacity): Constant $\eta = 5.0 \times 10^{-2}$.
- $\boldsymbol{c}_{sh}$ (SH coefficients): Constant $\eta = 2.5 \times 10^{-3}$.

---

## 9. Adaptive Density Control

Adaptive density control matches scene complexity by densifying under-reconstructed areas and pruning redundant or transparent Gaussians:

```
                      +-----------------------------+
                      |   Gradient on 2D Screen     |
                      |   Position > tau_grad       |
                      +-----------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v (Scale > tau_scale)             v (Scale <= tau_scale)
            [ SPLIT GAUSSIAN ]                [ CLONE GAUSSIAN ]
         - Divide scale by 1.6             - Duplicate Gaussian
         - Sample 2 Gaussians along        - Move along gradient
           covariance ellipsoid
```

### Densification Conditions
- **Threshold**: Screen-space position gradient average $\|\bar{\nabla}_{\boldsymbol{\mu}_{2D}}\| \ge \tau_{\nabla} = 0.0002$.
- **Split**: If $\max(\boldsymbol{s}) > \tau_{\text{scale}}$, split into 2 Gaussians scaled down by a factor of $1.6$, sampling their positions along the original Gaussian's 3D covariance.
- **Clone**: If $\max(\boldsymbol{s}) \le \tau_{\text{scale}}$, duplicate the Gaussian and displace slightly along the gradient.
- **Pruning**: Remove Gaussians where $\alpha < \epsilon_{\alpha} = 0.005$ or scale exceeds maximum scene bounds.
- **Opacity Reset**: Periodically (e.g. every 3000 iterations), reset opacities $\sigma \leftarrow \operatorname{logit}(0.01)$ to eliminate floating artifacts.

---

## 10. Camera Pose Estimation (Monocular SfM)

To support incremental video reconstruction without prior camera poses:
1. **Feature Detection**: High-speed FAST corner detection with Shi-Tomasi corner ranking or Harris response implemented in Rust/WASM.
2. **Feature Description**: ORB-style binary intensity centroid descriptors (BRIEF).
3. **Feature Matching**: Fast Hamming distance calculation with SIMD in WebAssembly, filtered via Lowe's ratio test ($d_{\text{best}} / d_{\text{second\_best}} < 0.75$).
4. **Essential Matrix & RANSAC**: Nistér 5-point algorithm with RANSAC to calculate relative pose $(R, \boldsymbol{t})$ between initial keyframes.
5. **Triangulation**: Direct Linear Transform (DLT) with chirality check ($z > 0$ in both cameras).
6. **Subsequent Frames (PnP)**: Perspective-n-Point with RANSAC (EPnP / P3P) to register new frames against existing 3D landmarks.
7. **Local Bundle Adjustment**: Levenberg-Marquardt optimization over a sliding window of recent keyframes to minimize reprojection error.

---

## 11. Incremental Scene State Management

For continuous video streaming reconstruction:
- **Keyframe Buffer**: Frames are added as keyframes only when visual disparity or parallax exceeds threshold $\Delta \theta > 5^\circ$ or median feature displacement $> 30$ px.
- **Spatial Coverage & Voxel Hash Grid**: 3D space is divided into a spatial hash grid. Cells track observed density.
- **Region Activation**: New frames initialize Gaussians only in unobserved or poorly reconstructed viewing cones.
- **Dual-Phase Optimization**:
  - **Local Optimization**: Fast iterations focusing only on Gaussians within the frustum of the latest keyframes.
  - **Global Consolidation**: Lower frequency background refinement of the global scene to prevent catastrophic forgetting.

---

## 12. Browser GPU Limitations & Mitigation

| Browser/WebGPU Limitation | Technical Risk | Architectural Mitigation |
| :--- | :--- | :--- |
| `maxStorageBufferBindingSize` (128 MB default) | A scene with millions of Gaussians exceeds single buffer limits | Chunked buffer layout: split large parameter arrays across multiple bound storage buffers or dynamically resize buffers via WebGPU adapter limits query. |
| Lack of Native 64-Bit Atomics | Cannot do single-pass atomic key-value sort with 64-bit keys | Use 32-bit composite keys or multi-pass radix sorting utilizing separate 32-bit key and index buffers. |
| Browser Device Loss (`gpudevice.lost`) | Heavy compute workloads trigger OS watchdog timer (TDR) | Chunk rendering/training dispatches into discrete batches of iterations; yield to browser event loop via `requestAnimationFrame` and `scheduler.yield()`. |
| WASM 4 GB Memory Ceiling | Large video decoding / point clouds can exhaust WASM linear memory | Stream video decoding via WebCodecs `VideoDecoder` directly into GPU textures or zero-copy ImageBitmap, bypassing raw byte retention in WASM. |
| Floating Point Precision in WGSL | Float32 precision loss at large scene coordinates | Camera-centered coordinate normalization per frame. |

---

## 13. Testing and Benchmarking Strategy

### 1. Unit Testing
- **Math & Algebra**: Comprehensive tests in Rust for quaternion conversions, covariance projection, and matrix inversions.
- **Gradient Verification**: Compare analytical compute results against finite differences.
- **PLY Parser Verification**: Validate against standard header variants (binary little-endian, ASCII, varying SH orders).

### 2. Integration & Rendering Tests
- Direct pixel comparison against reference renders for canonical test cases (single Gaussian, synthetic multi-Gaussian clouds).
- WebGL / WebGPU headless testing with Puppeteer / Playwright.

### 3. Benchmarking Metrics
- **Frame Rate**: Interactive rendering at 60+ FPS at $1920 \times 1080$.
- **Training Throughput**: Measured iterations per second (target $\ge 20-50$ it/s in browser).
- **Quality Metrics**: PSNR, SSIM, and L1 loss against ground truth images.
- **Camera Tracking**: Average reprojection error (pixels) and trajectory drift percentage.

---

## Milestone 1 Implementation Plan

To satisfy the first deliverable:
1. **Initialize Project Infrastructure**:
   - Modern React + TypeScript + Vite + Tailwind/Zustand workspace.
   - Core Rust / WASM library for high-performance PLY parsing and math utilities.
2. **Implement Custom WebGPU Gaussian Splat Renderer**:
   - WebGPU context and device initialization with optimal limits.
   - Standard 3DGS PLY binary parser (supporting float positions, normals, rotations, scales, spherical harmonics, opacity).
   - Custom WGSL compute shaders for 3D covariance, perspective projection, screen bounding, and sorting key generation.
   - Depth sorting implementation.
   - Custom WGSL rasterizer / alpha compositing pipeline rendering directly to canvas.
   - Interactive orbit/pan/zoom camera controller with standard 3D viewport controls.
3. **Verify with Real PLY Datasets**:
   - Test with actual 3D Gaussian Splat PLY files.
   - Demonstrate 60 FPS real-time rendering and navigation in the browser.

