# NeRFs

This repository is a study of **NeRF, DS-NeRF, NeRF++, NeRF in the Wild (NeRFW), and NSFF**. It integrates implementations from various state-of-the-art resources into a single framework. Many parts of the code are borrowed from:

* [nerf_pl](https://github.com/kwea123/nerf_pl) & [nsff_pl](https://github.com/kwea123/nsff_pl)
* [DSNeRF](https://github.com/dunbar12138/DSNeRF)
* [nerfplusplus](https://github.com/Kai-46/nerfplusplus)
* [Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields)

> 💡 **Feature:** Different versions of NeRF can be executed simply by switching the configuration file.

---

## Brief Introduction of Methods

### NeRF
An MLP takes a 3D position + viewing direction and outputs a color and a density. To draw one pixel, we shoot a ray through the scene, sample points along it, ask the MLP for color/density at each point, and blend them with volume rendering (near, dense stuff hides far stuff). The magic ingredient is **positional encoding**: feeding `sin/cos(2^k x)` of the coordinates into the MLP, which lets a small network represent sharp details. Training only needs posed images and a simple "rendered pixel vs. real pixel" MSE loss.

### DS-NeRF
Plain NeRF needs many views, otherwise the geometry it learns can be wrong even when the renders look okay. DS-NeRF's trick: we already ran COLMAP to get camera poses, and COLMAP also gives us a **free sparse 3D point cloud**. So add a second loss that tells rays passing through those points "your ray should terminate at this depth". This extra depth supervision makes training converge faster and produce correct geometry with far fewer input images.

### NeRF++
NeRF samples points inside a bounded volume, but in real 360° captures the background (sky, far buildings) is essentially at infinity — you cannot cover that range with a fixed near/far interval. NeRF++ **splits the world into two models**: everything inside a unit sphere is a normal NeRF, and everything outside is modeled by a second NeRF using the **inverted-sphere trick** — a far point is represented as (direction on the sphere, 1/distance), so "infinitely far" becomes the well-behaved coordinate `1/r → 0`. Foreground and background are then composited together.

### NeRFW (NeRF in the Wild)
Internet photos of a landmark differ in lighting, exposure, and contain tourists/cars that only exist in one photo. Two ideas fix this: (1) a learned **per-image appearance embedding** is fed to the color head, so each photo can have its own lighting while sharing one geometry; (2) a separate **transient head** (also with a per-image embedding) models the stuff that only exists in that one photo, together with an uncertainty value that automatically down-weights the loss on those pixels. At test time we drop the transient part and get a clean static scene.

### NSFF (Neural Scene Flow Fields)
One monocular video of a *moving* scene — so for every moment we only have ONE view, and classic NeRF is hopeless. NSFF uses **two fields**: a static NeRF for the background, and a dynamic NeRF that takes time as input and additionally predicts **scene flow** (where each 3D point moves to in the next/previous frame). The two are blended per 3D point with a learned blending weight. Because one view per time step is not enough supervision, NSFF leans on 2D priors: a monocular depth network (scale-invariant depth loss) and RAFT optical flow (the predicted 3D scene flow, projected to 2D, must match the observed flow). Warping the dynamic field to neighboring frames and re-rendering gives extra photometric supervision, with learned occlusion weights to ignore pixels where warping is invalid. Everything is done in **NDC space** (the forward-facing parameterization where the scene is squeezed into a unit cube — this only works if the scene scale is normalized correctly, see the bug notes below!). At test time, frame interpolation works by splatting the per-plane radiance along the predicted scene flow (softmax splatting).

---

## Environment Setup

A Docker environment is provided to ensure consistency across different NeRF implementations.

1. **Build the image:** `docker build -t colmap docker/`
2. **Launch the container:** `./docker/run_container.sh`

> **cupy note (NSFF test-time only):** NSFF's frame interpolation uses softmax splatting, which needs `cupy`. Training does not need it (the import is lazy). If you render NSFF GIFs outside the Docker image, install a version that matches your CUDA **and does not drag in numpy 2.x** (which breaks torch 2.1.1):
> ```bash
> pip install "cupy-cuda12x==13.6.0" "numpy==1.24.4"
> ```

## Data Preparation

For convenience, I have only tested a specific dataset for each NeRF version, though they are designed to be used interchangeably.

### NeRF & DS-NeRF
* **Dataset:** Download LLFF dataset from [here](https://github.com/Fyusion/LLFF).
* **Depth Assisted Loss:** To use DS-NeRF's depth loss, we need to generate depth maps from images:
    1. Follow the [Preparation Guide](preprocess/README.md).
    2. Run: `python preprocess/process_llff.py --root_dir <path/to/ROOT_DIR>`

### NeRF++
* **Dataset:** Download from [Tanks and Temples](https://www.tanksandtemples.org).
* **Preprocessing:**
    1. Follow the [Preparation Guide](preprocess/README.md).
    2. Run: `python preprocess/normalize_cam_dict.py --root_dir <path/to/ROOT_DIR>`
    * *Note: This normalizes poses into a unit circle as proposed by the original paper. Results are saved in `split_data` under the root directory.*

### NeRF in the Wild (NeRFW)
* **Dataset:** Download PhotoTourism dataset from [UBC PhotoTourism](https://www.cs.ubc.ca/~kmyi/imw2020/data.html).

### NSFF (Neural Scene Flow Fields)
* **Dataset:** Currently using the "Kid" series from [nsff_pl v2.0](https://github.com/kwea123/nsff_pl/releases/tag/v2.0).

---

## Training & Inference

### Training
Use the YAML files in `configs/` to specify your model and dataset preferences.
```bash
# Example: Training DS-NeRF on LLFF
python train.py --config configs/train_dsnerf.yaml
```

### Inference
Weights and logs are stored in the logs/ directory.
```bash
# Example: Run inference on a specific version
python inference.py --log_path logs/dsnerf/version_<ID>
```

#### NSFF inference modes
NSFF disentangles space (camera) from time (scene motion), so we can move each one independently. Use `--mode`:

```bash
# Default: camera spirals AND time advances together
python inference.py --log_path logs/nsff/version_<ID> --mode spiral

# "Bullet time": freeze time (the kid stops), orbit the camera to reveal 3D
python inference.py --log_path logs/nsff/version_<ID> --mode fixtime  --t_fixed 15

# Fixed view: hold the camera still, let time advance (the kid walks, slow-mo)
python inference.py --log_path logs/nsff/version_<ID> --mode fixview  --view_idx 15

# Render all three
python inference.py --log_path logs/nsff/version_<ID> --mode all
```
`--t_fixed` / `--view_idx` default to the middle frame. Each mode writes its own `inference_*` subfolder (per-frame PNGs + a looping `animation.gif`). The fixed-view mode interpolates fractional time steps via scene-flow warping + softmax splatting, so it needs `cupy` (see the cupy note below).

## Demo

### NeRF
<img src="demo/nerf_loop.gif" alt="nerf" width="800"/>

### NeRF++
<img src="demo/nerf++_loop.gif" alt="nerf++" width="800"/>

### NeRFW
<img src="demo/nerfw_loop.gif" alt="nerfw" width="800"/>

### NSFF
The camera spirals **and** time advances together (the boy walks while the view moves):

<img src="demo/nsff_loop.gif" alt="nsff spiral" width="800"/>

Because NSFF learns the scene as a function of camera position *and* time separately, we can freeze one axis and move the other:

**Bullet time** — time is frozen (the boy stops mid-stride) while the camera gently orbits, so the parallax between him and the background reveals the recovered 3D:

<img src="demo/nsff_bullet_time.gif" alt="nsff bullet time" width="800"/>

**Fixed view** — the camera is held still while time advances, so the boy walks across a rock-steady background as smooth slow motion:

<img src="demo/nsff_fixed_view.gif" alt="nsff fixed view" width="800"/>

*(Each frame shows rendered RGB on the left and estimated depth on the right; render them with `python inference.py --log_path logs/nsff/version_<ID> --mode all`.)*

---

## NSFF Debugging Notes (2026-07)

For ~187 training runs the NSFF implementation could not reproduce the paper results: the rendered depth was pure noise, the predicted scene flow was ~zero, and the static/dynamic decomposition failed (the static model was fog, the dynamic model memorized the whole scene). Below is the full list of bugs that were found and fixed, in decreasing order of importance. Reference implementations used for line-by-line comparison: [nsff_pl](https://github.com/kwea123/nsff_pl) and the original [Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields).

### 1. The scene scale was never applied (the killer bug)
`datasets/dynamic.py` computed a scale factor that should make the nearest scene content sit at depth ≈ 1.33, but applied it to a **stale local variable**:

```python
self.poses = center_poses(self.poses)   # self.poses is a NEW array here
...
poses[..., 3] /= self.scale_factor      # BUG: modifies the old local `poses`,
                                        # self.poses is left untouched!
```

Why this destroys everything: NDC ray parameterization assumes the near plane is at depth 1. Our COLMAP scene had its nearest content at depth ≈ 15 (scale factor was 11.2), so after the NDC mapping (`z_ndc = 1 - 2/z`) the ENTIRE scene got squeezed into the last ~15% of the ray. Consequences:
* almost all of the 128 samples per ray landed in empty space → depth maps were noise;
* the "zero the scene flow beyond z = 0.95" mask killed the flow for most of the actual scene → predicted flow ≈ 0;
* points near the far plane have almost no parallax between frames → the optical-flow loss had nothing to work with.

Fix: `self.poses[..., 3] /= self.scale_factor` (after `center_poses`).

### 2. The nearest-depth estimation regressed the wrong quantity
The files in `disps/` are **disparity** (1/depth) from a monodepth network, not depth. The code fitted `disp ≈ a·depth + b` (a line through a hyperbola — nonsense) and then also inverted the fit incorrectly. Fixed to match nsff_pl: fit `disp ≈ a·(1/depth) + b`, then `nearest_depth = a / (disp_95th_percentile − b)`. This value drives the scale factor of bug #1, so both had to be right.

### 3. Missing per-frame `shift_near` in the NDC conversion
When a camera moves forward past the global near plane (world z < −1), shifting ray origins to the global plane puts them BEHIND the camera. nsff_pl shifts each frame's ray origins to `max(1.0, −camera_z)` instead. Added the same (`get_ndc_rays(..., shift_near=...)`).

### 4. Missing entropy loss on the blending weight
The original adds `1e-3 · mean(−w·log w)` on the blending weight `w` so each 3D point decides clearly "I am static" or "I am dynamic". Without it the blend stays soft → the foggy static model we observed.

### 5. Occlusion weights were basically unregularized
The learned occlusion/visibility weights gate the warped-frame photometric losses. The original pushes them toward 1 with strength `0.1 · L1`; our code used `0.001 · (1−p)²` — about 100× weaker, so the network could "mute" the warping supervision for free. Also the 2D occlusion maps were integrated with the wrong ray weights (reference-frame blend weights instead of the warped rendering's weights).

### 6. Cycle-consistency loss 10× too weak
Forward flow at time t and backward flow at t+1 must be consistent. The original kid-running config uses `w_cycle = 1.0`; we had `0.1`.

### 7. No motion-mask hard mining
The dynamic object covers only ~8% of the pixels, so uniform ray sampling rarely sees it. The original samples **512 extra rays from the motion-mask region** every step during the first `decay_iteration·1000` steps (and excludes those extra rays from the full-render loss so they don't bias the static model). Our masks (`masks/*.png`) use 0 = dynamic, 255 = static — note this is inverted vs. what nsff_pl's comment says.

### 8. Static model got NO direct supervision → garbage in static-only renders
The blending weight `w(x)` has **no time input**, so every 3D region the kid ever walks through is marked "dynamic" for ALL frames. In those regions the static field receives zero gradient from the blended render loss (it is multiplied by `1−w ≈ 0`) and drifts to garbage — visible as colorful blobs in static-only renders. Fix (borrowed from DynamicNeRF, Gao et al. 2021): add a **masked static loss** — on pixels the motion mask marks as static, the *pure static* rendering must match the ground-truth image directly, bypassing the blend. Those road regions are mask-static in ~28 of 30 frames, so this pins the static field down everywhere that matters.

### 9. Training was disk-bound
Every training step re-loaded 8 full images + flows + disparity maps from disk and built rays for every pixel (~1M rays), then used only 2048 of them. Now all per-frame ray buffers are built once at startup (~3 s, ~1 GB RAM) and each step just indexes into them. Training went from "days, maybe" to ~7 h for 150k steps on one RTX 4090.

### 10. Inference / frame-interpolation bugs
* `interpolate()` multiplied the dynamic alphas by the blending weight **twice** (they already had it baked in).
* The MPI depth accumulation forgot the per-plane alpha (`depth += T·z` instead of `depth += T·α·z`).
* `imageio.mimsave(..., fps=30)` crashes on imageio ≥ 2.28 — replaced with a small version-compatible `save_gif` helper (all trainers).
* `cupy`/softsplat is only needed at test time, so its import is now lazy and training works without cupy installed.

### 11. Hyperparameters aligned with the original kid-running config
`N_samples 64 → 128`, `noise_std 0 → 1.0`, batch = 1024 uniform + 512 hard-mined rays, `decay_iteration 30` (depth/flow losses decay 10× every 30k steps, hard mining stops at 30k).

**Result:** best validation PSNR went from **29.7 → 33.5+**, with correct depth, real scene flow, and a clean static/dynamic separation.

Things that LOOK like bugs but are faithful to the original — do not "fix":
* the small-scene-flow regularizer sums over the *coordinate* axis (dim −1), exactly like the original;
* the depth loss is applied to the dynamic-only depth (not the blended one);
* the blending weight is predicted by the **static** model (paper Eq. for `(c, σ, v)`), and multiplies the alpha *outside* the exponent.