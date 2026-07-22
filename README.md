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
One monocular video of a *moving* scene — so for every moment we only have ONE view, and classic NeRF is hopeless. NSFF uses **two fields**: a static NeRF for the background, and a dynamic NeRF that takes time as input (here: a learned 48-dim per-frame embedding) and additionally predicts **scene flow** (where each 3D point moves to in the next/previous frame). Following [nsff_pl](https://github.com/kwea123/nsff_pl), the two fields are composited **additively in NeRF-W style** — `alpha = 1 − (1−α_static)(1−α_dynamic)` — instead of the original's learned per-point blending weight. Because one view per time step is not enough supervision, NSFF leans on 2D priors: a monocular depth network (scale-invariant depth loss) and RAFT optical flow (the predicted 3D scene flow, projected to 2D, must match the observed flow). Warping the dynamic field to neighboring frames and re-rendering gives extra photometric supervision; occlusion weights are **inferred from the difference between warped and reference rendering weights** (no learned occlusion head). Two entropy-style losses keep the decomposition honest: an entropy loss makes the dynamic object "thin" along each ray, and a thickness cross-entropy loss pushes the static field's density peaks away from the dynamic ones. Everything is done in **NDC space** (the forward-facing parameterization where the scene is squeezed into a unit cube — this only works if the scene scale is normalized correctly, see the bug notes below!). At test time, frame interpolation works by splatting the per-plane radiance along the predicted scene flow (softmax splatting), and the dynamic field is **visibility-culled** outside the training-camera frustum to avoid ghosting on novel camera paths.

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
`--t_fixed` / `--view_idx` default to the middle frame. Each mode writes its own `inference_*` subfolder (per-frame PNGs + a looping `animation.gif`, plus an `animation.mp4` when ffmpeg is available). Bullet time follows nsff_pl's sinusoidal *wander path* around the fixed view. The fixed-view mode interpolates fractional time steps via scene-flow warping + softmax splatting, so it needs `cupy` (see the cupy note below).

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

---

## NSFF v2: Why the faithful reproduction fell short — a full analysis (2026-07-22)

After all eleven fixes above, the pipeline was a *correct* reproduction of the original
[Neural-Scene-Flow-Fields](https://github.com/zhengqili/Neural-Scene-Flow-Fields): val PSNR 33.5,
correct depth, real scene flow, clean decomposition. Yet the demos were still clearly worse than
[nsff_pl](https://github.com/kwea123/nsff_pl)'s reference GIF: softer textures everywhere, ghost
fragments of the kid on novel camera paths, and speckled depth on the road. This section documents
the investigation of that residual gap, the conclusion, and every change that closed it.

### Step 1 — Ruling out the renderer

The natural first suspects were inference-side: maybe the softsplat/MPI interpolation path blurs,
maybe the NDC↔world round-trip is inexact, maybe the GIF export destroys detail. Controlled
experiments on the trained v189 checkpoint (same weights, same view, rendered several ways):

| Experiment | PSNR vs GT | Conclusion |
|---|---|---|
| Direct volume rendering, `perturb=0, noise_std=0` | 33.49 | baseline model quality |
| Direct rendering, `perturb=1, noise_std=1` (what inference was actually doing) | 33.38 | test-time noise costs only 0.1 dB in RGB — but it visibly speckles the depth |
| softsplat/MPI `interpolate()` path at `dt=0` | 33.489 | **numerically identical** to direct rendering — the splatting renderer was NOT the blur source (`ndc2world` was verified to be the exact inverse of `get_ndc_rays`) |
| GIF re-encode of a rendered frame | ~34 dB vs its own PNG | imageio's 256-color quantization adds visible posterization, but the reference GIF survives the same encoding — so this is cosmetic, not causal |

**Conclusion:** the inference stack was faithful. The trained model itself was the ceiling —
33.5 dB with soft textures is simply where *this formulation* converges. The reference demo was
produced by nsff_pl, which does **not** implement the paper's formulation verbatim; its quality
comes from a set of deliberate design changes. In other words: **there was no remaining bug to
find — the two repos train different methods.**

### Step 2 — The key reason: how the two fields are composited

This is the single most important difference, and it lives at the **rendering equation** level.

**The original (what we had).** The static model predicts a time-independent blending weight
$w(\mathbf{x}) \in [0,1]$, and the two fields are mixed through one gate inside the volume
rendering integral:

$$\alpha_{dy} = \big(1-e^{-\sigma_{dy}\delta}\big)\,w,\qquad
\alpha_{st} = \big(1-e^{-\sigma_{st}\delta}\big)\,(1-w),\qquad
T_i=\prod_{j<i}(1-\alpha_{dy})(1-\alpha_{st})$$

$$C = \sum_i T_i\,\big(\alpha_{dy}\,c_{dy} + \alpha_{st}\,c_{st}\big)$$

The failure mode is structural. Because $w$ has **no time input**, every 3D region the kid *ever*
occupies during the 30 frames must be classified "dynamic" *for all frames* — including the 28
frames in which that region is actually empty road. Two consequences:

1. **Gradient starvation of the static field.** Inside the kid's swept volume the static branch
   is multiplied by $1-w\approx 0$, so the photometric loss sends it almost no gradient. It
   drifts to garbage unless propped up by an auxiliary masked static loss (bug #8 above was
   exactly this band-aid — treating a symptom of the formulation, not a bug in the code).
2. **Soft gates render soft images.** Anywhere the optimizer leaves $w$ fractional, the output is
   a 50/50 blend of two half-trained fields. The original's entropy loss on $w$ pushes the *gate*
   toward 0/1, but does nothing to sharpen the *geometry* of either field along the ray — the mush
   is baked into the compositing.

**nsff_pl (what we ported).** Delete the gate entirely and composite the two fields as independent
alpha events, NeRF-W style:

$$\alpha_s = 1-e^{-\sigma_s\delta_s},\qquad
\alpha_d = 1-e^{-\sigma_d\delta_d},\qquad
\alpha = 1-(1-\alpha_s)(1-\alpha_d),\qquad
T_i=\prod_{j<i}(1-\alpha_j)$$

$$C=\sum_i T_i\,\big(\alpha_s\,c_s+\alpha_d\,c_d\big)$$

Now **both fields receive gradient at every sample, in every frame** — the static field keeps
learning the road even while the kid stands on it, because nothing multiplies it to zero. The
static/dynamic *separation* is no longer enforced by an architectural gate but by two explicit,
targeted losses (Step 3). This is why nsff_pl needs neither the blending weight nor the masked
static loss, and why its static background is tack-sharp: it gets ~30× more effective supervision
in every region the dynamic object ever visits.

### Step 3 — The other structural causes, in decreasing order of impact

1. **Learned occlusion weights can be bought off.** The original *predicts* disocclusion
   probabilities that gate the warped-frame photometric loss, regularized toward 1 with
   $0.1\cdot L_1$. The network can pay that small fixed penalty to mute the warping supervision
   exactly where scene flow is hardest to learn — which is exactly where supervision matters most.
   nsff_pl removes the head and *derives* occlusion from geometry:
   $\text{disocc}=1-\lvert w_{warped}-w_{ref}\rvert$ (detached). A derived quantity cannot be
   gamed by the optimizer, so the photometric pressure on scene flow never disappears.

2. **ReLU density is a rough optimization landscape.** With $\mathrm{relu}(\sigma+\varepsilon)$,
   any sample pushed below zero has exactly zero gradient — density boundaries rattle between
   "off" (no gradient) and "on" instead of settling. Softplus keeps a smooth gradient everywhere.
   Empirically this is what killed the depth speckle: the near-noise-free depth maps appeared in a
   20k-step smoke run, long before convergence, so it is the activation — not training length —
   that cleans the density field.

3. **Ray termination details.** The original used one huge last interval ($10^{10}$, scaled by
   $\lVert\mathbf{d}\rVert$) for both fields. nsff_pl uses **asymmetric last deltas**: 100 for the
   static field (the background is opaque — rays must terminate, no semi-transparent far-plane
   fog) and $10^{-3}$ for the dynamic field (a thin object must NOT be able to absorb the whole
   remaining ray). Deltas are also used in raw NDC-z units, not scaled by the ray norm.

4. **Supervision hygiene and budget.** (a) *Anti-correlated time sampling*: consecutive training
   batches are drawn from frames at least 5 apart, so the static field cannot slowly memorize the
   dynamic object from temporally adjacent, nearly identical views. (b) *Schedule*: 340k steps
   with cosine decay $5\times10^{-4}\rightarrow10^{-8}$, versus our 150k with a single late ×0.1
   drop — high-frequency texture needs the long low-LR tail (the original authors' released kid
   model was itself trained 360k steps). (c) A 48-dim time embedding (ours was 16) gives the
   dynamic field enough capacity to separate the 30 time instants sharply.

### Step 4 — What had to change at the rendering level (summary)

In `utils/nsff_rendering.py`, ranked by importance:

1. **Compositing equation**: blend-weight mixture → independent-alpha additive composition
   (the equations in Step 2). The static model loses its `v` head.
2. **Sigma activation**: ReLU → Softplus, for both fields, noise added *before* the activation.
3. **Last-sample deltas**: uniform $10^{10}\cdot\lVert\mathbf{d}\rVert$ → static 100 / dynamic
   $10^{-3}$, unscaled.
4. **Warped re-rendering** (the fw/bw photometric supervision) now composites the warped dynamic
   field **with the current static field** instead of rendering the dynamic field alone — the
   warped prediction is compared against a full image, so it must explain occlusions by the
   background correctly.
5. **Occlusion**: the learned 2-channel prob head is gone; disocclusion is computed from the
   warped vs reference rendering weights inside the renderer.
6. **Test time only**: `perturb=0, noise_std=0`; integer-time frames use direct volume rendering
   (the splat path is reserved for fractional-time interpolation); dynamic sigma is **visibility
   culled** (set to −10 before Softplus) at sample points that no training camera sees — this is
   what removes the ghost fragments on novel camera paths, since nothing in training ever
   constrains the dynamic field outside the training frusta.

### Step 5 — What had to change at the loss level

In `losses/nsff_loss.py`:

* **Entropy on the dynamic *rendering weights*** (not on the blend gate):
  $10^{-3}\sum_i -w_i\log w_i$ per ray — makes the dynamic object occupy few, concentrated
  samples along each ray ("thin"), which is a geometry statement, not a gate statement.
* **Thickness cross-entropy**: dilate the dynamic weights with a 15-sample box filter, then
  penalize $\sum_i \tilde{w}^{dy}_i \log(w^{st}_i+10^{-8})$ (weight ramping 0→2e-4 over 10
  epochs) — pushes static density peaks at least ~15 samples away from dynamic ones, so the two
  fields cannot both explain the same surface.
* Photometric warp loss normalized by mean disocclusion; cycle loss weighted per-sample by
  disocclusion; monodepth loss moves to the **composite** depth; monodepth/flow weights decay
  ×0.1 every 10 epochs. The remaining flow regularizers (temporal linearity, minimal flow,
  spatial smoothness) are unchanged — they were already correct.

### Symptom → root cause → fix

| Observed symptom (v189) | Root cause | Fix |
|---|---|---|
| Soft/mushy textures everywhere | gradient starvation + fractional blend gate + short LR schedule | additive composition; entropy + thickness losses; 340k cosine schedule |
| Ghost fragments of the kid at novel views | dynamic field unconstrained outside training frusta; swept volume marked dynamic for all t | test-time visibility culling; additive composition |
| Speckled/noisy depth | ReLU density boundaries + `noise_std=1, perturb=1` left on at eval | Softplus sigma; noise/perturb forced to 0 at test |
| Garbage in static-only renders (pre-v189 #8) | time-independent $w$ starves static field in swept volume | additive composition (masked static loss no longer needed) |
| Semi-transparent far-plane fog | shared huge last delta | asymmetric last deltas (static 100 / dynamic 1e-3) |
| Weak scene flow in hard regions | learned occlusion probabilities muting the warp loss | occlusion derived from warped-weight difference (ungameable) |
| GIF posterization | imageio single-pass palette | ffmpeg palettegen/paletteuse two-pass (+ mp4 export) |

### Outcome

`logs/nsff/version_191`: best val PSNR **33.5 → 34.96** (now measured noise-free, i.e. a stricter
metric), monotone convergence over 340k steps (~5.6 h on one RTX 4090 at 17.9 it/s), no ghosting
on the wander path, near noise-free depth, and bullet-time/fixed-view/spiral demos on par with the
nsff_pl reference. The decomposition needs no auxiliary props: the static-only render is clean in
the kid's swept volume *without* a masked static loss.

**The takeaway:** a faithful reproduction of the paper converges to the paper's quality, not to
nsff_pl's. The gap was not a bug but a formulation: a time-independent multiplicative gate between
the two fields (plus a learnable occlusion escape hatch) creates degenerate optima that no amount
of code-fixing removes. Recomposing the fields additively and replacing learned gates with derived
quantities and explicit geometry losses is what makes the sharp results possible.

*(Historical note: the pipeline described in the bug list above was faithful to the original
zhengqili implementation — learned blending weight, learned occlusion probabilities, dynamic-only
depth loss. The current code no longer contains those pieces; see the v2 section above.)*