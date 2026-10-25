---
title: Image to Splat
category: Presets / Models
---

# Image to Splat

Image to Splat turns one image into a 3D Gaussian splat scene. It trains nothing and generates nothing new: it lifts the picture into depth with MoGe-2 (Microsoft, MIT), fills the background hidden behind the foreground, and writes one Gaussian per pixel and per layer. A run takes seconds and a few GB of VRAM. The scene is seen from the source image's camera, so it suits small camera moves around that viewpoint: captures from a new angle and short orbit, dolly or crane shots.

The code lives in `src/platform/runtime/native/arch/image_to_splat/` (the lift, the PLY writer and the preview) and reuses the pure-torch MoGe-2 port in `arch/moge/`. There is no compiled CUDA extension; the lift runs on the CPU in numpy and OpenCV.

## Files & detection

| file | size | notes |
|---|---|---|
| `geometry_estimation/moge_2_vitl_normal_fp16.safetensors` (`Comfy-Org/MoGe`) | 0.66 GB | the only file; the same one Pixal3D's Auto camera FOV uses |

`model_loader/image_to_splat` loads it as fp32 (about 1.3 GB resident) under the `MODELS` key `native/moge/<file>` with the fingerprint `native/moge/<file>|float32`. Pixal3D's camera estimator uses the same key and fingerprint, so switching between the two presets reuses the loaded model. A file that is not a MoGe-2 checkpoint is refused at load. The normal and metric-scale heads are skipped.

The `image-to-splat-starter` recipe fetches this one file. An instance that already ran `pixal3d-starter` has it.

Licences: MoGe-2's code and weights are MIT (Microsoft, `Ruicheng/moge-2-vitl-normal`, repackaged by Comfy-Org), on a DINOv2 ViT-L/14 backbone that Meta releases under Apache-2.0.

## Pipeline

1. **Resize.** The image is shrunk so its long side is 1024 px (Detail Standard) or 1536 px (Detail High). It is never enlarged.
2. **Geometry.** MoGe-2 runs at resolution level 9 (3600 tokens) and predicts an affine-invariant point map and a mask. `MoGe2Model.estimate_geometry` recovers the focal length and the z-shift together (`recover_focal_shift`, Levenberg-Marquardt on a 64×64 downsample, as upstream), adds the shift back to z and unprojects the depth through the recovered pinhole, so every point lies on its pixel's ray. Pixels with a mask under 0.5, a non-finite or a non-positive depth are invalid.
3. **Scale.** The depth is scaled so the median valid depth is 4.0. This version has no metric scale.
4. **Sky.** Invalid pixels (mostly sky) are pushed to a fixed far depth: at least 40, and at least 1.25 times the 99.5th percentile of the valid depth, so the sky always sits behind the scene.
5. **Cuts and flying pixels.** A pixel is on a depth cut when its 5×5 neighbourhood spans more than 0.35 in log depth (a jump of about 40 %). MoGe-2 blurs such jumps over a few pixels, and those in-between pixels would float between the two surfaces. Each cut pixel is snapped to the depth of the side it is closest to (the minimum or the maximum of its neighbourhood, in log depth). A steep but continuous surface, such as a floor or a sea seen at a grazing angle, stays below the threshold and is never cut. Dropping the cut pixels instead (`EDGE_PIXELS = EDGE_DROP`, which puts the background layer under them) leaves slightly fewer holes, but costs about 4 dB of PSNR in the source view, so snapping is the default.
6. **Hidden region.** The background layer is laid behind two kinds of pixels:
   - foreground segments: the map minus its cuts splits into connected regions, and a region is foreground when at least 60 % of the cut pixels bordering it are near sides;
   - disocclusion bands: from every near-side cut pixel, a band reaches into the occluder as far as the parallax a camera move of 0.3 × the median depth would reveal there, `f_px × 0.3 × 4 × (1/z_near − 1/z_far)` pixels, at least 4 px and at most 15 % of the long side. This covers an object that touches the ground and so never forms a segment of its own, such as a boat on the water.

   The sky is never hidden.
7. **Hidden background.** Each hidden pixel takes its background only from the surfaces that lie behind it, so a rope lying in a boat gets the boat behind it and the boat gets the sea, instead of both getting the sea.
   - Threshold: the pixel's log depth, raised to the depth of its nearest occluding outline (the closest near-side cut pixel) when that outline is less than 0.6 farther in log depth, so the inside of a face or a coat does not borrow its own silhouette; plus 0.175, half the cut threshold.
   - Samples: the visible pixels off the cuts, whose colours are not blurred across a jump.
   - Fill: the thresholds are split into levels 0.1 apart in log depth (at most 48). At each level, the colour and the log depth of the samples farther than the level are spread over the image by a pull-push pyramid at half resolution: each coarser level averages the samples of the finer one, a block counts as known once a quarter of it is, and the way back down keeps the finest known value. Every hidden pixel blends the two levels around its threshold, so the background varies continuously along an occluder whose depth changes.

   The filled depth is kept at least 3 % behind the foreground it hides.
8. **Gaussians.** Every valid or sky pixel becomes a front Gaussian, and every hidden pixel also becomes a background Gaussian from the inpainted layer. Each sits at its pixel's unprojected position, with a SH0 colour from the pixel and an opacity of 0.95, lowered to 0.35 on the cuts. Its covariance lies on the surface:
   - the tangents `t_u` and `t_v` are central differences of the unprojected point map along the image axes;
   - next to a cut (a step over 0.35 in log depth) only the continuous side is used, forward or backward, so a Gaussian never stretches across a jump; a pixel cut off on both sides falls back to its pixel footprint `z / f_px` along the image axes;
   - each tangent is clamped to 128 footprints;
   - the covariance is `0.6² (t_u t_uᵀ + t_v t_vᵀ)` in the surface plane, plus a thickness of 0.1 footprint along the normal `t_u × t_v`. Its eigenvectors give the quaternion, and the square roots of its eigenvalues give the log-scales.

   On a wall facing the camera, this is a pixel-sized disc. On an oblique surface (an arm, a raincoat, the deck) it is stretched along the slope, so neighbouring Gaussians still overlap when the camera turns.
9. **Write.** The `.ply` below, then a preview: a point render from the source view (nearest Gaussian per pixel, holes filled from neighbours) that becomes the file's thumbnails.

Expect about 1.1 to 1.7 Gaussians per pixel: roughly 1 to 1.3 million and 55 to 75 MB at Standard for a 4:3 image, twice that at High.

Every threshold above is a named constant in `arch/image_to_splat/config.py`: `EDGE_JUMP_LOG_RATIO`, `EDGE_SNAP_WINDOW_PX`, `EDGE_PIXELS`, `EDGE_OPACITY`, `FOREGROUND_VOTE_RATIO`, `DISOCCLUSION_BASELINE`, `HIDDEN_MIN_PX`, `HIDDEN_MAX_FRACTION`, `BACKGROUND_BEHIND_LOG`, `BACKGROUND_OUTLINE_REACH`, `BACKGROUND_LEVEL_STEP`, `BACKGROUND_MAX_LEVELS`, `PULL_PUSH_GAIN`, `INPAINT_SCALE`, `BACKGROUND_DEPTH_MARGIN`, `TANGENT_SCALE`, `TANGENT_MAX_FOOTPRINTS`, `NORMAL_THICKNESS`, `OPACITY`, `SKY_DEPTH_MIN`, `SKY_BEYOND_FARTHEST` and `KEEP_SKY`.

The current values were tuned on four real photographs (a fisherman with a coiled rope on a bow, a trawler in a harbour, boats in a misty bay, a boat at sunset). Each splat was rendered with an EWA Gaussian rasteriser at 640 px wide from seven cameras: the source view, an orbit of −10, +10 and +20°, a dolly of 15 and 30 % of the depth, and a crane of 10°. Against the first version (camera-facing discs flattened along the ray, 3×3 cuts, foreground segments only), small holes inside the scene (pixels left more than half transparent that a 9×9 closing would fill) fell on average from 1.6, 1.3 and 2.0 % to 0.35, 0.27 and 0.16 % for the three orbits, and from 3.9 % to 0.27 % for the crane. The source view kept its fidelity: its PSNR against the input went from 31.8 to 32.5 dB.

The background fill was measured on the same four photographs with an occluder pasted in front of each: a striped bar or disc, five per image, set at 0.55 times the depth behind it. The composite is lifted, and the background laid behind the occluder within 24 px of its outline is compared with the pixels and the depth it really hides. Against the earlier fill (OpenCV Telea over the whole foreground mask), the colour PSNR went from 16.3 to 19.9 dB. The share of pixels whose background depth lies within 0.2 in log depth (about 20 %) of the truth went from 65 to 81 %, and the median log-depth error from 0.26 to 0.07. On the fisherman's bow these figures go from 8.9 to 17.6 dB and from 31 to 84 %. In the renders, what the rope uncovers in the boat is the boat's dark interior instead of a patch of sky. The source view's PSNR stays at 32.5 dB, and the lift takes the same time, about a second at Standard. The background now sits at its real depth, farther than before, so a large move opens more of it. Small holes rose from 0.35, 0.27 and 0.16 % to 0.36, 0.31 and 0.30 % for the three orbits, and from 0.27 % to 0.34 % for the crane.

## Output

One `.ply` per source image, stored as a `SPLAT` file (see [the splat output type](../plugin-api.md)):

- binary little-endian, one `vertex` element with the float properties `x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`: SH degree 0, logit opacity, log scales, quaternion `w x y z`; no `f_rest_*`, no normals;
- frame: the source camera in OpenCV convention (x right, y down, z forward), origin at the optical centre;
- header comment `potionui_view fov_y=<degrees> aspect=<width/height> depth=<median depth> width=<px> height=<px> convention=opencv`, with `fov_y` the vertical field of view and `depth` 4.0. `src.plugin_api.probe_splat` reads it back.

The generator also reports a "3D scene" text artifact with the Gaussian count, the visible and hidden split, the image size and the field of view.

## Presets & modes

The shipped preset is `content/presets/marketplace/ImageToSplat/` (id `01KX47GSP4TQ0000000000SP3D`, category `3d`) with one mode, `img2splat` (Image to 3D scene):

- References tab: Source Image (required).
- Generation tab: Detail, Standard (1024) or High (1536), default Standard; Geometry Estimator (MoGe-2) under Models, required, with the Comfy-Org recommendation.

It uses `model_loader/image_to_splat` and `generator/image_to_splat`. There is no seed: the run is deterministic.

## Limits

- No generative completion. Behind the foreground there is only the filled background, and outside the frame there is nothing. Past an orbit of about 15° or a dolly of about 20 % of the depth, stretched edges, holes and empty borders show.
- Two layers only: each point has a single background. The boat behind a rope is filled in, but the sea behind that hidden part of the boat does not exist, and a large camera move can open it.
- Wide disocclusions (behind a boat or a person) are filled with a smooth pull-push of the surfaces behind: right colour and depth, no texture. Where that fill joins surfaces of very different depths, such as the sea and the sky behind a person, its Gaussians stretch along the slope and show as streaks at an orbit of 20°.
- MoGe-2 often blurs a step of 30 to 40 % in depth over 4 to 15 px, as between a raincoat and the rope behind it. Such a step stays under the cut threshold, so the surface stretches across it and streaks when the camera turns. Lowering the threshold to 0.30 or 0.25 did not reduce the streaks and added holes.
- Thin structures (hair, branches, fences) cut poorly and tend to leave halos.
- Without metric scale, the scene is in arbitrary units with a median depth of 4.

## Hardware

`requires:` asks for 4 GB minimum and recommends 8 GB. MoGe-2 holds about 1.3 GB of fp32 weights on the GPU while it runs and returns to the CPU before the lift. The lift and the write run on the CPU in about a second at High.
