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
5. **Cuts.** A pixel is on a depth cut when its 3×3 neighbourhood spans more than 0.04 in log depth. Its near side faces the camera.
6. **Foreground.** The map minus its cuts splits into connected regions. A region is foreground when at least 60 % of the cut pixels bordering it are near sides; near-side cut pixels are foreground too. The sky never is.
7. **Hidden background.** The foreground mask is dilated by 6 px and the image and the log depth are inpainted over it with OpenCV Telea (radius 5 px, at half resolution). OpenCV's Telea gives noise on float input, so the log depth goes through a 16-bit quantisation of its range. The filled depth is kept at least 3 % behind the foreground it hides.
8. **Gaussians.** Every valid or sky pixel becomes a front Gaussian, and every foreground pixel also becomes a background Gaussian from the inpainted layer. Each sits at its pixel's unprojected position with a SH0 colour from the pixel, an isotropic scale of `z / f_px × 0.6` flattened to 0.4 along its ray (the quaternion turns the local z axis onto the ray), and an opacity of 0.95, lowered to 0.35 on the cuts.
9. **Write.** The `.ply` below, then a preview: a point render from the source view (nearest Gaussian per pixel, holes filled from neighbours) that becomes the file's thumbnails.

Expect about 1.1 to 1.4 Gaussians per pixel: roughly 1 million and 50 MB at Standard for a 4:3 image, 2 million and 110 MB at High.

Every threshold above is a named constant in `arch/image_to_splat/config.py` (`DEPTH_EDGE_LOG_RATIO`, `FOREGROUND_VOTE_RATIO`, `INPAINT_DILATE_PX`, `INPAINT_RADIUS_PX`, `INPAINT_SCALE`, `BACKGROUND_DEPTH_MARGIN`, `SCALE_PER_PIXEL`, `RADIAL_FLATTEN`, `OPACITY`, `EDGE_OPACITY`, `SKY_DEPTH_MIN`, `SKY_BEYOND_FARTHEST`, `KEEP_SKY`...). The values are first guesses, to be tuned on real images (landscape, interior, character) on a GPU.

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

- No generative completion. Behind the foreground there is only the inpainted background, and outside the frame there is nothing. Past an orbit of about 15° or a dolly of about 20 % of the depth, stretched edges, holes and empty borders show.
- Two layers only: a middle ground hidden behind a foreground that is itself in front of a far background is filled with the far background.
- Thin structures (hair, branches, fences) cut poorly and tend to leave halos.
- Without metric scale, the scene is in arbitrary units with a median depth of 4.

## Hardware

`requires:` asks for 4 GB minimum and recommends 8 GB. MoGe-2 holds about 1.3 GB of fp32 weights on the GPU while it runs and returns to the CPU before the lift. The lift and the write run on the CPU in about a second at High.
