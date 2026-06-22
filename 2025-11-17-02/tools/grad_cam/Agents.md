You are an expert coding assistant specialized in **PyTorch + OpenMMLab MMSegmentation/MMEngine**, and you will implement a **Grad-CAM visualization tool** for a DeepLabv3+ family with RBNet-style ablations.

### Project Paths

* Project root:
  `/home/jetson/RB_REC_TOOL/2025-11-17-02/`
* Model config directory (four configs live here):
  `/home/jetson/RB_REC_TOOL/2025-11-17-02/configs/my_model_configs`
* Custom decode head that contains both `tr` (BiFormer) and `sae` (ACE):
  `/home/jetson/RB_REC_TOOL/2025-11-17-02/mmseg/models/decode_heads/sep_aspp_head_saetr.py`
* Where the Grad-CAM tool must be placed:
  `/home/jetson/RB_REC_TOOL/2025-11-17-02/tools/grad_cam`

### Models to Support (exactly these four)

1. DeepLabv3+ baseline
2. DeepLabv3+ + BiFormer (in code abbreviated as `tr`)
3. DeepLabv3+ + ACE (in code abbreviated as `sae`)
4. DeepLabv3+ + BiFormer + ACE (`tr + sae`)

The configs reflect this via the decode head type and flags. The custom head is registered via `@MODELS.register_module()` and implements conditional paths for `tr_model` and `sae_model`.

---

# 1. First: Read and Understand the Head Forward (this determines correct CAM hook points)

Open and carefully read:
`/home/jetson/RB_REC_TOOL/2025-11-17-02/mmseg/models/decode_heads/sep_aspp_head_saetr.py`

Key forward sequence you must verify (do not assume; confirm in code):

1. ASPP multi-branch outputs concatenation (`aspp_outs = cat([...])`)
2. If `tr_model` exists: `aspp_outs = tr_model(aspp_outs) + aspp_outs`
3. `output = bottleneck(aspp_outs)`
4. Low-level `c1_output = c1_bottleneck(inputs[0])`, upsample, then `cat([output, c1_output])`
5. If `sae_model` exists: `output = sae_model(output) + output`
6. `output = sep_bottleneck(output)`
7. `output = cls_seg(output)` → segmentation logits `[N, C, H, W]`

Important nuance: BiFormer routing may use `detach` on routing weights (depends on `diff_routing`), which can alter gradient flow. This should not break Grad-CAM if we hook at the right places, but you must be mindful about selecting layer points that actually receive gradients.

---

# 2. Goal: Build a Reproducible, Comparable Grad-CAM Tool Across All 4 Models

You must produce a tool that:

* Loads 4 models (config + checkpoint pairs)
* Runs Grad-CAM on the same sample set
* Produces “paper-friendly” grid images comparing the four models side by side
* Supports segmentation-specific target definitions (not classification)

The output must be robust and easy to reproduce.

---

# 3. Grad-CAM Layer Selection (Where to hook)

To make the 4-model comparison valid, layers must be “semantically aligned” across models. Implement 3 **primary** CAM locations for every model:

### CAM-Enc (encoder semantic layer)

* Hook: backbone output at the last stage (typically ResNet layer4 output)
* In MMSeg, backbone outputs a tuple/list based on `out_indices`. Take the last feature map.
* Implement as a forward hook on `model.backbone` and cache only the last feature map.

### CAM-ASPPpost (post-ASPP fusion, just before bottleneck)

* The cleanest aligned point: **the input tensor of `decode_head.bottleneck`**
* Implement using a **forward_pre_hook** on `model.decode_head.bottleneck` to capture its input (`aspp_outs`).
* This works for baseline and for `tr` models; for `tr` it captures the residual-merged ASPP features.

### CAM-Decpost (decoder fusion output, just before sep_bottleneck)

* Aligned point: **the input tensor of `decode_head.sep_bottleneck`**
* Implement using a **forward_pre_hook** on `model.decode_head.sep_bottleneck` to capture its input (`output`).
* This tensor includes low-level concat and (if enabled) SAE residual fusion.

These three are the default outputs used for paper grids.

---

# 4. Optional Fine-Grained CAMs (only if toggled on)

If the user wants deeper module-specific visualization, implement optional hooks:

* CAM-TRin: input to `decode_head.tr_model` (forward_pre_hook)
* CAM-SAEout: output of `decode_head.sae_model` (forward hook)

But default to only CAM-Enc / CAM-ASPPpost / CAM-Decpost to keep figures manageable.

---

# 5. Segmentation Targets (What to backprop)

Implement at least these two targets (must be selectable via CLI):

### Target T1 (global lesion logit mean) — mainline

* `y = seg_logits[:, lesion_idx, :, :].mean()`
* Very stable across samples; good for “global attention / connectivity” narrative.

### Target T3 (GT boundary-band weighted lesion logit) — ACE/boundary focused

* Build a narrow boundary band `B` (3–5 px width) from GT lesion mask.
* Use morphological ops with pooling (no external dependencies required):

  * dilation via `max_pool2d`
  * erosion via `-max_pool2d(-mask)`
  * band = clamp(dilation - erosion, 0, 1) or XOR equivalent
* Then:

  * `y = (seg_logits[:, lesion_idx] * band).sum() / (band.sum() + eps)`

Notes:

* Use logits (pre-softmax) for gradients.
* `lesion_idx` must be configurable (default 1, but allow override).
* If you implement Pred-mask variants, detach the mask to avoid non-differentiable paths.

---

# 6. Avoid the Common MMSeg Pitfall: Inference Path Must Allow Gradients

Do NOT use:

* `inference_model`
* `model.test_step`
* any `no_grad()` path

Also avoid slide inference (`test_cfg: mode='slide'`) for Grad-CAM because gradient stitching is messy and not comparable.

Instead:

* Build the input as mmseg expects using its pipeline/data_preprocessor
* Call a forward path that keeps autograd:

  * Preferred: `model.encode_decode(img, img_metas)` if available and supports grads
  * Or: `feats = model.extract_feat(img)` then `seg_logits = model.decode_head(feats)` and resize to input if needed
* Ensure model is in `eval()` but gradients enabled.

Remember: data_preprocessor uses `bgr_to_rgb=True` and mean=0/std=1 in these configs; keep consistency.

---

# 7. Grad-CAM Computation Implementation Requirements

Implement a hook manager that:

* Captures activation `A` (tensor) at selected hook point
* Calls `A.retain_grad()` so you can read `A.grad` after backward
* Backprop:

  * `model.zero_grad(set_to_none=True)`
  * `y.backward()`
* Compute CAM:

  * `alpha = grad.mean(dim=(2,3), keepdim=True)`
  * `cam = relu((alpha * A).sum(dim=1, keepdim=True))`
* Normalize to [0,1]; recommend percentile clipping (e.g., 99th percentile) to reduce outliers
* Upsample CAM to input resolution (bilinear), then overlay on RGB image

You must handle cases where hooks output tuples/lists (e.g., backbone). Always select the correct tensor.

---

# 8. Output Layout (How to compare)

For each input sample, generate a single grid figure:

Columns: 4 models

* baseline / tr / sae / tr+sae

Rows:

* Row 0: Original RGB (optionally with GT overlay)
* Row 1: CAM-Enc overlay
* Row 2: CAM-ASPPpost overlay
* Row 3: CAM-Decpost overlay
* (Optional Row 4): predicted mask overlay

File naming:

* Per-grid: `{img_id}__{target=T1|T3}.grid.png`
* If saving individual overlays: `{img_id}__{model}__{layer}__{target}.png`

Make sure the grid is consistent in size and ordering for direct inclusion in a paper.

---

# 9. Compatibility: Both Baseline Head and Custom SAETR Head Must Work

Baseline model uses standard `DepthwiseSeparableASPPHead`. Others use `DepthwiseSeparableASPPHeadSAETR` with flags.

Your tool must:

* Detect if `model.decode_head` has `tr_model` / `sae_model` via `hasattr`
* Always hook `bottleneck` and `sep_bottleneck` inputs as the primary aligned points (these exist in the custom head; baseline will have `bottleneck` but may differ—verify and implement fallback behavior)
* If baseline head’s module names differ, implement a small adapter:

  * locate ASPP fusion output point and decoder fusion point in baseline head
  * keep naming “CAM-ASPPpost” and “CAM-Decpost” consistent

---

# 10. Custom Module Registration Requirement (Must not fail to build)

Because `DepthwiseSeparableASPPHeadSAETR` is custom, you must ensure it’s imported before model building.

At script startup (before reading config / building model), do:

* `import mmseg.models.decode_heads.sep_aspp_head_saetr` (or equivalent absolute import)
* Or update `mmseg/models/decode_heads/__init__.py` if required, but prefer not modifying core unless necessary.

---

# 11. Deliverables (What you must produce)

Place the tool in:
`/home/jetson/RB_REC_TOOL/2025-11-17-02/tools/grad_cam`

Deliver:

1. A CLI script, e.g. `tools/grad_cam/run_grad_cam.py`
2. Minimal README usage instructions
3. A small configuration snippet or example command showing how to run four models at once

CLI parameters required:

* `--configs` (4 paths)
* `--checkpoints` (4 paths)
* `--input` (single image path or a text file list)
* `--target` (`T1` or `T3`)
* `--lesion-idx` (default 1)
* `--out-dir`
* `--device` (`cuda:0`)

Acceptance criteria:

* For at least one sample image, the tool generates:

  * `T1.grid.png` and `T3.grid.png`
* Activations have non-None gradients at the chosen hook points
* Outputs are visually aligned and comparable across models