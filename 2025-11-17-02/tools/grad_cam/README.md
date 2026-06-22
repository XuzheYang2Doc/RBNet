# Grad-CAM Visualization Tool for DeepLabv3+ Family

This tool generates Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations for comparing four DeepLabv3+ model variants:

1. **Baseline**: Standard DeepLabv3+
2. **TR (BiFormer)**: DeepLabv3+ with BiFormer attention
3. **SAE (ACE)**: DeepLabv3+ with Coordinated Attention Enhancement
4. **TR+SAE**: DeepLabv3+ with both BiFormer and ACE

## Features

- **Multi-model comparison**: Generates side-by-side grids comparing all 4 models
- **Semantically aligned layers**: Hooks into comparable layers across all models:
  - **CAM-Enc**: Backbone encoder output (ResNet layer4)
  - **CAM-ASPPpost**: Post-ASPP fusion (before bottleneck)
  - **CAM-Decpost**: Decoder fusion output (before sep_bottleneck)
- **Segmentation-specific targets**:
  - **T1**: Global lesion logit mean (stable, global attention)
  - **T3**: GT boundary-band weighted lesion logit (boundary-focused, requires GT masks)
- **Paper-ready output**: High-quality comparison grids for publication

## Requirements

The tool requires the following Python packages (already in your MMSegmentation environment):
- PyTorch
- OpenCV (cv2)
- NumPy
- Matplotlib
- MMSegmentation (mmseg)
- MMEngine

## Installation

No additional installation needed. The tool is ready to use in your existing MMSegmentation environment.

## Usage

### Basic Command Structure

```bash
python tools/grad_cam/run_grad_cam.py \
    --configs <config1> <config2> <config3> <config4> \
    --checkpoints <ckpt1> <ckpt2> <ckpt3> <ckpt4> \
    --input <image_path_or_list> \
    --target <T1|T3> \
    --out-dir <output_directory> \
    [--device cuda:0] \
    [--lesion-idx 1] \
    [--gt-dir <gt_mask_directory>] \
    [--band-width 3]
```

### Example: Single Image with T1 Target

```bash
python tools/grad_cam/run_grad_cam.py \
    --configs \
        configs/my_model_configs/deeplabv3plus.py \
        configs/my_model_configs/deeplabv3plus_tr.py \
        configs/my_model_configs/deeplabv3plus_sae.py \
        configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoints \
        work_dirs/deeplabv3plus/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_tr/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_sae/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_all/best_mIoU_iter_10000.pth \
    --input data/rice_leaf_lesion/images/test/sample_001.jpg \
    --target T1 \
    --out-dir results/grad_cam_t1 \
    --device cuda:0 \
    --lesion-idx 1
```

### Example: Multiple Images with T3 Target (Boundary-Focused)

```bash
python tools/grad_cam/run_grad_cam.py \
    --configs \
        configs/my_model_configs/deeplabv3plus.py \
        configs/my_model_configs/deeplabv3plus_tr.py \
        configs/my_model_configs/deeplabv3plus_sae.py \
        configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoints \
        work_dirs/deeplabv3plus/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_tr/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_sae/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_all/best_mIoU_iter_10000.pth \
    --input image_list.txt \
    --target T3 \
    --out-dir results/grad_cam_t3 \
    --gt-dir data/rice_leaf_lesion/masks/test \
    --band-width 3 \
    --device cuda:0 \
    --lesion-idx 1
```

Where `image_list.txt` contains:
```
data/rice_leaf_lesion/images/test/sample_001.jpg
data/rice_leaf_lesion/images/test/sample_002.jpg
data/rice_leaf_lesion/images/test/sample_003.jpg
```

### Example: Process Entire Test Set

```bash
# First, create an image list
find data/rice_leaf_lesion/images/test -name "*.jpg" > test_images.txt

# Run Grad-CAM on all test images
python tools/grad_cam/run_grad_cam.py \
    --configs \
        configs/my_model_configs/deeplabv3plus.py \
        configs/my_model_configs/deeplabv3plus_tr.py \
        configs/my_model_configs/deeplabv3plus_sae.py \
        configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoints \
        work_dirs/deeplabv3plus/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_tr/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_sae/best_mIoU_iter_10000.pth \
        work_dirs/deeplabv3plus_all/best_mIoU_iter_10000.pth \
    --input test_images.txt \
    --target T1 \
    --out-dir results/grad_cam_testset_t1 \
    --device cuda:0
```

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--configs` | Yes | Paths to 4 config files (order: baseline, tr, sae, tr+sae) |
| `--checkpoints` | Yes | Paths to 4 checkpoint files (same order as configs) |
| `--input` | Yes | Single image path or text file with image list |
| `--target` | No | Target type: `T1` (global mean) or `T3` (boundary). Default: `T1` |
| `--lesion-idx` | No | Index of lesion class. Default: `1` |
| `--out-dir` | No | Output directory. Default: `./grad_cam_output` |
| `--device` | No | Device for computation. Default: `cuda:0` |
| `--gt-dir` | No | Directory with GT masks (required for T3). Default: `None` |
| `--band-width` | No | Boundary band width in pixels for T3. Default: `3` |

## Output

For each input image, the tool generates a comparison grid:

```
{image_id}__{target}.grid.png
```

Example: `sample_001__T1.grid.png`, `sample_002__T3.grid.png`

### Grid Layout

The output grid has:
- **Columns**: 4 models (Baseline, BiFormer(TR), ACE(SAE), TR+SAE)
- **Rows**:
  1. Original RGB image
  2. CAM-Enc overlay (backbone output)
  3. CAM-ASPPpost overlay (post-ASPP fusion)
  4. CAM-Decpost overlay (decoder fusion)

Each CAM is overlayed on the original image using a JET colormap (red = high attention, blue = low attention).

## Target Types Explained

### T1: Global Lesion Logit Mean
```python
y = seg_logits[:, lesion_idx, :, :].mean()
```
- **Use case**: Visualizing global attention and feature connectivity
- **Advantage**: Stable across all samples, no GT masks required
- **Interpretation**: Shows which features contribute to the overall lesion prediction

### T3: GT Boundary-Band Weighted Lesion Logit
```python
# Extract 3-5 pixel boundary band from GT lesion mask
band = extract_boundary_band(gt_mask, band_width=3)
# Weight lesion logits by boundary band
y = (seg_logits[:, lesion_idx] * band).sum() / band.sum()
```
- **Use case**: Visualizing boundary-focused attention (ACE module effectiveness)
- **Advantage**: Highlights features that contribute to boundary detection
- **Interpretation**: Shows which features help refine lesion edges
- **Requirement**: Needs GT masks in `--gt-dir`

## Technical Details

### Hook Points

The tool uses PyTorch hooks to capture activations and gradients at semantically aligned points:

1. **CAM-Enc**: `model.backbone` forward hook (captures last feature map from ResNet layer4)
2. **CAM-ASPPpost**: `model.decode_head.bottleneck` forward pre-hook (captures input = post-ASPP fusion)
3. **CAM-Decpost**: `model.decode_head.sep_bottleneck` forward pre-hook (captures input = post-decoder fusion)

These points are carefully chosen to be:
- **Comparable** across all 4 models
- **Gradient-friendly** (no detached paths)
- **Semantically meaningful** (ASPP fusion, decoder fusion, encoder output)

### Grad-CAM Computation

1. Forward pass with hooks capturing activations
2. Backward pass computing gradients w.r.t. target
3. For each layer:
   ```python
   alpha = gradient.mean(dim=(2,3))  # Global average pooling
   cam = relu((alpha * activation).sum(dim=1))  # Weighted sum
   cam = normalize(cam)  # Percentile clipping + [0,1] normalization
   ```

### Gradient Flow Considerations

- The BiFormer module may use `detach()` on routing weights (controlled by `diff_routing` flag)
- This does NOT break Grad-CAM because:
  - We hook at the **output** of BiFormer (which has gradients)
  - The main information pathway (value aggregation) is differentiable
  - Only routing weights are detached, not the feature flow

## Troubleshooting

### Issue: "RuntimeError: element 0 of tensors does not require grad"

**Solution**: Make sure the model is in eval mode but gradients are enabled:
```python
model.eval()
with torch.set_grad_enabled(True):
    # forward pass
```

### Issue: "All gradients are None"

**Cause**: Using inference path that disables gradients (e.g., `inference_model`, `test_step`)

**Solution**: The tool correctly uses `extract_feat` + `decode_head` forward, which preserves gradients.

### Issue: "T3 target requires GT masks"

**Solution**: Provide the `--gt-dir` argument pointing to directory with GT masks. Mask files should have the same name as images (e.g., `sample_001.png` for `sample_001.jpg`).

### Issue: "CAM is all black / no activation"

**Possible causes**:
1. Model didn't converge (check checkpoint)
2. Wrong lesion class index (check `--lesion-idx`)
3. Image preprocessing mismatch (tool uses mean=0, std=1, bgr_to_rgb=True)

**Solution**: Verify model predictions are reasonable first using standard inference.

## Performance Notes

- Each model forward+backward takes ~100-200ms on GPU
- Processing 4 models for 1 image with 3 layers: ~1-2 seconds
- Memory usage: ~2-3GB per model (batch size 1)
- Recommended: Process images sequentially (not in parallel) to avoid OOM

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gradcam_deeplabv3plus,
  title={Grad-CAM Visualization Tool for DeepLabv3+ with Attention Modules},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo}
}
```

## License

This tool is released under the same license as MMSegmentation (Apache 2.0).

## Acknowledgments

- Based on the [Grad-CAM paper](https://arxiv.org/abs/1610.02391) by Selvaraju et al.
- Built on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework
- Inspired by the ablation study methodology in ResNet papers
