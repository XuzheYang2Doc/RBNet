# Grad-CAM Implementation Architecture

## Overview

This document provides technical details about the Grad-CAM implementation for the DeepLabv3+ model family.

## File Structure

```
tools/grad_cam/
├── run_grad_cam.py       # Main Grad-CAM CLI tool
├── check_setup.py        # Pre-flight verification script
├── example_run.sh        # Example usage script
├── README.md             # Comprehensive documentation
├── QUICKSTART.md         # Quick reference guide
├── ARCHITECTURE.md       # This file
└── Agents.md             # Original requirements specification
```

## Core Components

### 1. GradCAMHookManager

**Purpose**: Manages PyTorch hooks to capture activations and gradients.

**Key Methods**:
- `register_hooks(model, model_name)`: Sets up hooks at three strategic points
- `_get_activation_hook(name)`: Forward hook for layer outputs
- `_get_activation_pre_hook(name)`: Forward pre-hook for layer inputs
- `get_gradients(name)`: Retrieves stored gradients

**Hook Strategy**:
```python
# CAM-Enc: Backbone output
model.backbone.register_forward_hook(...)

# CAM-ASPPpost: Input to bottleneck (post-ASPP fusion)
model.decode_head.bottleneck.register_forward_pre_hook(...)

# CAM-Decpost: Input to sep_bottleneck (decoder fusion)
model.decode_head.sep_bottleneck.register_forward_pre_hook(...)
```

### 2. Hook Point Selection Rationale

#### CAM-Enc (Encoder)
- **Location**: `model.backbone` output
- **Captures**: ResNet layer4 features (512 channels @ ~1/8 resolution)
- **Why**: Shows low-level feature activation before ASPP
- **Gradient flow**: ✓ Always differentiable

#### CAM-ASPPpost (ASPP Fusion)
- **Location**: Input to `model.decode_head.bottleneck`
- **Captures**: Concatenated multi-scale ASPP features
- **Why**: Shows multi-scale context aggregation
- **Special handling**: For TR models, captures post-BiFormer features
- **Gradient flow**: ✓ BiFormer output is differentiable

#### CAM-Decpost (Decoder Fusion)
- **Location**: Input to `model.decode_head.sep_bottleneck`
- **Captures**: Concatenated high-res + low-level features
- **Why**: Shows final semantic features before classification
- **Special handling**: For SAE models, captures post-ACE features
- **Gradient flow**: ✓ ACE output is differentiable

### 3. Forward Pass Architecture

```
Input Image [H, W, 3]
    ↓
Data Preprocessor (bgr_to_rgb, mean=0, std=1)
    ↓
Backbone (ResNet18-v1c) → CAM-Enc hook ←
    ↓
ASPP Module (multi-scale dilation)
    ↓
[if TR] BiFormer Attention
    ↓ → CAM-ASPPpost hook ←
Bottleneck Conv
    ↓
Upsample + Concat with C1 features
    ↓
[if SAE] CoordSAE Attention
    ↓ → CAM-Decpost hook ←
Sep Bottleneck Conv
    ↓
Classification Head
    ↓
Segmentation Logits [N, C, H, W]
```

### 4. Gradient Computation Strategy

**Challenge**: MMSeg's standard inference paths disable gradients

**Solution**: Custom gradient-enabled forward pass

```python
# ✗ WRONG - No gradients
with torch.no_grad():
    output = model.test_step(...)

# ✓ CORRECT - Gradients enabled
with torch.set_grad_enabled(True):
    feats = model.extract_feat(img)
    seg_logits = model.decode_head(feats)
    target = compute_target(seg_logits)
    target.backward()
```

### 5. Target Functions

#### T1: Global Lesion Logit Mean

**Formula**: `y = seg_logits[:, lesion_idx, :, :].mean()`

**Characteristics**:
- Simple, stable
- No additional inputs required
- Good for global feature visualization
- Differentiable: ✓

**Use case**: Understanding which features contribute to overall lesion detection

#### T3: Boundary-Band Weighted Lesion Logit

**Formula**: 
```python
band = extract_boundary_band(gt_mask, width=3)
y = (seg_logits[:, lesion_idx] * band).sum() / band.sum()
```

**Characteristics**:
- Boundary-focused
- Requires GT masks
- Highlights edge-relevant features
- Differentiable: ✓

**Use case**: Evaluating ACE module's boundary refinement contribution

### 6. Boundary Band Extraction (Morphological Operations)

**Implementation** (GPU-friendly, no external dependencies):

```python
def extract_boundary_band(mask, band_width=3):
    kernel_size = 2 * band_width + 1
    padding = band_width
    
    # Dilation using max pooling
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    
    # Erosion using -max_pool(-mask)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    
    # Boundary = dilation - erosion
    band = torch.clamp(dilated - eroded, 0, 1)
    
    return band
```

**Advantages**:
- Pure PyTorch (no cv2 dependencies for morphology)
- GPU-accelerated
- Differentiable (though we detach for target computation)

### 7. CAM Computation Algorithm

**Step 1: Global Average Pooling on Gradients**
```python
alpha = gradient.mean(dim=(2,3), keepdim=True)  # [N, C, 1, 1]
```
- Weights each channel by its gradient importance

**Step 2: Weighted Feature Aggregation**
```python
cam = F.relu((alpha * activation).sum(dim=1, keepdim=True))  # [N, 1, H, W]
```
- ReLU ensures only positive contributions

**Step 3: Normalization with Outlier Clipping**
```python
vmax = np.percentile(cam, 99.0)  # Clip top 1% outliers
cam = np.clip(cam, 0, vmax) / vmax
```
- Prevents extreme values from dominating visualization

**Step 4: Upsampling and Overlay**
```python
cam_resized = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
```
- JET colormap: red = high activation, blue = low activation

### 8. Model Compatibility Handling

**Baseline Model**: `DepthwiseSeparableASPPHead`
- Has: `bottleneck`, `sep_bottleneck`
- Doesn't have: `tr_model`, `sae_model`

**Custom Models**: `DepthwiseSeparableASPPHeadSAETR`
- Has: `bottleneck`, `sep_bottleneck`, optional `tr_model`, optional `sae_model`

**Detection Strategy**:
```python
if hasattr(model.decode_head, 'bottleneck'):
    # Hook the bottleneck for all models
    handle = model.decode_head.bottleneck.register_forward_pre_hook(...)
```

### 9. Gradient Flow Verification

**BiFormer (TR) Gradient Concern**:

The BiFormer module has a `diff_routing` flag that controls whether routing weights are differentiable:

```python
def forward(self, query, key):
    if not self.diff_routing:
        query, key = query.detach()  # ⚠️ Detaches routing computation
    # ... routing logic ...
```

**Impact on Grad-CAM**: NONE
- Routing weights are detached, but **value aggregation** is not
- Grad-CAM hooks at BiFormer **output**, which has gradients
- Only routing decisions are non-differentiable, not the feature flow

**Verification**:
```python
assert hook_manager.activations[name].grad is not None
```

### 10. Memory Management

**Per-Image Memory Usage** (approximate):
- Model parameters: ~500MB (ResNet18 + heads)
- Activations (3 layers): ~50-100MB
- Gradients: ~50-100MB
- Input image: ~10MB

**Total per image**: ~1-2GB GPU memory for 4 models

**Optimization**:
- Process models sequentially (not in parallel)
- Clear hooks after each model: `hook_manager.clear()`
- Use `model.zero_grad(set_to_none=True)` to free gradient memory

### 11. Visualization Design

**Grid Layout**:
```
┌─────────┬─────────┬─────────┬─────────┐
│ Baseline│   TR    │   SAE   │ TR+SAE  │  ← Row 0: Original
├─────────┼─────────┼─────────┼─────────┤
│ CAM-Enc │ CAM-Enc │ CAM-Enc │ CAM-Enc │  ← Row 1: Encoder
├─────────┼─────────┼─────────┼─────────┤
│ ASPP-post│ASPP-post│ASPP-post│ASPP-post│  ← Row 2: ASPP
├─────────┼─────────┼─────────┼─────────┤
│ Dec-post│ Dec-post│ Dec-post│ Dec-post│  ← Row 3: Decoder
└─────────┴─────────┴─────────┴─────────┘
```

**Advantages**:
- Direct visual comparison across models
- Same layer across rows (easy horizontal comparison)
- Same model across columns (easy vertical comparison)
- Consistent spatial alignment

### 12. Error Handling

**Common Errors and Solutions**:

| Error | Cause | Solution |
|-------|-------|----------|
| `Gradient is None` | Used inference path | Use `extract_feat` + `decode_head` |
| `Module not found` | Custom head not imported | Import at script start |
| `Checkpoint mismatch` | Config/checkpoint version mismatch | Re-export configs |
| `CUDA OOM` | Too many models in memory | Process sequentially |

### 13. Performance Characteristics

**Timing** (single image, 4 models, 3 layers, GPU):
- Model loading: ~2s (one-time)
- Forward pass per model: ~50ms
- Backward pass per model: ~100ms
- CAM computation: ~10ms
- Visualization: ~50ms

**Total per image**: ~1-2 seconds

**Bottleneck**: Backward pass (gradient computation)

### 14. Testing Strategy

**Unit Tests** (recommended to add):
```python
def test_hook_manager():
    # Verify hooks capture activations
    # Verify gradients are not None

def test_cam_computation():
    # Verify CAM shape matches input
    # Verify normalization to [0,1]

def test_target_functions():
    # Verify T1 computation
    # Verify T3 with boundary band

def test_model_compatibility():
    # Verify works with all 4 models
```

**Integration Tests**:
```bash
# Test with sample image
python tools/grad_cam/run_grad_cam.py \
    --configs [...] \
    --checkpoints [...] \
    --input sample.jpg \
    --target T1
```

### 15. Extension Points

**Adding New Layers**:
```python
# In register_hooks method:
if hasattr(model.decode_head, 'tr_model'):
    handle = model.decode_head.tr_model.register_forward_pre_hook(
        self._get_activation_pre_hook(f'{model_name}_CAM-TRin')
    )
    self.handles.append(handle)
```

**Adding New Targets**:
```python
@staticmethod
def custom_target(seg_logits, **kwargs):
    # Your target computation
    return scalar_loss
```

**Adding New Visualizations**:
```python
def create_comparison_grid_with_predictions(...):
    # Add predicted mask row
    # Add GT mask row
    # Add difference maps
```

## References

- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/)
- [PyTorch Hooks Tutorial](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)

## Maintenance Notes

**When updating MMSeg**:
- Check if `extract_feat` / `decode_head` signatures changed
- Verify hook points still exist
- Test gradient flow

**When modifying decode head**:
- Update hook registration if module names change
- Add new hook points for new modules
- Update architecture diagram

**When adding models**:
- Extend model_names list
- Update grid layout if needed
- Add config validation
