# Grad-CAM Quick Reference

## Quick Start

```bash
cd /home/jetson/RB_REC_TOOL/2025-11-17-02

# Run T1 (global lesion mean) - no GT masks needed
python tools/grad_cam/run_grad_cam.py \
    --configs \
        configs/my_model_configs/deeplabv3plus.py \
        configs/my_model_configs/deeplabv3plus_tr.py \
        configs/my_model_configs/deeplabv3plus_sae.py \
        configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoints \
        work_dirs/deeplabv3plus/iter_10000.pth \
        work_dirs/deeplabv3plus_tr/iter_10000.pth \
        work_dirs/deeplabv3plus_sae/iter_10000.pth \
        work_dirs/deeplabv3plus_all/iter_10000.pth \
    --input /path/to/test_image.jpg \
    --target T1 \
    --out-dir results/grad_cam
```

## Output

- File: `results/grad_cam/{image_name}__T1.grid.png`
- Grid: 4 columns (models) × 4 rows (original + 3 CAM layers)

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--target T1` | Global lesion mean (no GT needed) | Stable, shows global attention |
| `--target T3` | Boundary-weighted (needs `--gt-dir`) | Boundary-focused, shows edge attention |
| `--lesion-idx 1` | Lesion class index | Usually 1 for binary segmentation |
| `--band-width 3` | Boundary band width (T3 only) | 3-5 pixels typical |

## Layers Visualized

1. **CAM-Enc**: Backbone encoder (ResNet layer4) - low-level features
2. **CAM-ASPPpost**: Post-ASPP fusion - multi-scale context
3. **CAM-Decpost**: Decoder fusion - high-level semantic features

## Models Compared

| Column | Model | Description |
|--------|-------|-------------|
| 1 | Baseline | Standard DeepLabv3+ |
| 2 | BiFormer(TR) | + BiFormer attention on ASPP |
| 3 | ACE(SAE) | + Coordinated attention on decoder |
| 4 | TR+SAE | + Both BiFormer and ACE |

## Common Issues

**"Checkpoint not found"**: Update checkpoint paths to match your `work_dirs/` structure

**"T3 needs GT masks"**: Use `--gt-dir` pointing to mask directory, or use `--target T1` instead

**"CUDA OOM"**: Process fewer images at once or use `--device cpu` (slower)

## Tips

- Use **T1** for paper figures showing global model behavior
- Use **T3** for analyzing boundary refinement (ACE effectiveness)
- Process 3-5 representative images per experiment
- Compare before/after training using checkpoints from different iterations
