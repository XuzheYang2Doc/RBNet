# Grad-CAM Visualization Tool - Index

Welcome to the Grad-CAM visualization tool for DeepLabv3+ model family!

## 📚 Documentation Guide

Start here based on your needs:

### 🚀 Quick Start
- **New user?** → [QUICKSTART.md](QUICKSTART.md) - Get running in 5 minutes
- **Need examples?** → [example_run.sh](example_run.sh) - Copy-paste ready commands

### 📖 Comprehensive Documentation
- **Full guide** → [README.md](README.md) - Complete documentation with all features
- **Technical details** → [ARCHITECTURE.md](ARCHITECTURE.md) - Implementation deep-dive
- **What's included?** → [SUMMARY.md](SUMMARY.md) - Feature checklist and overview

### 🔧 Setup and Testing
- **Check setup** → Run `python tools/grad_cam/check_setup.py`
- **Basic test** → Run `python tools/grad_cam/test_basic.py`
- **Requirements** → See [README.md#requirements](README.md#requirements)

### 📋 Reference
- **Original spec** → [Agents.md](Agents.md) - Implementation requirements

## 📁 File Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_grad_cam.py` | Main CLI tool | Running Grad-CAM visualization |
| `check_setup.py` | Pre-flight checker | Verifying environment setup |
| `test_basic.py` | Basic syntax test | Quick sanity check |
| `example_run.sh` | Example commands | Learning usage patterns |
| `QUICKSTART.md` | Quick reference | Fast lookup of commands |
| `README.md` | Full documentation | Understanding all features |
| `ARCHITECTURE.md` | Technical guide | Understanding implementation |
| `SUMMARY.md` | Feature overview | Checking what's included |

## 🎯 Common Workflows

### First Time Setup
```bash
# 1. Check if everything is ready
python tools/grad_cam/check_setup.py

# 2. Review examples
bash tools/grad_cam/example_run.sh

# 3. Update checkpoint paths in example_run.sh
nano tools/grad_cam/example_run.sh
```

### Running Grad-CAM
```bash
# Single image, T1 target (global attention)
python tools/grad_cam/run_grad_cam.py \
    --configs config1.py config2.py config3.py config4.py \
    --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth ckpt4.pth \
    --input test_image.jpg \
    --target T1 \
    --out-dir results/

# Multiple images, T3 target (boundary attention)
python tools/grad_cam/run_grad_cam.py \
    [...same as above...] \
    --target T3 \
    --gt-dir path/to/masks/
```

### Analyzing Results
```bash
# Output location
ls results/
# → {image_name}__T1.grid.png
# → {image_name}__T3.grid.png

# View with image viewer
eog results/*.png  # Linux
# or
open results/*.png  # Mac
```

## 🔍 What Does This Tool Do?

Creates **side-by-side comparison grids** showing:
- 4 models: Baseline, BiFormer(TR), ACE(SAE), TR+SAE
- 3 layers: Encoder, ASPP fusion, Decoder fusion
- 2 targets: Global (T1) and Boundary-focused (T3)

**Output**: Paper-ready figures comparing attention patterns across models

## ✅ Feature Highlights

- ✅ Semantically aligned layer hooks across all models
- ✅ Gradient-enabled forward pass (no inference pitfalls)
- ✅ Segmentation-specific targets (not classification)
- ✅ GPU-accelerated boundary extraction (pure PyTorch)
- ✅ Handles BiFormer routing gradients correctly
- ✅ Paper-friendly grid layouts
- ✅ Comprehensive documentation

## 🆘 Getting Help

**Issue**: "How do I run this?"
→ [QUICKSTART.md](QUICKSTART.md)

**Issue**: "What do the parameters mean?"
→ [README.md#arguments](README.md#arguments)

**Issue**: "How does it work internally?"
→ [ARCHITECTURE.md](ARCHITECTURE.md)

**Issue**: "Checkpoint not found"
→ Update paths in `example_run.sh`

**Issue**: "T3 needs GT masks"
→ Use `--target T1` or provide `--gt-dir`

**Issue**: "Import errors"
→ Run `python tools/grad_cam/check_setup.py`

## 📊 Expected Output

For input `sample_001.jpg`, generates:

```
results/
├── sample_001__T1.grid.png     # Global attention comparison
└── sample_001__T3.grid.png     # Boundary attention comparison (if GT available)
```

Each grid shows:
```
┌─────────────────────────────────────────────┐
│  Baseline │   TR    │   SAE   │  TR+SAE    │ ← Original images
├─────────────────────────────────────────────┤
│  CAM-Enc  │ CAM-Enc │ CAM-Enc │  CAM-Enc   │ ← Encoder layer
├─────────────────────────────────────────────┤
│ ASPPpost  │ASPPpost │ASPPpost │ ASPPpost   │ ← ASPP fusion
├─────────────────────────────────────────────┤
│ Decpost   │ Decpost │ Decpost │  Decpost   │ ← Decoder fusion
└─────────────────────────────────────────────┘
```

## 🎓 Understanding the Outputs

**High activation (red)**: Model focuses attention here for prediction  
**Low activation (blue)**: Model ignores this region  

**Comparing columns**: See how different modules (TR/SAE) change attention  
**Comparing rows**: See how attention evolves through network layers  

**T1 vs T3**: T1 shows global connectivity, T3 shows boundary sensitivity

## 🚀 Next Steps

1. ✅ Tool is implemented and ready
2. ⬜ Run `check_setup.py` to verify environment
3. ⬜ Update checkpoint paths in `example_run.sh`
4. ⬜ Test on sample image with T1 target
5. ⬜ Generate figures for paper

## 📞 Support

For detailed troubleshooting, see:
- [README.md#troubleshooting](README.md#troubleshooting)
- [ARCHITECTURE.md#error-handling](ARCHITECTURE.md#error-handling)

---

**Version**: 1.0  
**Date**: 2026-01-11  
**Status**: ✅ Ready to Use  
