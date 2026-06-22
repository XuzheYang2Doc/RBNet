# RBNet

Code for **An Instance-Semantic Cascade Framework with RBNet for In-Situ Rice Blast Severity Quantification in Multi-Leaf Field Scenes**.

RBNet uses a two-stage pipeline for field images with overlapping rice leaves:

1. **Leaf instance segmentation** with Mask2Former. This stage isolates each leaf instance and produces masked leaf crops.
2. **Lesion semantic segmentation** with RBNet, a DeepLabV3+ variant with BiFormer routing and an adaptive coordinate-excitation branch.
3. **Severity estimation** from the ratio of lesion pixels to leaf pixels for each isolated leaf.

## Repository Layout

```text
configs/
  instance/                    Mask2Former leaf instance segmentation config
  semantic/                    RBNet and semantic baseline configs
src/rbnet/
  datasets/                    Rice blast lesion dataset registration
  hooks/                       Feature-map dumping hook
  models/                      RBNet semantic head
tools/
  data/                        Annotation conversion utilities
  inference/                   Leaf cropping and severity inference scripts
  analysis/                    Annotation, feature, attention, and human-comparison tools
  benchmark/                   Jetson benchmark utilities
assets/                        Paper and README figures
paper/                         LaTeX manuscript files, ignored by default
```

The original dated framework dumps have been removed from the project tree. Install MMDetection and MMSegmentation as dependencies instead of keeping their source code inside this repository.

## Environment

Tested dependencies are listed in `requirements.txt`. A typical setup is:

```bash
conda create -n rbnet python=3.10 -y
conda activate rbnet
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

If `mmcv` installation fails on your CUDA/PyTorch combination, install OpenMMLab packages with `mim`:

```bash
pip install -U openmim
mim install mmcv==2.2.0
pip install mmdet==3.3.0 mmsegmentation==1.2.2
```

## Data Layout

Expected paths are relative to the repository root.

```text
data/instance/
  train2017/
  val2017/
  test2017/
  annotations/
    instances_train2017.json
    instances_val2017_no_lesion.json
    instances_test2017_no_lesion.json

data/semantic/
  train/images1024/
  train/labels1024/
  val/images1024/
  val/labels1024/
  test/images1024/
  test/labels1024/
```

For LabelMe annotations, convert polygons to COCO instance JSON with:

```bash
python tools/data/convert_labelme_to_coco.py
```

## Training

Leaf instance segmentation:

```bash
mim train mmdet configs/instance/mask2former_leaf.py --work-dir work_dirs/mask2former_leaf
```

RBNet lesion segmentation:

```bash
mim train mmseg configs/semantic/deeplabv3plus_all.py --work-dir work_dirs/deeplabv3plus_all
```

Useful semantic baselines are also kept under `configs/semantic/`, including DeepLabV3+, PSPNet, U-Net, UPerNet, SegFormer, Swin, and RBNet ablations.

## Inference

First crop leaf instances with the Mask2Former model:

```bash
python tools/inference/crop_leaf_instances.py \
  --config configs/instance/mask2former_leaf.py \
  --checkpoint checkpoints/mask2former_leaf.pth \
  --input-dir data/instance/test2017 \
  --output-dir outputs/leaf_crops/images \
  --label-output-dir outputs/leaf_crops/labels
```

Then run lesion segmentation and export severity values:

```bash
python tools/inference/compute_severity.py
```

The script writes visual masks to `outputs/semantic_masks/` and severity values to `outputs/severity/semantic_results.xlsx`.

## Citation

If you use this code, please cite the paper once the final bibliographic information is available.
