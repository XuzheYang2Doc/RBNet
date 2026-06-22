## An Instance-Semantic Cascade Framework with RBNet for In-Situ Rice Blast Severity Quantification in Multi-Leaf Field Scenes

<div>
  <strong>Authors:</strong><br>
  Xuzhe Yang<sup>a</sup>, Chuanjing Jin<sup>a</sup>, Juntao Hu<sup>a</sup>, Wanneng Yang<sup>d</sup>,
  Congcong Sun<sup>c</sup>, Qin Gu<sup>b</sup>, Bing Liu<sup>a</sup>, Weixing Cao<sup>a</sup>,
  Yan Zhu<sup>a</sup>, Liujun Xiao<sup>a,*</sup>
</div>

<br>

<div>
  <sup>a</sup> National Engineering and Technology Center for Information Agriculture, Engineering Research Center of Smart Agriculture, Ministry of Education, Key Laboratory for Crop System Analysis and Decision Making, Ministry of Agriculture and Rural Affairs, Jiangsu Key Laboratory for Information Agriculture, Jiangsu Collaborative Innovation Center for Modern Crop Production, Nanjing Agricultural University, Nanjing 210095, Jiangsu, China<br>
  <sup>b</sup> State Key Laboratory of Agricultural and Forestry Biosecurity, College of Plant Protection, Nanjing Agricultural University, Nanjing 211800, Jiangsu, China<br>
  <sup>c</sup> Agricultural Biosystems Engineering Group, Wageningen University, Wageningen 6700AA, The Netherlands<br>
  <sup>d</sup> National Key Laboratory of Crop Genetic Improvement, National Center of Plant Gene Research, Hubei Hongshan Laboratory, Huazhong Agricultural University, Wuhan 430070, Hubei, China
</div>

<br>

<div>
  <sup>*</sup> Corresponding author
</div><hr>

<figure align="center">
  <img src="assets/1.png" alt="Schematic overview of the proposed technical workflow" width="100%">
  <figcaption>
    <strong>Figure 1.</strong> Schematic overview of the proposed technical workflow.
    (A) Sampling sites for dataset images.
    (B) RBNet cascade framework.
    (C) Web application architecture.
    (D) Handheld device architecture.
  </figcaption>
</figure>

<hr>

To enhance the algorithm's usability in real-world applications, we are currently optimizing the code related to post-processing and user experience, and plan to open-source the project once all optimizations are complete. Whether you are a peer reviewer evaluating the algorithmic aspects of a paper or a researcher engaged in rice blast resistance breeding or crop pathology, we are happy to share our current work with you, even though it is not yet perfect. Please feel free to contact the corresponding author at any time; we look forward to discussing and collaborating on related research.

The source code of <strong>RBNet</strong> is currently being organized and will be made publicly available upon completion.

Repository notes:
- Source code and rice blast image dataset will be publicly available once the paper is accepted.
- `tools/analysis/analyze_annotations.py` is used for visual analysis of annotations.
- `tools/analysis/human_compare.py` is used for comparing the assessment results between humans and RBNet.

## Code and Reproducibility Notes

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
paper/                         Local LaTeX manuscript files, ignored by default
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
