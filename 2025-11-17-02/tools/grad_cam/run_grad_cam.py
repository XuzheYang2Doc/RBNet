#!/usr/bin/env python3
"""
Grad-CAM visualization tool for DeepLabv3+ family with RBNet-style ablations.

This tool generates Grad-CAM heatmaps for 4 model variants:
- Baseline DeepLabv3+
- DeepLabv3+ + BiFormer (tr)
- DeepLabv3+ + ACE (sae)
- DeepLabv3+ + BiFormer + ACE (tr+sae)

The tool hooks into semantically aligned layers across all models and produces
paper-friendly comparison grids.
"""

import argparse
from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

# Ensure custom modules are imported before building models
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mmseg.models.decode_heads.sep_aspp_head_saetr  # noqa: F401
from mmseg.registry import MODELS


class GradCAMHookManager:
    """Manages forward hooks to capture activations and gradients for Grad-CAM."""

    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.handles = []

    def clear(self):
        """Remove all hooks and clear cached data."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()

    def _get_activation_hook(self, name: str):
        """Create a forward hook to capture activations."""

        def hook(module, input, output):
            # Handle tuple/list outputs (e.g., from backbone)
            if isinstance(output, (tuple, list)):
                # For backbone, take the last feature map
                activation = output[-1]
            else:
                activation = output

            if isinstance(activation, torch.Tensor):
                activation.retain_grad()
                self.activations[name] = activation

        return hook

    def _get_activation_pre_hook(self, name: str):
        """Create a forward pre-hook to capture input activations."""

        def hook(module, input):
            # Input is a tuple of arguments
            if isinstance(input, tuple) and len(input) > 0:
                activation = input[0]
                if isinstance(activation, torch.Tensor):
                    activation.retain_grad()
                    self.activations[name] = activation

        return hook

    def register_hooks(self, model: nn.Module, model_name: str):
        """Register hooks for the three primary CAM locations."""
        self.clear()

        # CAM-Enc: backbone output (last feature map)
        if hasattr(model, 'backbone'):
            handle = model.backbone.register_forward_hook(
                self._get_activation_hook(f'{model_name}_CAM-Enc')
            )
            self.handles.append(handle)

        # CAM-ASPPpost: input to bottleneck (post-ASPP fusion)
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'bottleneck'):
            handle = model.decode_head.bottleneck.register_forward_pre_hook(
                self._get_activation_pre_hook(f'{model_name}_CAM-ASPPpost')
            )
            self.handles.append(handle)

        # CAM-Decpost: input to sep_bottleneck (decoder fusion output)
        if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'sep_bottleneck'):
            handle = model.decode_head.sep_bottleneck.register_forward_pre_hook(
                self._get_activation_pre_hook(f'{model_name}_CAM-Decpost')
            )
            self.handles.append(handle)

    def get_gradients(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve gradients for a specific activation."""
        if name in self.activations:
            activation = self.activations[name]
            if activation.grad is not None:
                return activation.grad
        return None


def compute_grad_cam(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    percentile: float = 99.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap from activation and gradient.

    Args:
        activation: Feature map tensor [N, C, H, W]
        gradient: Gradient tensor [N, C, H, W]
        percentile: Percentile for clipping outliers (default: 99.0)

    Returns:
        CAM heatmap as numpy array [H, W]. If normalize=True, it is normalized to [0, 1].
    """
    # Global average pooling on gradients
    alpha = gradient.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]

    # Weighted combination of feature maps
    cam = F.relu((alpha * activation).sum(dim=1, keepdim=True))  # [N, 1, H, W]

    # Convert to numpy
    cam = cam.squeeze().detach().cpu().numpy()

    if cam.size == 0:
        return np.zeros((1, 1))

    # if normalize:
    #     # Normalize using percentile clipping
    #     cam_flat = cam.flatten()
    #     if len(cam_flat) > 0:
    #         vmax = np.percentile(cam_flat, percentile)
    #         cam = np.clip(cam, 0, vmax)
    #         if vmax > 0:
    #             cam = cam / vmax

    return cam


def normalize_cam(cam: np.ndarray, vmax: float) -> np.ndarray:
    """Normalize a CAM map to [0, 1] with a shared vmax."""
    if cam.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    cam = np.clip(cam, 0, vmax)
    if vmax > 0:
        cam = cam / vmax
    return cam.astype(np.float32)


def extract_boundary_band(
    mask: torch.Tensor, band_width: int = 3, device: str = 'cuda'
) -> torch.Tensor:
    """
    Extract narrow boundary band from GT mask using morphological operations.

    Args:
        mask: Binary mask [H, W] or [N, 1, H, W]
        band_width: Width of boundary band in pixels (default: 3)
        device: Device for computation

    Returns:
        Boundary band mask [N, 1, H, W]
    """
    if mask is None:
        raise ValueError("GT mask is required for boundary extraction. Please provide --gt-dir argument.")
    
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)

    mask = mask.float().to(device)

    # Morphological dilation and erosion using max pooling
    kernel_size = 2 * band_width + 1
    padding = band_width

    # Dilation: max_pool2d
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

    # Erosion: -max_pool2d(-mask)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)

    # Boundary band: dilation - erosion
    band = torch.clamp(dilated - eroded, 0, 1)

    return band


class SegmentationTarget:
    """Defines segmentation-specific targets for Grad-CAM."""

    @staticmethod
    def global_lesion_mean(seg_logits: torch.Tensor, lesion_idx: int = 1) -> torch.Tensor:
        """
        Target T1: Global lesion logit mean.

        Args:
            seg_logits: Segmentation logits [N, C, H, W]
            lesion_idx: Index of lesion class (default: 1)

        Returns:
            Scalar loss value
        """
        return seg_logits[:, lesion_idx, :, :].mean()

    @staticmethod
    def global_lesion_topk_mean(
        seg_logits: torch.Tensor, lesion_idx: int = 1, topk_ratio: float = 0.01
    ) -> torch.Tensor:
        """A sharper variant of T1: mean of top-k lesion logits over spatial locations.

        Args:
            seg_logits: Segmentation logits [N, C, H, W]
            lesion_idx: Index of lesion class
            topk_ratio: Ratio in (0,1], e.g. 0.01 means top 1% pixels

        Returns:
            Scalar loss value
        """
        lesion_map = seg_logits[:, lesion_idx, :, :].reshape(-1)
        total = lesion_map.numel()
        if total == 0:
            return lesion_map.sum()
        k = max(1, int(round(total * float(topk_ratio))))
        topk_vals = torch.topk(lesion_map, k=k, largest=True, sorted=False).values
        return topk_vals.mean()

    @staticmethod
    def boundary_weighted_lesion(
        seg_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        lesion_idx: int = 1,
        band_width: int = 3,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        Target T3: GT boundary-band weighted lesion logit.

        Args:
            seg_logits: Segmentation logits [N, C, H, W]
            gt_mask: Ground truth mask [H, W] or [N, 1, H, W]
            lesion_idx: Index of lesion class (default: 1)
            band_width: Width of boundary band (default: 3)
            device: Device for computation

        Returns:
            Scalar loss value
        """
        # Extract boundary band
        band = extract_boundary_band(gt_mask, band_width, device)

        # Resize band to match logits if needed
        if band.shape[-2:] != seg_logits.shape[-2:]:
            band = F.interpolate(
                band, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False
            )

        # Weighted sum over boundary band
        lesion_logits = seg_logits[:, lesion_idx : lesion_idx + 1, :, :]
        weighted_sum = (lesion_logits * band).sum()
        band_sum = band.sum() + 1e-8

        return weighted_sum / band_sum


def load_model(config_path: str, checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load a model from config and checkpoint.

    Args:
        config_path: Path to model config file
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    cfg = Config.fromfile(config_path)

    # Initialize registry scope
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # Build model
    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()

    # Load checkpoint
    load_checkpoint(model, checkpoint_path, map_location=device)

    return model


def preprocess_image(
    image_path: str, mean: List[float], std: List[float], bgr_to_rgb: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image for model input.

    Args:
        image_path: Path to image file
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        bgr_to_rgb: Whether to convert BGR to RGB

    Returns:
        Preprocessed image tensor [1, 3, H, W] and original RGB image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert to RGB if needed
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

    # Convert to float and normalize
    img_float = img_rgb.astype(np.float32)
    # NOTE: np.array(mean/std) defaults to float64 which would upcast the whole
    # image to float64, producing a torch.double tensor and causing dtype
    # mismatch with model weights (float32).
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_normalized = (img_float - mean_arr) / std_arr

    # Convert to tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img_rgb


def load_gt_mask(mask_path: str, lesion_idx: int = 1) -> Optional[torch.Tensor]:
    """
    Load ground truth mask.

    Args:
        mask_path: Path to mask file
        lesion_idx: Index of lesion class

    Returns:
        Binary lesion mask [H, W] or None if not found
    """
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Convert to binary lesion mask
    lesion_mask = (mask == lesion_idx).astype(np.float32)
    return torch.from_numpy(lesion_mask)


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    blur_ksize: int = 0,
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: RGB image [H, W, 3]
        heatmap: Heatmap [H, W] normalized to [0, 1]
        alpha: Blending factor (default: 0.5)
        colormap: OpenCV colormap (default: COLORMAP_JET)

    Returns:
        Overlayed image [H, W, 3]
    """
    h, w = image.shape[:2]

    # Resize heatmap to match image
    if heatmap.shape != (h, w):
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heatmap_resized = heatmap

    # Convert heatmap to colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    if blur_ksize and blur_ksize > 0:
        # Must be odd for GaussianBlur
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (k, k), 0)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    # heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_rgb = heatmap_colored

    # Blend with original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)

    return overlay


def run_grad_cam_single_model(
    model: nn.Module,
    model_name: str,
    img_tensor: torch.Tensor,
    hook_manager: GradCAMHookManager,
    target_fn,
    cam_percentile: float = 99.0,
    cam_normalize: bool = True,
    device: str = 'cuda',
) -> Dict[str, np.ndarray]:
    """
    Run Grad-CAM on a single model.

    Args:
        model: Model to run Grad-CAM on
        model_name: Name of the model
        img_tensor: Input image tensor [1, 3, H, W]
        hook_manager: Hook manager instance
        target_fn: Target function for backpropagation
        device: Device for computation

    Returns:
        Dictionary of CAM heatmaps {layer_name: heatmap}
    """
    # Match model parameter dtype (usually fp32) to avoid conv dtype mismatch.
    model_dtype = next(model.parameters()).dtype
    img_tensor = img_tensor.to(device=device, dtype=model_dtype)
    img_tensor.requires_grad = True

    # Register hooks
    hook_manager.register_hooks(model, model_name)

    # Forward pass
    with torch.set_grad_enabled(True):
        # Extract features and decode
        feats = model.extract_feat(img_tensor)
        seg_logits = model.decode_head(feats)

        # Resize to input resolution if needed
        if seg_logits.shape[-2:] != img_tensor.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=img_tensor.shape[-2:],
                mode='bilinear',
                align_corners=model.decode_head.align_corners,
            )

        # Compute target
        target = target_fn(seg_logits)

        # Backward pass
        model.zero_grad(set_to_none=True)
        target.backward()

    # Compute CAMs for each hooked layer
    cams = {}
    for layer_name in ['CAM-Enc', 'CAM-ASPPpost', 'CAM-Decpost']:
        full_name = f'{model_name}_{layer_name}'
        if full_name in hook_manager.activations:
            activation = hook_manager.activations[full_name]
            gradient = hook_manager.get_gradients(full_name)

            if gradient is not None:
                cam = compute_grad_cam(
                    activation,
                    gradient,
                    percentile=cam_percentile,
                    normalize=cam_normalize,
                )
                cams[layer_name] = cam

    return cams


def post_normalize_cams_per_layer(
    model_cams: Dict[str, Dict[str, np.ndarray]],
    layers: List[str],
    percentile: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Normalize CAMs with a shared vmax per layer across all models."""
    normalized: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, cams in model_cams.items():
        normalized[model_name] = dict(cams)

    for layer in layers:
        all_vals: List[np.ndarray] = []
        for model_name in model_cams.keys():
            cam = model_cams.get(model_name, {}).get(layer)
            if cam is None:
                continue
            all_vals.append(cam.reshape(-1))
        if not all_vals:
            continue
        stacked = np.concatenate(all_vals, axis=0)
        if stacked.size == 0:
            continue
        vmax = float(np.percentile(stacked, percentile))
        if vmax <= 0:
            vmax = float(stacked.max(initial=0.0))

        for model_name in model_cams.keys():
            cam = model_cams.get(model_name, {}).get(layer)
            if cam is None:
                continue
            normalized[model_name][layer] = normalize_cam(cam, vmax)

    return normalized


def create_comparison_grid(
    image: np.ndarray,
    model_cams: Dict[str, Dict[str, np.ndarray]],
    model_names: List[str],
    target_name: str,
    output_path: str,
    cam_alpha: float = 0.5,
    cam_blur_ksize: int = 0,
):
    """
    Create a comparison grid figure with all models and layers.

    Args:
        image: Original RGB image [H, W, 3]
        model_cams: Dictionary {model_name: {layer_name: heatmap}}
        model_names: List of model names in order
        target_name: Name of target (e.g., 'T1', 'T3')
        output_path: Path to save the grid figure
    """
    n_models = len(model_names)
    layers = ['CAM-Enc', 'CAM-ASPPpost', 'CAM-Decpost']
    n_rows = len(layers) + 1  # +1 for original image row

    fig = plt.figure(figsize=(4 * n_models, 3 * n_rows))
    gs = GridSpec(n_rows, n_models, figure=fig, hspace=0.3, wspace=0.1)

    # Row 0: Original images
    for col, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(image)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Rows 1-3: CAM overlays
    for row_idx, layer_name in enumerate(layers, start=1):
        for col, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[row_idx, col])

            if model_name in model_cams and layer_name in model_cams[model_name]:
                heatmap = model_cams[model_name][layer_name]
                overlay = overlay_heatmap(
                    image,
                    heatmap,
                    alpha=cam_alpha,
                    blur_ksize=cam_blur_ksize,
                )
                ax.imshow(overlay)
            else:
                # Show original if CAM not available
                ax.imshow(image)
                ax.text(
                    0.5,
                    0.5,
                    'N/A',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=14,
                    color='red',
                )

            if col == 0:
                ax.set_ylabel(layer_name, fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.suptitle(f'Grad-CAM Comparison - Target: {target_name}', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Grad-CAM visualization for DeepLabv3+ model family'
    )
    parser.add_argument(
        '--configs', nargs=4, required=True, help='Paths to 4 model config files (in order: baseline, tr, sae, tr+sae)'
    )
    parser.add_argument(
        '--checkpoints', nargs=4, required=True, help='Paths to 4 checkpoint files (same order as configs)'
    )
    parser.add_argument('--input', required=True, help='Input image path or text file with image list')
    parser.add_argument(
        '--target', choices=['T1', 'T3'], default='T1', help='Target type (T1: global mean, T3: boundary)'
    )
    parser.add_argument('--lesion-idx', type=int, default=1, help='Index of lesion class (default: 1)')
    parser.add_argument('--out-dir', default='./grad_cam_output', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--gt-dir', default=None, help='Directory containing ground truth masks (for T3)')
    parser.add_argument('--band-width', type=int, default=3, help='Boundary band width for T3 (default: 3)')

    parser.add_argument(
        '--cam-percentile',
        type=float,
        default=99.0,
        help='Percentile for CAM clipping/normalization (default: 99.0)'
    )
    parser.add_argument(
        '--normalize-scope',
        choices=['per_cam', 'per_layer'],
        default='per_cam',
        help='CAM normalization scope: per_cam (old behavior) or per_layer (shared per layer across models)'
    )
    parser.add_argument(
        '--cam-alpha',
        type=float,
        default=0.5,
        help='Overlay alpha for heatmap (default: 0.5)'
    )
    parser.add_argument(
        '--cam-blur',
        type=int,
        default=0,
        help='Gaussian blur kernel size for heatmap visualization (odd int; 0 disables)'
    )
    parser.add_argument(
        '--t1-topk',
        type=float,
        default=0.0,
        help='If >0, use top-k ratio for T1 target (e.g. 0.01 means top 1%%). 0 keeps global mean.'
    )

    args = parser.parse_args()

    if args.target == 'T3' and not args.gt_dir:
        raise SystemExit('Error: --target T3 requires --gt-dir pointing to GT masks directory.')

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Model names
    model_names = ['Baseline', 'BiFormer(TR)', 'ACE(SAE)', 'TR+SAE']

    # Load models
    print("Loading models...")
    models = []
    for i, (config_path, checkpoint_path) in enumerate(zip(args.configs, args.checkpoints)):
        print(f"  [{i+1}/4] Loading {model_names[i]}: {config_path}")
        model = load_model(config_path, checkpoint_path, args.device)
        models.append(model)

    # Get image list
    if os.path.isfile(args.input) and args.input.endswith('.txt'):
        with open(args.input, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        image_paths = [args.input]

    # Hook manager
    hook_manager = GradCAMHookManager()

    # Process each image
    for img_idx, img_path in enumerate(image_paths):
        print(f"\nProcessing image [{img_idx+1}/{len(image_paths)}]: {img_path}")

        # Load image
        img_tensor, img_rgb = preprocess_image(img_path, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])

        # Get image ID for naming
        img_id = Path(img_path).stem

        # Load GT mask if needed
        gt_mask = None
        if args.target == 'T3' and args.gt_dir:
            mask_path = os.path.join(args.gt_dir, f'{img_id}.png')
            gt_mask = load_gt_mask(mask_path, args.lesion_idx)
            if gt_mask is None:
                print(f"  Warning: GT mask not found at {mask_path}, skipping T3")
                continue

        # Define target function
        if args.target == 'T1':
            if args.t1_topk and args.t1_topk > 0:
                target_fn = lambda logits: SegmentationTarget.global_lesion_topk_mean(
                    logits, args.lesion_idx, args.t1_topk
                )
            else:
                target_fn = lambda logits: SegmentationTarget.global_lesion_mean(logits, args.lesion_idx)
        else:  # T3
            target_fn = lambda logits: SegmentationTarget.boundary_weighted_lesion(
                logits, gt_mask, args.lesion_idx, args.band_width, args.device
            )

        # Run Grad-CAM for each model
        model_cams = {}
        for model, model_name in zip(models, model_names):
            print(f"  Computing Grad-CAM for {model_name}...")
            cams = run_grad_cam_single_model(
                model,
                model_name,
                img_tensor,
                hook_manager,
                target_fn,
                cam_percentile=args.cam_percentile,
                cam_normalize=(args.normalize_scope == 'per_cam'),
                device=args.device,
            )
            model_cams[model_name] = cams

        if args.normalize_scope == 'per_layer':
            layers = ['CAM-Enc', 'CAM-ASPPpost', 'CAM-Decpost']
            model_cams = post_normalize_cams_per_layer(
                model_cams, layers=layers, percentile=args.cam_percentile
            )

        # Create comparison grid
        output_path = os.path.join(args.out_dir, f'{img_id}_{args.target}.{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        print(f"  Saving grid to {output_path}")
        create_comparison_grid(
            img_rgb,
            model_cams,
            model_names,
            args.target,
            output_path,
            cam_alpha=args.cam_alpha,
            cam_blur_ksize=args.cam_blur,
        )

    print(f"\n✓ Done! Results saved to {args.out_dir}")


if __name__ == '__main__':
    main()
