#!/usr/bin/env python3
"""Feature visualization hook script for BiformerBlock and SaELayer.

This script:
- Loads an MMSeg model from config + checkpoint
- Runs a single forward pass on an input image
- Registers forward hooks on all BiformerBlock and SaELayer instances (and their
  immediate child layers by default)
- Saves one PNG per layer with the layer name embedded in the filename

Output defaults to: 2025-11-17-02/results/

Example:
  conda run -n rbnet python tools/feature_hooks/vis_biformer_sae_features.py \
    --config configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoint work_dirs/deeplabv3plus_all/iter_10000.pth \
    --input datasets/test/images1024/IMG_20250324_094104.png \
    --out-dir results
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

# Ensure custom modules are importable/registered
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mmseg.models.decode_heads.sep_aspp_head_saetr  # noqa: F401

from mmseg.registry import MODELS
from mmseg.models.decode_heads.sep_aspp_head_saetr import BiformerBlock, SaELayer


def _sanitize(name: str) -> str:
    # Keep filenames readable and filesystem-safe.
    name = name.replace(' ', '_')
    name = name.replace('/', '_')
    name = name.replace('\\', '_')
    name = name.replace(':', '_')
    name = name.replace('[', '_').replace(']', '_')
    name = name.replace('(', '_').replace(')', '_')
    name = name.replace('{', '_').replace('}', '_')
    name = name.replace(',', '_')
    name = name.replace('..', '.')
    name = re.sub(r'[^0-9A-Za-z_.\-]+', '_', name)
    return name.strip('_')


def load_model(config_path: str, checkpoint_path: str, device: str) -> Tuple[nn.Module, Config]:
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    model = MODELS.build(cfg.model)
    model.to(device)
    model.eval()

    load_checkpoint(model, checkpoint_path, map_location=device)
    return model, cfg


def preprocess_image(image_path: str, preprocessor_cfg: Dict[str, Any]) -> Tuple[torch.Tensor, np.ndarray]:
    mean = preprocessor_cfg.get('mean', [0.0, 0.0, 0.0])
    std = preprocessor_cfg.get('std', [1.0, 1.0, 1.0])
    bgr_to_rgb = bool(preprocessor_cfg.get('bgr_to_rgb', True))

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Failed to load image: {image_path}')

    if bgr_to_rgb:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()

    img_float = img_rgb.astype(np.float32)
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_norm = (img_float - mean_arr) / std_arr

    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, img_rgb


def _extract_tensors(output: Any) -> List[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        return [x for x in output if isinstance(x, torch.Tensor)]
    if isinstance(output, dict):
        return [x for x in output.values() if isinstance(x, torch.Tensor)]
    return []


def _to_nchw_4d(x: torch.Tensor) -> torch.Tensor:
    """Convert a 4D tensor to NCHW by inferring channel/spatial dims.

    Works for both NCHW and NHWC (and other 4D layouts) by treating the two
    largest dims among {1,2,3} as spatial dims, and the remaining dim as channel.
    """
    if x.ndim != 4:
        raise ValueError('Expected 4D tensor')

    sizes = list(x.shape)
    dims = [1, 2, 3]
    spatial = sorted(dims, key=lambda d: sizes[d], reverse=True)[:2]
    spatial_set = set(spatial)
    ch_dim = list(set(dims) - spatial_set)
    if len(ch_dim) != 1:
        # Fallback: assume NCHW
        return x
    ch_dim = ch_dim[0]

    # Keep spatial dims in their original order for nicer visualization
    spatial_sorted = sorted(spatial)
    return x.permute(0, ch_dim, spatial_sorted[0], spatial_sorted[1]).contiguous()


def _make_channel_grid(chw: np.ndarray, max_channels: int = 16) -> np.ndarray:
    """Create a single tiled image from CxHxW feature maps."""
    c, h, w = chw.shape
    c = min(c, max_channels)
    chw = chw[:c]

    grid_size = int(np.ceil(np.sqrt(c)))
    canvas = np.zeros((grid_size * h, grid_size * w), dtype=np.float32)

    for idx in range(c):
        r = idx // grid_size
        col = idx % grid_size
        fm = chw[idx]
        # Per-channel normalization for display
        vmin = float(fm.min())
        vmax = float(fm.max())
        if vmax > vmin:
            fm = (fm - vmin) / (vmax - vmin)
        else:
            fm = np.zeros_like(fm)
        canvas[r * h : (r + 1) * h, col * w : (col + 1) * w] = fm

    return canvas


def _save_spatial_feature(
    out_path: str,
    title: str,
    rgb: np.ndarray,
    mean_map: np.ndarray,
    ch_grid: Optional[np.ndarray],
) -> None:
    fig = plt.figure(figsize=(12, 4))

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(rgb)
    ax0.set_title('Input')
    ax0.axis('off')

    ax1 = fig.add_subplot(1, 3, 2)
    m = mean_map
    m = m.astype(np.float32)
    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()
    ax1.imshow(m, cmap='jet')
    ax1.set_title('Channel-mean')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 3)
    if ch_grid is not None:
        ax2.imshow(ch_grid, cmap='gray')
        ax2.set_title('First channels (tiled)')
    else:
        ax2.text(0.5, 0.5, 'N/A', ha='center', va='center')
    ax2.axis('off')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_vector_feature(out_path: str, title: str, vec: np.ndarray) -> None:
    # Keep plots readable
    vec = vec.astype(np.float32)
    if vec.size > 4096:
        step = int(np.ceil(vec.size / 4096))
        vec = vec[::step]

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(vec)
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


class FeatureHooker:
    def __init__(self, out_dir: str, img_id: str, rgb: np.ndarray, max_channels: int = 16):
        self.out_dir = out_dir
        self.img_id = img_id
        self.rgb = rgb
        self.max_channels = int(max_channels)
        self.handles: List[Any] = []
        self.saved: int = 0

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def register(self, module: nn.Module, name: str) -> None:
        def hook(_module: nn.Module, _input: Tuple[Any, ...], output: Any):
            tensors = _extract_tensors(output)
            if not tensors:
                return

            for out_idx, t in enumerate(tensors):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue

                layer_tag = name
                if len(tensors) > 1:
                    layer_tag = f'{name}__out{out_idx}'

                file_name = f'{self.img_id}__{_sanitize(layer_tag)}.png'
                out_path = os.path.join(self.out_dir, file_name)

                try:
                    if t.ndim == 4:
                        t_nchw = _to_nchw_4d(t.detach())
                        # take first sample
                        feat = t_nchw[0].float()
                        mean_map = feat.mean(dim=0).cpu().numpy()
                        ch_grid = _make_channel_grid(feat.cpu().numpy(), max_channels=self.max_channels)
                        _save_spatial_feature(out_path, layer_tag, self.rgb, mean_map, ch_grid)
                        self.saved += 1
                    else:
                        # Reduce to a 1D vector for plotting
                        tt = t.detach().float()
                        v = tt[0]
                        if v.ndim == 0:
                            vec = v.reshape(1)
                        elif v.ndim == 1:
                            vec = v
                        elif v.ndim == 2:
                            # Common for (L, D): average over L
                            vec = v.mean(dim=0)
                        else:
                            # Average all dims except last
                            reduce_dims = tuple(range(0, v.ndim - 1))
                            vec = v.mean(dim=reduce_dims)
                        vec_np = vec.reshape(-1).cpu().numpy()
                        _save_vector_feature(out_path, layer_tag, vec_np)
                        self.saved += 1
                except Exception:
                    # Best-effort: do not crash the run due to a single layer.
                    return

        self.handles.append(module.register_forward_hook(hook))


def iter_target_layers(model: nn.Module, recursive: bool) -> Iterable[Tuple[str, nn.Module]]:
    """Yield (name, module) pairs to hook.

    We hook:
    - every BiformerBlock / SaELayer instance output
    - plus its immediate children (or all descendants if recursive=True)

    Names are fully qualified module paths from model.named_modules().
    """
    for name, module in model.named_modules():
        if isinstance(module, (BiformerBlock, SaELayer)):
            # Hook the module itself
            yield name, module
            if recursive:
                for child_name, child_mod in module.named_modules():
                    if child_name == '':
                        continue
                    yield f'{name}.{child_name}', child_mod
            else:
                for child_name, child_mod in module.named_children():
                    yield f'{name}.{child_name}', child_mod


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize intermediate features for BiformerBlock and SaELayer')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument(
        '--out-dir',
        default='results',
        help='Output directory (relative to 2025-11-17-02 by default)'
    )
    parser.add_argument('--channels', type=int, default=16, help='Number of channels to tile for 4D features')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='If set, hook all descendant layers inside the target modules (can produce many images)'
    )

    args = parser.parse_args()

    # Resolve output directory relative to repo root (2025-11-17-02)
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.config, args.checkpoint, args.device)

    # Prefer config-driven preprocessor to avoid mismatch
    preprocessor_cfg = {}
    if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
        preprocessor_cfg = cfg.model.get('data_preprocessor', {}) or {}

    img_tensor, img_rgb = preprocess_image(args.input, preprocessor_cfg)
    img_id = Path(args.input).stem

    # Match model dtype
    model_dtype = next(model.parameters()).dtype
    img_tensor = img_tensor.to(device=args.device, dtype=model_dtype)

    hooker = FeatureHooker(str(out_dir), img_id, img_rgb, max_channels=args.channels)

    # Register hooks
    hooked = 0
    for layer_name, layer_mod in iter_target_layers(model, recursive=args.recursive):
        hooker.register(layer_mod, layer_name)
        hooked += 1

    if hooked == 0:
        hooker.close()
        raise SystemExit('No BiformerBlock/SaELayer found in this model. Check config (tr/sae flags).')

    # Forward once to trigger hooks
    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        _ = model.decode_head(feats)

    hooker.close()
    print(f'✓ Saved {hooker.saved} feature images to: {out_dir}')


if __name__ == '__main__':
    main()
