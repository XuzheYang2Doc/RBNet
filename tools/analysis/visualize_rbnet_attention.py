#!/usr/bin/env python3
"""Overlay attention heatmaps for BiFormer (routing) and SAE (CoordAtt) on the input image.

What this script visualizes (only spatially visualizable maps):
- BiFormerBlock / BiLevelRoutingAttention: routing weights (per window) mapped to a coarse
  spatial grid then upsampled and overlaid on the original image.
- CoordSaeLayer / CoordAtt: spatial attention map (a_w * a_h), averaged over channels
  then overlaid on the original image.

Notes:
- SaELayer produces channel-only weights (C,1,1). This is NOT a spatial attention map;
  by default we skip it to satisfy “only draw visualizable maps”.

Outputs one PNG per attention source under outputs/analysis by default.

Example:
  conda run -n rbnet python tools/feature_hooks/vis_biformer_sae_attention.py \
    --config configs/my_model_configs/deeplabv3plus_all.py \
    --checkpoint work_dirs/deeplabv3plus_all/iter_10000.pth \
    --input data/semantic/test/images1024/IMG_20250324_094104.png \
    --out-dir results
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
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
from mmseg.models.decode_heads.sep_aspp_head_saetr import BiLevelRoutingAttention, CoordAtt


def _sanitize(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
    name = name.replace('[', '_').replace(']', '_')
    name = name.replace('(', '_').replace(')', '_')
    name = name.replace('{', '_').replace('}', '_')
    name = name.replace(',', '_')
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


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap01: np.ndarray,
    alpha: float,
    colormap: int = cv2.COLORMAP_JET,
    blur_ksize: int = 0,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    hm = heatmap01
    if hm.shape != (h, w):
        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)

    hm = hm.astype(np.float32)
    hm = np.clip(hm, 0.0, 1.0)
    hm_u8 = (hm * 255).astype(np.uint8)

    if blur_ksize and blur_ksize > 0:
        k = int(blur_ksize)
        if k % 2 == 0:
            k += 1
        hm_u8 = cv2.GaussianBlur(hm_u8, (k, k), 0)

    hm_color = cv2.applyColorMap(hm_u8, colormap)
    hm_rgb = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    out = cv2.addWeighted(image_rgb, 1 - alpha, hm_rgb, alpha, 0)
    return out


def normalize_percentile(x: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Normalize to [0,1] with low/high percentile clipping.

    This improves contrast when there are outliers at the min or max.
    """
    x = x.astype(np.float32)
    x = np.maximum(x, 0)
    flat = x.reshape(-1)
    if flat.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    lp = float(low_percentile)
    hp = float(high_percentile)
    if hp < lp:
        hp, lp = lp, hp

    vmin = float(np.percentile(flat, lp))
    vmax = float(np.percentile(flat, hp))
    if vmax < vmin:
        vmax = vmin

    # Clip only high outliers, then min-max scale.
    x = np.clip(x, vmin, vmax)
    denom = vmax - vmin
    if denom <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / denom).astype(np.float32)


def _stats(name: str, x: np.ndarray) -> str:
    x = np.asarray(x)
    if x.size == 0:
        return f'{name}: empty'
    return (
        f'{name}: shape={tuple(x.shape)} min={float(x.min()):.6g} '
        f'max={float(x.max()):.6g} mean={float(x.mean()):.6g} std={float(x.std()):.6g}'
    )


class AttentionCollector:
    def __init__(self, percentile: float = 99.0, biformer_metric: str = 'max'):
        # BiFormer routing weights: name -> (routing_map_2d, feat_hw)
        self.biformer_routing: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        # BiFormer pixel-level attn confidence: name -> (attn_map_2d, feat_hw)
        self.biformer_attn: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        # CoordAtt attention: name -> (attn_map_2d, feat_hw)
        self.coord_att: Dict[str, Tuple[np.ndarray, Tuple[int, int]]] = {}
        self.handles: List[Any] = []
        self._patched: List[Tuple[BiLevelRoutingAttention, Any]] = []
        self.percentile = float(percentile)
        if biformer_metric not in ('max', 'entropy'):
            raise ValueError('biformer_metric must be one of: max, entropy')
        self.biformer_metric = biformer_metric

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for module, old_forward in self._patched:
            module.forward = old_forward
        self._patched.clear()

    def patch_bilevel_routing(self, module: BiLevelRoutingAttention, name: str) -> None:
        old_forward = module.forward

        def forward_wrapped(x: torch.Tensor, ret_attn_mask: bool = False):
            # Always request masks so we can visualize routing. Return only `out` to keep model behavior.
            out, r_weight, _r_idx, attn_weight = old_forward(x, ret_attn_mask=True)

            # x is NHWC. Use its spatial resolution.
            n, h, w, _c = x.shape
            # r_weight: (n, p2, topk) where p2 = n_win*n_win (window grid)
            # NOTE: r_weight is a softmax over topk, so mean(topk) is always 1/topk (constant).
            # Use a non-trivial metric to get spatial variation.
            rw = r_weight.detach().float().cpu().numpy()
            if self.biformer_metric == 'max':
                # peakedness of topk distribution (non-constant)
                rw2 = rw.max(axis=-1)  # (n, p2)
            else:
                # confidence = 1 - normalized entropy (non-constant)
                eps = 1e-12
                p = np.clip(rw, eps, 1.0)
                ent = -(p * np.log(p)).sum(axis=-1)
                ent = ent / np.log(p.shape[-1])
                rw2 = 1.0 - ent  # (n, p2)
            # Infer window grid size as sqrt(p2)
            p2 = rw2.shape[1]
            g = int(round(p2**0.5))
            if g * g == p2:
                coarse = rw2.reshape(n, g, g)
                self.biformer_routing[name] = (coarse[0], (h, w))

            # Also derive a pixel-level attention confidence map from attn_weight.
            # attn_weight: (n*p2, heads, q, k) where softmax is over k.
            # Use max(k) or entropy(k) per query pixel to avoid constant mean.
            if isinstance(attn_weight, torch.Tensor) and attn_weight.ndim == 4:
                aw = attn_weight.detach().float()  # (n*p2, m, q, k)
                if self.biformer_metric == 'max':
                    conf = aw.max(dim=3).values.mean(dim=1)  # (n*p2, q)
                else:
                    # confidence = 1 - normalized entropy over k
                    eps_t = 1e-12
                    p_t = torch.clamp(aw, min=eps_t)
                    ent_t = -(p_t * torch.log(p_t)).sum(dim=3)  # (n*p2, m, q)
                    ent_t = ent_t / float(np.log(aw.shape[3]))
                    conf = (1.0 - ent_t).mean(dim=1)  # (n*p2, q)

                # Reconstruct window layout back to (n, h, w)
                n_win = int(getattr(module, 'n_win', 1))
                if n_win > 0 and h % n_win == 0 and w % n_win == 0:
                    wh = h // n_win
                    ww = w // n_win
                    q = wh * ww
                    p2_expected = n_win * n_win
                    if conf.numel() == n * p2_expected * q:
                        conf_map = conf.view(n, p2_expected, wh, ww)
                        conf_map = conf_map.view(n, n_win, n_win, wh, ww)
                        conf_map = conf_map.permute(0, 1, 3, 2, 4).contiguous().view(n, h, w)
                        self.biformer_attn[name] = (conf_map[0].cpu().numpy(), (h, w))
            return out

        module.forward = forward_wrapped
        self._patched.append((module, old_forward))

    def hook_coord_att(self, module: CoordAtt, name: str) -> None:
        def hook(_m: nn.Module, _inp: Tuple[Any, ...], out: Any):
            if not isinstance(out, torch.Tensor) or out.ndim != 4:
                return
            # out: (N, C, H, W) attention weights
            attn = out.detach().float()[0]
            attn2d = attn.mean(dim=0).cpu().numpy()  # (H, W)
            self.coord_att[name] = (attn2d, (int(out.shape[2]), int(out.shape[3])))

        self.handles.append(module.register_forward_hook(hook))


def main() -> None:
    parser = argparse.ArgumentParser(description='Overlay attention heatmaps for BiFormer/SAE modules')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--device', default='cuda:0', help='Device (default: cuda:0)')
    parser.add_argument('--out-dir', default='results', help='Output directory (relative to the repository root)')
    parser.add_argument('--alpha', type=float, default=0.45, help='Overlay alpha (default: 0.45)')
    parser.add_argument('--blur', type=int, default=3, help='Gaussian blur kernel size (0 disables)')
    parser.add_argument(
        '--percentile',
        type=float,
        default=99.0,
        help='High percentile for attention normalization (default: 99.0)'
    )
    parser.add_argument(
        '--low-percentile',
        type=float,
        default=1.0,
        help='Low percentile for attention normalization (default: 1.0)'
    )
    parser.add_argument(
        '--biformer-metric',
        choices=['max', 'entropy'],
        default='max',
        help='How to convert BiFormer routing weights (softmax over topk) to a spatial strength map.'
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.config, args.checkpoint, args.device)

    preprocessor_cfg: Dict[str, Any] = {}
    if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
        preprocessor_cfg = cfg.model.get('data_preprocessor', {}) or {}

    img_tensor, img_rgb = preprocess_image(args.input, preprocessor_cfg)
    img_id = Path(args.input).stem

    model_dtype = next(model.parameters()).dtype
    img_tensor = img_tensor.to(device=args.device, dtype=model_dtype)

    collector = AttentionCollector(percentile=args.percentile, biformer_metric=args.biformer_metric)

    # Register collectors
    found_biformer = 0
    found_coordatt = 0
    for name, module in model.named_modules():
        if isinstance(module, BiLevelRoutingAttention):
            collector.patch_bilevel_routing(module, name)
            found_biformer += 1
        elif isinstance(module, CoordAtt):
            collector.hook_coord_att(module, name)
            found_coordatt += 1

    if found_biformer == 0 and found_coordatt == 0:
        collector.close()
        raise SystemExit('No BiLevelRoutingAttention or CoordAtt found in this model. Check tr/sae flags.')

    # Run forward once
    with torch.no_grad():
        feats = model.extract_feat(img_tensor)
        _ = model.decode_head(feats)

    # Save overlays
    saved = 0

    # BiFormer routing overlays (coarse window grid)
    for name, (coarse_grid, _feat_hw) in collector.biformer_routing.items():
        hm = normalize_percentile(coarse_grid, args.low_percentile, args.percentile)
        print(_stats(f'BiFormer routing raw {name}', coarse_grid))
        print(_stats(f'BiFormer routing norm {name}', hm))
        overlay = overlay_heatmap(img_rgb, hm, alpha=args.alpha, blur_ksize=args.blur)
        out_path = out_dir / f'{img_id}__{_sanitize(name)}__biformer_routing.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved += 1

    # BiFormer pixel-level attention confidence overlays
    for name, (attn_map, _feat_hw) in collector.biformer_attn.items():
        hm = normalize_percentile(attn_map, args.low_percentile, args.percentile)
        print(_stats(f'BiFormer attn raw {name}', attn_map))
        print(_stats(f'BiFormer attn norm {name}', hm))
        overlay = overlay_heatmap(img_rgb, hm, alpha=args.alpha, blur_ksize=args.blur)
        out_path = out_dir / f'{img_id}__{_sanitize(name)}__biformer_attn_conf.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved += 1

    # CoordAtt overlays
    for name, (attn2d, _feat_hw) in collector.coord_att.items():
        hm = normalize_percentile(attn2d, args.low_percentile, args.percentile)
        print(_stats(f'CoordAtt raw {name}', attn2d))
        print(_stats(f'CoordAtt norm {name}', hm))
        overlay = overlay_heatmap(img_rgb, hm, alpha=args.alpha, blur_ksize=args.blur)
        out_path = out_dir / f'{img_id}__{_sanitize(name)}__coordatt.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved += 1

    collector.close()
    print(f'✓ Saved {saved} attention overlay images to: {out_dir}')


if __name__ == '__main__':
    main()
