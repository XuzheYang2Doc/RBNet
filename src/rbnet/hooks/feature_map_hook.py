# Copyright (c) OpenMMLab. All rights reserved.
"""Feature map dumping hook used for lightweight layer inspection.

The hook is intentionally simple: it registers forward hooks on the requested
modules, periodically snapshots their outputs, and writes a compact visual
summary to the runner work directory. This keeps the config-side dependency
small while still making the hook importable during training.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS


def _first_tensor(output: Any) -> Optional[torch.Tensor]:
    """Return the first tensor found in a nested output structure."""
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    if isinstance(output, dict):
        for item in output.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor

    return None


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalize an array to the 0-255 range for visualization."""
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    array = array - float(array.min())
    vmax = float(array.max())
    if vmax > 0:
        array = array / vmax
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _tile_channels(chw: np.ndarray, max_channels: int) -> np.ndarray:
    """Tile CxHxW feature maps into a single grayscale canvas."""
    if chw.ndim != 3:
        raise ValueError('Expected a 3D CxHxW tensor for tiling')

    c, h, w = chw.shape
    c = min(c, max_channels)
    chw = chw[:c]

    grid = int(np.ceil(np.sqrt(c)))
    canvas = np.zeros((grid * h, grid * w), dtype=np.uint8)

    for idx in range(c):
        row = idx // grid
        col = idx % grid
        feat = _normalize_to_uint8(chw[idx])
        canvas[row * h:(row + 1) * h, col * w:(col + 1) * w] = feat

    return canvas


def _resolve_module(root_module: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """Resolve a dotted module path from the model root."""
    module = root_module
    for part in layer_path.split('.'):
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            raise AttributeError(f'Cannot resolve layer "{layer_path}"')
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f'Layer "{layer_path}" does not resolve to a module')
    return module


@HOOKS.register_module()
class DzymFeatureMapHook(Hook):
    """Periodically dump compact feature-map visualizations to disk."""

    def __init__(self,
                 interval: int = 50,
                 num_samples: int = 1,
                 layers: Optional[List[str]] = None,
                 mode: str = 'iter',
                 out_dir: Optional[str] = None,
                 max_channels: int = 16) -> None:
        self.interval = int(interval)
        self.num_samples = max(1, int(num_samples))
        self.layers = list(layers or [])
        self.mode = mode
        self.out_dir = out_dir
        self.max_channels = max(1, int(max_channels))

        self._handles: List[Any] = []
        self._latest_features: Dict[str, torch.Tensor] = {}
        self._target_dir: Optional[Path] = None
        self._capture_this_iter = False

        if self.interval <= 0:
            warnings.warn('DzymFeatureMapHook interval <= 0; no samples will be dumped.')

    def before_train(self, runner: Runner) -> None:
        if self.mode in ('iter', 'train'):
            self._attach_hooks(runner)

    def after_train(self, runner: Runner) -> None:
        if self.mode in ('iter', 'train'):
            self._detach_hooks()

    def before_train_iter(self, runner: Runner, batch_idx: int,
                          data_batch: dict) -> None:
        if self.mode != 'iter' or self.interval <= 0:
            self._capture_this_iter = False
            return

        self._capture_this_iter = self.every_n_inner_iters(batch_idx, self.interval)
        if self._capture_this_iter:
            self._latest_features.clear()

    def before_val(self, runner: Runner) -> None:
        if self.mode == 'val':
            self._attach_hooks(runner)

    def after_val(self, runner: Runner) -> None:
        if self.mode == 'val':
            self._detach_hooks()

    def before_test(self, runner: Runner) -> None:
        if self.mode == 'test':
            self._attach_hooks(runner)

    def after_test(self, runner: Runner) -> None:
        if self.mode == 'test':
            self._detach_hooks()

    def before_val_iter(self, runner: Runner, batch_idx: int,
                        data_batch: dict) -> None:
        if self.mode != 'val' or self.interval <= 0:
            self._capture_this_iter = False
            return

        self._capture_this_iter = self.every_n_inner_iters(batch_idx, self.interval)
        if self._capture_this_iter:
            self._latest_features.clear()

    def before_test_iter(self, runner: Runner, batch_idx: int,
                         data_batch: dict) -> None:
        if self.mode != 'test' or self.interval <= 0:
            self._capture_this_iter = False
            return

        self._capture_this_iter = self.every_n_inner_iters(batch_idx, self.interval)
        if self._capture_this_iter:
            self._latest_features.clear()

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                         outputs: Any) -> None:
        if self.mode != 'iter' or self.interval <= 0 or not self._capture_this_iter:
            return
        self._dump_features(runner, tag=f'iter_{runner.iter}')
        self._capture_this_iter = False

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Any) -> None:
        if self.mode != 'val' or self.interval <= 0 or not self._capture_this_iter:
            return
        self._dump_features(runner, tag=f'val_{batch_idx}')
        self._capture_this_iter = False

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Any) -> None:
        if self.mode != 'test' or self.interval <= 0 or not self._capture_this_iter:
            return
        self._dump_features(runner, tag=f'test_{batch_idx}')
        self._capture_this_iter = False

    def _attach_hooks(self, runner: Runner) -> None:
        if self._handles:
            return

        model = getattr(runner.model, 'module', runner.model)
        for layer_name in self.layers:
            try:
                module = _resolve_module(model, layer_name)
            except Exception as exc:
                warnings.warn(f'Failed to resolve feature-map layer "{layer_name}": {exc}')
                continue

            def _make_hook(name: str):
                def _hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
                    if not self._capture_this_iter:
                        return
                    tensor = _first_tensor(output)
                    if tensor is None:
                        return
                    self._latest_features[name] = tensor.detach().cpu()

                return _hook

            self._handles.append(module.register_forward_hook(_make_hook(layer_name)))

        if not self._handles:
            warnings.warn('DzymFeatureMapHook did not attach to any layers.')

    def _detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._latest_features.clear()

    def _prepare_target_dir(self, runner: Runner, tag: str) -> Path:
        base = Path(self.out_dir) if self.out_dir else Path(runner.work_dir) / 'feature_maps'
        target_dir = base / tag
        target_dir.mkdir(parents=True, exist_ok=True)
        self._target_dir = target_dir
        return target_dir

    def _dump_features(self, runner: Runner, tag: str) -> None:
        if not self._latest_features:
            return

        if hasattr(runner, 'rank') and runner.rank not in (0, None):
            return

        target_dir = self._prepare_target_dir(runner, tag)
        for layer_name, tensor in self._latest_features.items():
            if tensor.ndim == 4:
                samples = tensor[:self.num_samples]
            else:
                samples = tensor.unsqueeze(0)[:self.num_samples]

            for sample_idx, sample in enumerate(samples):
                sample_np = sample.numpy()
                base_name = layer_name.replace('.', '_')
                stem = f'{base_name}__sample{sample_idx:02d}'

                np.save(target_dir / f'{stem}.npy', sample_np)

                if sample_np.ndim == 3:
                    mean_map = sample_np.mean(axis=0)
                    mean_u8 = _normalize_to_uint8(mean_map)
                    heatmap = cv2.applyColorMap(mean_u8, cv2.COLORMAP_JET)
                    cv2.imwrite(str(target_dir / f'{stem}.mean.png'), heatmap)

                    grid = _tile_channels(sample_np, self.max_channels)
                    cv2.imwrite(str(target_dir / f'{stem}.channels.png'), grid)
                elif sample_np.ndim == 2:
                    mean_u8 = _normalize_to_uint8(sample_np)
                    heatmap = cv2.applyColorMap(mean_u8, cv2.COLORMAP_JET)
                    cv2.imwrite(str(target_dir / f'{stem}.heatmap.png'), heatmap)
                elif sample_np.ndim == 1:
                    vec = _normalize_to_uint8(sample_np[np.newaxis, :])
                    cv2.imwrite(str(target_dir / f'{stem}.vector.png'), vec)

        self._latest_features.clear()
