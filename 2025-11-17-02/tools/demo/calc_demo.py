import os
from argparse import ArgumentParser

import mmcv
import numpy as np
import torch

from mmseg.apis import inference_model, init_model
from mmseg.utils import get_palette


def parse_args():
    parser = ArgumentParser(description='Segmentation inference with area calculation')
    parser.add_argument('--config', default='deeplabv3plus_all', help='Config file name without extension')
    parser.add_argument('--work_dirs', default='work_dirs', help='Directory that stores checkpoints')
    parser.add_argument('--checkpoint', default='iter_10000.pth', help='Checkpoint file name')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--input-dir', default='datasets/test/images', help='Directory with input images')
    parser.add_argument(
        '--seg-output-dir',
        default='datasets/test/pred_mask',
        help='Directory to save segmentation visualizations')
    parser.add_argument(
        '--palette',
        default='my',
        help='Color palette used for visualization (mmseg palette name or comma separated RGB values)')
    parser.add_argument('--opacity', type=float, default=1.0, help='Opacity of painted segmentation map')
    parser.add_argument('--target-class', type=int, default=1, help='Class index that represents the disease area')
    return parser.parse_args()


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def resolve_palette(palette_arg):
    if isinstance(palette_arg, str):
        if palette_arg == 'my':
            return [[0, 0, 0], [255, 255, 255]]
        if palette_arg.endswith('.npy') and os.path.isfile(palette_arg):
            return mmcv.load(palette_arg).tolist()
        return get_palette(palette_arg)
    return palette_arg


def save_segmentation_mask(seg_map: np.ndarray, palette, out_path: str):
    """Save predicted seg map as an RGB mask image.

    For exact 0/255 distribution, prefer saving as PNG (lossless).
    """
    ensure_dir(os.path.dirname(out_path))
    palette_arr = np.array(palette, dtype=np.uint8)
    if palette_arr.ndim != 2 or palette_arr.shape[1] != 3:
        raise ValueError(f'Invalid palette shape: {palette_arr.shape}')
    if int(seg_map.max()) >= palette_arr.shape[0]:
        raise ValueError(
            f'Palette has {palette_arr.shape[0]} colors but seg_map max is {int(seg_map.max())}')

    mask_rgb = palette_arr[seg_map]  # (H, W, 3) in RGB
    # mmcv.imwrite uses OpenCV backend (BGR). Channel order does not affect 0/255 values,
    # but convert to BGR for consistent viewing.
    mask_bgr = mask_rgb[..., ::-1]
    mmcv.imwrite(mask_bgr, out_path)


def extract_seg_map(result) -> np.ndarray:
    """Extract predicted semantic segmentation map from mmseg 1.x result."""
    if isinstance(result, (list, tuple)):
        result = result[0]
    if not hasattr(result, 'pred_sem_seg') or result.pred_sem_seg is None:
        raise TypeError('inference result has no pred_sem_seg')

    pred = result.pred_sem_seg
    if hasattr(pred, 'data') and pred.data is not None:
        seg = pred.data
    elif hasattr(pred, 'sem_seg') and pred.sem_seg is not None:
        seg = pred.sem_seg
    else:
        raise TypeError('pred_sem_seg has no data/sem_seg field')

    if isinstance(seg, torch.Tensor):
        seg = seg.detach().cpu().numpy()
    seg = np.array(seg)
    # expected shape: (1, H, W) or (H, W)
    if seg.ndim == 3 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim != 2:
        raise ValueError(f'Unexpected seg map shape: {seg.shape}')
    return seg.astype(np.int32)


def compute_leaf_mask(image: np.ndarray) -> np.ndarray:
    return np.any(image != 0, axis=-1)


def main():
    args = parse_args()
    seg_model = init_model(
        os.path.join('configs/my_model_configs', args.config + '.py'),
        os.path.join(args.work_dirs, args.config, args.checkpoint),
        device=args.device)

    input_dir = args.input_dir
    seg_output_dir = args.seg_output_dir + f'/{args.config}'
    ensure_dir(seg_output_dir)
    palette = resolve_palette(args.palette)

    image_files = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    image_files.sort()

    total_leaf_area = 0
    total_disease_area = 0

    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        result = inference_model(seg_model, image_path)

        seg_map = extract_seg_map(result)

        stem, _ = os.path.splitext(file_name)
        vis_path = os.path.join(seg_output_dir, stem + '.png')
        save_segmentation_mask(seg_map, palette, vis_path)

        image = mmcv.imread(image_path)
        leaf_mask = compute_leaf_mask(image)
        disease_mask = seg_map == args.target_class
        disease_on_leaf = disease_mask & leaf_mask

        disease_area = int(np.count_nonzero(disease_on_leaf))
        leaf_area = int(np.count_nonzero(leaf_mask))
        ratio = disease_area / leaf_area if leaf_area > 0 else 0.0

        total_leaf_area += leaf_area
        total_disease_area += disease_area

        print(f'{file_name}: disease area {disease_area} px, leaf area {leaf_area} px, ratio {ratio:.4f}')

    if total_leaf_area > 0:
        total_ratio = total_disease_area / total_leaf_area
        print(
            f'Total disease area {total_disease_area} px, total leaf area {total_leaf_area} px, '
            f'overall ratio {total_ratio:.4f}')
    else:
        print('No leaf pixels detected in the dataset.')


if __name__ == '__main__':
    main()

