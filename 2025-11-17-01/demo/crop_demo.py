import argparse
import os
from pathlib import Path
from typing import List, Sequence

import cv2
import mmcv
import numpy as np
import torch
import logging
from mmengine import Config
from mmengine.logging import MMLogger, print_log

from mmdet.apis import init_detector, inference_detector

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Mask2Former leaf crop demo')
    parser.add_argument(
        '--config',
        type=str,
        default='../my_config/mask2former.py',
        help='Model config path')
    parser.add_argument(
        '--checkpoint', type=str, default="../work_dirs/mask2former/iter_5000.pth", help='Checkpoint file path')
    parser.add_argument(
        '--input-dir', type=str, default="../../2025-11-17-02/datasets/test/images", help='Directory with images')
    parser.add_argument(
        '--output-dir',
        type=str,
        default="../../2025-11-17-02/datasets/test/crop_images",
        help='Directory to store cropped results')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Computation device, e.g. cuda:0 or cpu')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='Minimum score to keep an instance')
    parser.add_argument(
        '--leaf-label',
        type=int,
        default=0,
        help='Label id corresponding to Leaf category')
    return parser.parse_args()


def collect_images(folder: Path) -> List[Path]:
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [
        path for path in sorted(folder.glob('*'))
        if path.suffix.lower() in valid_exts and path.is_file()
    ]
    return images


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), 0, width - 1))
    y2 = int(np.clip(np.ceil(y2), 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return np.array([0, 0, 0, 0])
    return np.array([x1, y1, x2, y2])


def save_crops(image: np.ndarray, masks: np.ndarray, bboxes: np.ndarray,
               scores: np.ndarray, stem: str, out_dir: Path):
    h, w = image.shape[:2]
    for idx, (mask, bbox, score) in enumerate(zip(masks, bboxes, scores)):
        clipped_bbox = clip_bbox(bbox, w, h)
        if clipped_bbox.sum() == 0:
            continue
        x1, y1, x2, y2 = clipped_bbox
        crop = image[y1:y2, x1:x2].copy()
        mask_crop = mask[y1:y2, x1:x2].astype(bool)
        if mask_crop.size == 0:
            continue
        crop[~mask_crop] = 0
        out_path = out_dir / f'{stem}_leaf_{idx:02d}_s{score:.2f}.png'
        cv2.imwrite(str(out_path), crop)


def masks_to_numpy(masks) -> np.ndarray:
    if hasattr(masks, 'to_ndarray'):
        mask_array = masks.to_ndarray()
    elif isinstance(masks, torch.Tensor):
        mask_array = masks.detach().cpu().numpy()
    elif hasattr(masks, 'numpy'):
        mask_array = masks.numpy()
    elif hasattr(masks, 'cpu'):
        mask_array = masks.cpu().numpy()
    elif hasattr(masks, 'detach'):
        mask_array = masks.detach().cpu().numpy()
    else:
        mask_array = np.asarray(masks)
    return mask_array.astype(bool)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    logger = MMLogger.get_instance('crop_demo')

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist.')

    image_paths = collect_images(input_dir)
    if not image_paths:
        print_log(
            f'No images found in {input_dir}, nothing to process.',
            logger=logger,
            level=logging.WARNING)
        return

    model = init_detector(args.config, args.checkpoint, device=args.device)

    resize_cfg = None
    pipeline = None
    if hasattr(cfg, 'test_dataloader'):
        dataset = cfg.test_dataloader.get('dataset', {})
        pipeline = dataset.get('pipeline', cfg.get('test_pipeline', []))
    if pipeline is None:
        pipeline = cfg.get('test_pipeline', [])
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == 'Resize':
            resize_cfg = step
            break

    for img_path in image_paths:
        orig_img = mmcv.imread(str(img_path))
        infer_img = orig_img
        scale_factor = None
        if resize_cfg is not None:
            scale = resize_cfg.get('scale')
            keep_ratio = resize_cfg.get('keep_ratio', False)
            if scale is not None:
                if keep_ratio:
                    infer_img, scale_factor = mmcv.imrescale(
                        orig_img, scale, return_scale=True)
                else:
                    infer_img = mmcv.imresize(orig_img, scale)
                    scale_factor = (scale[0] / orig_img.shape[1],
                                    scale[1] / orig_img.shape[0])

        result = inference_detector(model, infer_img)
        data_sample = result if hasattr(result, 'pred_instances') else result[0]
        instances = data_sample.pred_instances

        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        bboxes = instances.bboxes.detach().cpu().numpy()
        masks = masks_to_numpy(instances.masks)

        if scale_factor is not None:
            if np.isscalar(scale_factor):
                sx = sy = scale_factor
            else:
                sx, sy = scale_factor[0], scale_factor[1]
            if sx > 0 and sy > 0:
                bboxes[:, 0::2] /= sx
                bboxes[:, 1::2] /= sy
                resized_masks = []
                h_ori, w_ori = orig_img.shape[:2]
                for mask in masks:
                    resized = cv2.resize(
                        mask.astype(np.uint8), (w_ori, h_ori),
                        interpolation=cv2.INTER_NEAREST) > 0
                    resized_masks.append(resized.astype(bool))
                masks = np.stack(resized_masks,
                                 axis=0) if resized_masks else np.empty(
                                     (0, h_ori, w_ori), dtype=bool)

        keep = (scores >= args.score_thr) & (labels == args.leaf_label)
        if not np.any(keep):
            print_log(
                f'No Leaf detections for {img_path.name}',
                logger=logger,
                level=logging.INFO)
            continue

        save_crops(
            image=orig_img,
            masks=masks[keep],
            bboxes=bboxes[keep],
            scores=scores[keep],
            stem=img_path.stem,
            out_dir=output_dir)

        print_log(
            f'Processed {img_path.name}, saved {keep.sum()} crops.',
            logger=logger,
            level=logging.INFO)


if __name__ == '__main__':
    main()

