import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Dict, Any, Optional

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
        default='work_dirs/mask2former/mask2former.py',
        help='Model config path')
    parser.add_argument(
        '--checkpoint', type=str, default="work_dirs/mask2former/iter_5000.pth", help='Checkpoint file path')
    parser.add_argument(
        '--input-dir', type=str, default="data/coco/all_images", help='Directory with images')
    parser.add_argument(
        '--output-dir',
        type=str,
        default="work_dirs/crop_images",
        help='Directory to store cropped results')
    parser.add_argument(
        '--label-output-dir',
        type=str,
        default="work_dirs/crop_labels",
        help='Directory to store semantic segmentation labels')
    parser.add_argument(
        '--annotation-file',
        type=str,
        default="data/coco/annotations/instances_test2017.json",
        help='COCO format annotation JSON file')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:3',
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
    parser.add_argument(
        '--target-category-id',
        type=int,
        default=2,
        help='Category ID to extract for semantic segmentation labels')
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


def load_coco_annotations(annotation_file: str) -> Dict[str, Any]:
    """Load COCO format annotations from JSON file."""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image_id to filename mapping
    image_id_to_filename = {}
    for img_info in coco_data.get('images', []):
        image_id_to_filename[img_info['id']] = img_info['file_name']
    
    # Build filename to image_id mapping
    filename_to_image_id = {v: k for k, v in image_id_to_filename.items()}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    return {
        'image_id_to_filename': image_id_to_filename,
        'filename_to_image_id': filename_to_image_id,
        'annotations_by_image': annotations_by_image,
        'images': coco_data.get('images', []),
        'categories': coco_data.get('categories', [])
    }


def polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if len(polygon) < 6:  # Need at least 3 points
            continue
        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def get_annotations_in_bbox(annotations: List[Dict], bbox: np.ndarray, 
                            target_category_id: int, img_height: int, 
                            img_width: int) -> List[Dict]:
    """Get annotations of target category that overlap with the given bbox."""
    x1, y1, x2, y2 = bbox
    overlapping_anns = []
    
    for ann in annotations:
        if ann.get('category_id') != target_category_id:
            continue
        
        # Get annotation bbox [x, y, width, height] in COCO format
        ann_bbox = ann.get('bbox', [])
        if len(ann_bbox) != 4:
            continue
        
        ax, ay, aw, ah = ann_bbox
        ax2, ay2 = ax + aw, ay + ah
        
        # Check if annotation bbox overlaps with leaf bbox
        if ax2 >= x1 and ax >= x2:
            continue
        if ay2 >= y1 and ay >= y2:
            continue
        if ax2 < x1 or ax > x2 or ay2 < y1 or ay > y2:
            continue
            
        overlapping_anns.append(ann)
    
    return overlapping_anns


def create_semantic_label(annotations: List[Dict], bbox: np.ndarray,
                          img_height: int, img_width: int,
                          target_category_id: int) -> np.ndarray:
    """Create semantic segmentation label for the cropped region."""
    x1, y1, x2, y2 = bbox.astype(int)
    crop_height = y2 - y1
    crop_width = x2 - x1
    
    if crop_height <= 0 or crop_width <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Create mask for the cropped region
    label_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    
    for ann in annotations:
        if ann.get('category_id') != target_category_id:
            continue
        
        segmentation = ann.get('segmentation', [])
        if not segmentation:
            continue
        
        # Create full image mask
        full_mask = polygon_to_mask(segmentation, img_height, img_width)
        
        # Crop the mask to bbox region
        crop_mask = full_mask[y1:y2, x1:x2]
        
        # Add to label mask (use category_id as label value, or 1 for binary)
        label_mask = np.maximum(label_mask, crop_mask * 1)
    
    return label_mask


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
               scores: np.ndarray, stem: str, out_dir: Path,
               label_out_dir: Optional[Path] = None,
               annotations: Optional[List[Dict]] = None,
               target_category_id: int = 2):
    """Save cropped images and optionally semantic segmentation labels."""
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
        
        # Save cropped image
        out_path = out_dir / f'{stem}_leaf_{idx:02d}_s{score:.2f}.png'
        cv2.imwrite(str(out_path), crop)
        
        # Save semantic segmentation label if annotations provided
        if label_out_dir is not None and annotations is not None:
            # Create semantic label for this crop
            label_mask = create_semantic_label(
                annotations=annotations,
                bbox=clipped_bbox,
                img_height=h,
                img_width=w,
                target_category_id=target_category_id
            )
            
            # Apply the instance mask to the label
            if label_mask.shape == mask_crop.shape:
                label_mask[~mask_crop] = 0
            
            label_path = label_out_dir / f'{stem}_leaf_{idx:02d}_s{score:.2f}_label.png'
            cv2.imwrite(str(label_path), label_mask)


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
    label_output_dir = Path(args.label_output_dir)
    ensure_output_dir(output_dir)
    ensure_output_dir(label_output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist.')

    image_paths = collect_images(input_dir)
    if not image_paths:
        print_log(
            f'No images found in {input_dir}, nothing to process.',
            logger=logger,
            level=logging.WARNING)
        return

    # Load COCO annotations
    coco_data = None
    if args.annotation_file and Path(args.annotation_file).exists():
        print_log(
            f'Loading annotations from {args.annotation_file}',
            logger=logger,
            level=logging.INFO)
        coco_data = load_coco_annotations(args.annotation_file)
    else:
        print_log(
            f'Annotation file not found: {args.annotation_file}, '
            'semantic labels will not be generated.',
            logger=logger,
            level=logging.WARNING)

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

        # Get annotations for current image
        current_annotations = None
        if coco_data is not None:
            image_id = coco_data['filename_to_image_id'].get(img_path.name)
            if image_id is not None:
                current_annotations = coco_data['annotations_by_image'].get(image_id, [])

        save_crops(
            image=orig_img,
            masks=masks[keep],
            bboxes=bboxes[keep],
            scores=scores[keep],
            stem=img_path.stem,
            out_dir=output_dir,
            label_out_dir=label_output_dir if coco_data else None,
            annotations=current_annotations,
            target_category_id=args.target_category_id)

        label_info = ""
        if coco_data and current_annotations:
            cat2_count = sum(1 for ann in current_annotations 
                           if ann.get('category_id') == args.target_category_id)
            label_info = f", {cat2_count} category_{args.target_category_id} annotations"
        
        print_log(
            f'Processed {img_path.name}, saved {keep.sum()} crops{label_info}.',
            logger=logger,
            level=logging.INFO)


if __name__ == '__main__':
    main()

