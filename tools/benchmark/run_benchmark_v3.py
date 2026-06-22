"""
Benchmark script v3 - Batch processing to minimize model loading overhead.
Strategy: Run all stage-1 inferences first, then all stage-2 inferences.
This avoids repeated model loading (only 2 loads total instead of 2N loads).
"""
import argparse
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import shutil

import cv2
import numpy as np
import torch

# Add benchmark utils to path
sys.path.insert(0, str(Path(__file__).parent))
from profiler import Timer, MemoryTracker, LatencyCollector
from tegrastats import TegrastatsMonitor
from report import print_summary, save_csv_latency, save_csv_telemetry, save_json_summary

_SCRIPT_DIR = Path(__file__).parent.parent.parent


def run_stage1_batch(image_paths: List[str], config: str, checkpoint: str,
                     score_thr: float, device: str, temp_dir: Path,
                     warmup: int = 0) -> List[Dict]:
    """Run stage-1 on all images in one session."""
    sys.path.insert(0, str(_SCRIPT_DIR / "2025-11-17-01"))
    
    import mmcv
    from mmengine import Config
    from mmdet.apis import init_detector, inference_detector
    
    print(f"Loading stage-1 model from {checkpoint}...")
    model = init_detector(config, checkpoint, device=device)
    cfg = Config.fromfile(config)
    
    # Get resize parameters from config
    pipeline = cfg.get('test_pipeline', [])
    if hasattr(cfg, 'test_dataloader'):
        dataset = cfg.test_dataloader.get('dataset', {})
        pipeline = dataset.get('pipeline', pipeline)
    
    resize_scale = None
    keep_ratio = False
    for step in pipeline:
        if isinstance(step, dict) and step.get('type') == 'Resize':
            resize_scale = step.get('scale')
            keep_ratio = step.get('keep_ratio', False)
            break
    
    results = []
    total_images = len(image_paths)
    
    for idx, img_path in enumerate(image_paths):
        is_warmup = idx < warmup
        if is_warmup:
            print(f"  Warmup {idx+1}/{warmup}...", end='\r')
        else:
            actual_idx = idx - warmup
            if (actual_idx + 1) % 50 == 0 or actual_idx == 0:
                print(f"  Stage-1: {actual_idx+1}/{total_images-warmup} images...")
        
        orig_img = mmcv.imread(img_path)
        infer_img = orig_img
        scale_factor = None
        
        if resize_scale:
            if keep_ratio:
                infer_img, scale_factor = mmcv.imrescale(orig_img, resize_scale, return_scale=True)
            else:
                infer_img = mmcv.imresize(orig_img, resize_scale)
                scale_factor = (resize_scale[0] / orig_img.shape[1], resize_scale[1] / orig_img.shape[0])
        
        # Time inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        result = inference_detector(model, infer_img)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - t0) * 1000
        
        # Extract instances
        data_sample = result if hasattr(result, 'pred_instances') else result[0]
        instances = data_sample.pred_instances
        scores = instances.scores.cpu().numpy()
        labels = instances.labels.cpu().numpy()
        bboxes = instances.bboxes.cpu().numpy()
        masks = instances.masks.cpu().numpy() if hasattr(instances.masks, 'cpu') else np.array(instances.masks)
        if hasattr(instances.masks, 'to_ndarray'):
            masks = instances.masks.to_ndarray()
        
        # Scale back to original size
        h, w = orig_img.shape[:2]
        if scale_factor:
            sx, sy = (scale_factor, scale_factor) if np.isscalar(scale_factor) else scale_factor
            if sx > 0 and sy > 0:
                bboxes[:, 0::2] /= sx
                bboxes[:, 1::2] /= sy
                masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), 
                                            interpolation=cv2.INTER_NEAREST) > 0 
                                 for m in masks]) if len(masks) > 0 else np.empty((0, h, w), dtype=bool)
        
        # Filter and save ROIs
        keep = (scores >= score_thr) & (labels == 0)
        rois = []
        for roi_idx, (mask, bbox, score) in enumerate(zip(masks[keep], bboxes[keep], scores[keep])):
            x1 = int(np.clip(np.floor(bbox[0]), 0, w))
            y1 = int(np.clip(np.floor(bbox[1]), 0, h))
            x2 = int(np.clip(np.ceil(bbox[2]), 0, w))
            y2 = int(np.clip(np.ceil(bbox[3]), 0, h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = orig_img[y1:y2, x1:x2].copy()
            mask_crop = mask[y1:y2, x1:x2]
            
            if mask_crop.size == 0:
                continue
            
            crop[~mask_crop] = 0
            
            # Save to temp directory
            crop_file = temp_dir / f"roi_{idx}_{roi_idx}.png"
            mask_file = temp_dir / f"mask_{idx}_{roi_idx}.npy"
            cv2.imwrite(str(crop_file), crop)
            np.save(str(mask_file), mask_crop)
            
            rois.append({
                'crop_file': str(crop_file),
                'mask_file': str(mask_file),
                'bbox': [x1, y1, x2, y2],
                'score': float(score)
            })
        
        if not is_warmup:
            results.append({
                'image_path': img_path,
                'inference_ms': inference_ms,
                'rois': rois
            })
    
    print(f"  Stage-1 complete: processed {len(results)} images")
    
    # Save intermediate results
    import json
    stage1_cache = temp_dir.parent / "stage1_results.json"
    with open(stage1_cache, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Intermediate results saved to {stage1_cache}")
    
    return results


def run_stage2_batch(stage1_results: List[Dict], config: str, checkpoint: str,
                     device: str, output_dir: Path, warmup_done: bool = True) -> List[Dict]:
    """Run stage-2 on all ROIs from stage-1 results."""
    sys.path.insert(0, str(_SCRIPT_DIR / "2025-11-17-02"))
    
    import mmcv
    from mmengine import Config
    from mmseg.apis import init_model, inference_model
    
    print(f"Loading stage-2 model from {checkpoint}...")
    model = init_model(config, checkpoint, device=device)
    cfg = Config.fromfile(config)
    
    # Get class indices from config
    class_names = cfg.get('class_names', ['background', 'lesion'])
    lesion_indices = [i for i, name in enumerate(class_names) if name.lower() == 'lesion']
    
    results = []
    total_rois = sum(len(r['rois']) for r in stage1_results)
    processed_rois = 0
    
    for img_idx, stage1_result in enumerate(stage1_results):
        rois = stage1_result['rois']
        stage2_times = []
        postprocess_times = []
        ds_values = []
        
        for roi in rois:
            crop_file = roi['crop_file']
            mask_file = roi['mask_file']
            
            # Load crop and mask
            crop = mmcv.imread(crop_file)
            leaf_mask = np.load(mask_file)
            
            # Time inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            seg_result = inference_model(model, crop)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - t0) * 1000
            
            # Time postprocessing
            t1 = time.perf_counter()
            
            # Compute DS
            seg_logits = seg_result.seg_logits.data.cpu().numpy()
            pred_mask = seg_logits.argmax(axis=0)
            
            lesion_mask = np.isin(pred_mask, lesion_indices)
            lesion_mask = lesion_mask & leaf_mask
            
            lesion_pixels = int(lesion_mask.sum())
            total_pixels = int(leaf_mask.sum())
            ds = (lesion_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            postprocess_ms = (time.perf_counter() - t1) * 1000
            
            stage2_times.append(inference_ms)
            postprocess_times.append(postprocess_ms)
            ds_values.append(ds)
            
            processed_rois += 1
            if processed_rois % 50 == 0 or processed_rois == total_rois:
                print(f"  Stage-2: {processed_rois}/{total_rois} ROIs...")
        
        results.append({
            'image_path': stage1_result['image_path'],
            'stage1_ms': stage1_result['inference_ms'],
            'stage2_ms_list': stage2_times,
            'postprocess_ms_list': postprocess_times,
            'ds_values': ds_values,
            'num_rois': len(rois)
        })
    
    print(f"  Stage-2 complete: processed {total_rois} ROIs from {len(results)} images")
    
    # Save intermediate results
    import json
    stage2_cache = output_dir / "stage2_results.json"
    with open(stage2_cache, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Intermediate results saved to {stage2_cache}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark two-stage pipeline (batch mode)')
    parser.add_argument('--images', type=str, required=True, help='Image directory or file list')
    parser.add_argument('--stage1-config', type=str, 
                       default='2025-11-17-01/my_config/mask2former.py')
    parser.add_argument('--stage1-checkpoint', type=str,
                       default='2025-11-17-01/work_dirs/mask2former/iter_5000.pth')
    parser.add_argument('--stage2-config', type=str,
                       default='2025-11-17-02/configs/my_model_configs/deeplabv3plus_all.py')
    parser.add_argument('--stage2-checkpoint', type=str,
                       default='2025-11-17-02/work_dirs/deeplabv3plus_all/iter_10000.pth')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Stage-1 score threshold')
    parser.add_argument('--num-images', type=int, default=500, help='Number of images to process')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup paths
    _SCRIPT_DIR_ABS = _SCRIPT_DIR.resolve()
    args.stage1_config = str(_SCRIPT_DIR_ABS / args.stage1_config)
    args.stage1_checkpoint = str(_SCRIPT_DIR_ABS / args.stage1_checkpoint)
    args.stage2_config = str(_SCRIPT_DIR_ABS / args.stage2_config)
    args.stage2_checkpoint = str(_SCRIPT_DIR_ABS / args.stage2_checkpoint)
    
    # Get image list
    images_path = Path(args.images)
    if images_path.is_dir():
        image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
    else:
        with open(images_path) as f:
            image_files = [Path(line.strip()) for line in f if line.strip()]
    
    image_files = [str(f.resolve()) for f in image_files[:args.num_images + args.warmup]]
    
    print(f"Found {len(image_files)} images (including {args.warmup} warmup)")
    print(f"Using device: {args.device}")
    
    # Create temp and output directories
    temp_dir = Path(args.output) / "temp_rois"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Memory tracking
    use_cuda = 'cuda' in args.device
    mem_tracker = MemoryTracker(use_cuda=use_cuda)
    mem_tracker.reset()
    
    # Start tegrastats monitoring
    tegrastats_log = Path(args.output) / "tegrastats.log"
    tegra_monitor = TegrastatsMonitor(interval_ms=1000, log_file=str(tegrastats_log))
    tegra_monitor.start()
    
    # Record total time
    total_start_time = time.perf_counter()
    
    print("\n=== Running Stage-1 (Mask2Former) ===")
    stage1_results = run_stage1_batch(
        image_files, args.stage1_config, args.stage1_checkpoint,
        args.score_thr, args.device, temp_dir, warmup=args.warmup
    )
    
    mem_tracker.update()
    
    print("\n=== Running Stage-2 (DeepLabV3+) ===")
    output_dir = Path(args.output)
    stage2_results = run_stage2_batch(
        stage1_results, args.stage2_config, args.stage2_checkpoint,
        args.device, output_dir
    )
    
    mem_tracker.update()
    
    # Stop tegrastats monitoring and get summary
    tegra_monitor.stop()
    tegra_summary = tegra_monitor.get_summary()
    
    # Record total time
    total_elapsed_sec = time.perf_counter() - total_start_time
    
    # Compute metrics
    print("\n=== Computing Metrics ===")
    e2e_latencies = []
    stage1_latencies = []
    stage2_latencies = []
    postprocess_latencies = []
    
    for result in stage2_results:
        stage1_ms = result['stage1_ms']
        stage2_total_ms = sum(result['stage2_ms_list'])
        postprocess_total_ms = sum(result['postprocess_ms_list'])
        e2e_ms = stage1_ms + stage2_total_ms + postprocess_total_ms
        
        e2e_latencies.append(e2e_ms)
        stage1_latencies.append(stage1_ms)
        stage2_latencies.extend(result['stage2_ms_list'])
        postprocess_latencies.extend(result['postprocess_ms_list'])
    
    # Compute statistics using the compute_stats function directly
    from profiler import compute_stats
    
    e2e_stats = compute_stats(e2e_latencies)
    stage1_stats = compute_stats(stage1_latencies)
    stage2_stats = compute_stats(stage2_latencies)
    postprocess_stats = compute_stats(postprocess_latencies)
    
    # Print simplified summary
    print("\n" + "=" * 80)
    print("Benchmark Summary")
    print("=" * 80)
    print("\n--- Latency Metrics ---")
    print(f"End-to-End  : {e2e_stats['mean']:7.2f}±{e2e_stats['std']:5.2f} ms (p95={e2e_stats['p95']:7.2f} ms, n={e2e_stats['count']})")
    print(f"  Stage-1   : {stage1_stats['mean']:7.2f}±{stage1_stats['std']:5.2f} ms (p95={stage1_stats['p95']:7.2f} ms, n={stage1_stats['count']})")
    print(f"  Stage-2   : {stage2_stats['mean']:7.2f}±{stage2_stats['std']:5.2f} ms (p95={stage2_stats['p95']:7.2f} ms, n={stage2_stats['count']})")
    print(f"  Postproc  : {postprocess_stats['mean']:7.2f}±{postprocess_stats['std']:5.2f} ms (p95={postprocess_stats['p95']:7.2f} ms, n={postprocess_stats['count']})")
    
    print(f"\nTotal Time          : {total_elapsed_sec:.2f} seconds ({total_elapsed_sec/60:.2f} minutes)")
    
    mem_stats = mem_tracker.get_peak_memory_gb()
    print("\n--- Memory Usage ---")
    print(f"Peak VRAM: {mem_stats.get('peak_vram_gb', 0):.2f} GB")
    print(f"Peak RAM : {mem_stats.get('peak_ram_gb', 0):.2f} GB")
    
    # Print power and temperature metrics
    print("\n--- Power & Temperature ---")
    if tegra_summary['avg_power_w'] is not None:
        print(f"Avg Power: {tegra_summary['avg_power_w']:.2f} W")
    else:
        print("Avg Power: N/A (tegrastats unavailable)")
    
    if tegra_summary['max_temp_c'] is not None:
        print(f"Max Temp : {tegra_summary['max_temp_c']:.1f} °C")
    else:
        print("Max Temp : N/A (tegrastats unavailable)")
    
    if tegra_summary.get('sample_count', 0) > 0:
        print(f"Samples  : {tegra_summary['sample_count']}")
    
    print("=" * 80)
    
    # Save results (output_dir already defined above)
    # Build per-image records for CSV
    per_image_records = []
    for result in stage2_results:
        stage2_list = result['stage2_ms_list']
        post_list = result['postprocess_ms_list']
        per_image_records.append({
            'image_name': Path(result['image_path']).name,
            'e2e_ms': result['stage1_ms'] + sum(stage2_list) + sum(post_list),
            'stage1_ms': result['stage1_ms'],
            'stage2_ms_mean': sum(stage2_list) / len(stage2_list) if stage2_list else 0,
            'postprocess_ms_mean': sum(post_list) / len(post_list) if post_list else 0,
            'num_rois': result['num_rois'],
        })
    
    save_csv_latency(per_image_records, output_dir / "latency.csv")
    
    # Save telemetry samples to CSV
    print(f"Saving telemetry data to CSV...")
    tegra_samples = tegra_monitor.get_samples()
    if tegra_samples:
        import csv
        telemetry_csv = output_dir / "telemetry.csv"
        with open(telemetry_csv, 'w', newline='') as f:
            # Determine all available fields from samples
            all_fields = set()
            for sample in tegra_samples:
                all_fields.update(sample.keys())
            
            # Define field order for better readability
            ordered_fields = ['time_sec', 'power_w', 'temp_c', 'gpu_clock_mhz', 'cpu_clock_mhz', 
                            'ram_used_mb', 'ram_total_mb']
            # Add any extra fields not in the ordered list
            extra_fields = sorted(all_fields - set(ordered_fields) - {'timestamp'})
            fieldnames = ordered_fields + extra_fields
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Get start time from first sample
            if tegra_samples:
                start_time = tegra_samples[0]['timestamp']
                for sample in tegra_samples:
                    row = {'time_sec': sample['timestamp'] - start_time}
                    for field in fieldnames:
                        if field != 'time_sec':
                            row[field] = sample.get(field)
                    writer.writerow(row)
        print(f"  Saved {len(tegra_samples)} telemetry samples to: {telemetry_csv}")
    else:
        print("  No telemetry samples available (tegrastats not available or no samples collected)")
    
    # Save JSON summary
    summary_file = output_dir / "summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w') as f:
        json.dump({
            'latency': {
                'e2e': e2e_stats,
                'stage1': stage1_stats,
                'stage2': stage2_stats,
                'postprocess': postprocess_stats
            },
            'total_time_sec': total_elapsed_sec,
            'memory': mem_tracker.get_peak_memory_gb(),
            'telemetry': tegra_summary,
            'config': vars(args),
            'num_images': len(stage2_results),
            'total_rois': sum(r['num_rois'] for r in stage2_results)
        }, f, indent=2)
    print(f"Saved JSON summary to: {summary_file}")
    
    # Cleanup temp files
    print(f"\nCleaning up temp files in {temp_dir}...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✓ Results saved to {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
