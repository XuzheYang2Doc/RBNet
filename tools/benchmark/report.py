"""Report generation and output utilities."""
import json
import csv
from pathlib import Path
from typing import Dict, Any, List


def format_latency_row(name: str, stats: Dict[str, float], indent: int = 0) -> str:
    """Format a latency metric row."""
    prefix = "  " * indent
    mean = stats.get('mean', 0)
    std = stats.get('std', 0)
    p95 = stats.get('p95', 0)
    count = stats.get('count', 0)
    return f"{prefix}{name:30s}: {mean:7.2f}±{std:5.2f} ms (p95={p95:7.2f} ms, n={count})"


def print_summary(
    latency_stats: Dict[str, Any],
    memory_stats: Dict[str, float],
    telemetry_stats: Dict[str, Any],
    mode: str,
    images_processed: int = 0,
    duration_s: float = 0.0
):
    """Print a compact summary to console."""
    print("\n" + "=" * 80)
    print(f"Benchmark Summary ({mode} mode)")
    print("=" * 80)
    
    # Latency metrics
    print("\n--- Latency Metrics ---")
    print(format_latency_row("End-to-end", latency_stats['e2e']))
    print(format_latency_row("  Stage-1 (Mask2Former)", latency_stats['stage1'], indent=1))
    print(format_latency_row("  Stage-2 (per ROI)", latency_stats['stage2_per_roi'], indent=1))
    print(format_latency_row("  Post-processing", latency_stats['post'], indent=1))
    
    # Additional info
    total_images = latency_stats.get('total_images', 0)
    zero_roi_images = latency_stats.get('images_with_zero_rois', 0)
    total_rois = latency_stats.get('total_rois', 0)
    
    print(f"\n  Total images: {total_images}")
    print(f"  Images with 0 ROIs: {zero_roi_images}")
    print(f"  Total ROIs processed: {total_rois}")
    
    # Memory metrics
    print("\n--- Memory Usage ---")
    peak_vram = memory_stats.get('peak_vram_gb', 0)
    peak_ram = memory_stats.get('peak_ram_gb', 0)
    print(f"  Peak VRAM: {peak_vram:.3f} GB")
    print(f"  Peak RAM: {peak_ram:.3f} GB")
    print(f"  VRAM/RAM: {peak_vram:.3f}/{peak_ram:.3f} GB")
    
    # Telemetry metrics
    print("\n--- Power & Thermal ---")
    avg_power = telemetry_stats.get('avg_power_w')
    max_temp = telemetry_stats.get('max_temp_c')
    throttling = telemetry_stats.get('throttling', False)
    
    if avg_power is not None:
        print(f"  Average power: {avg_power:.2f} W")
    else:
        print(f"  Average power: N/A")
    
    if max_temp is not None:
        print(f"  Max temperature: {max_temp:.1f} °C")
    else:
        print(f"  Max temperature: N/A")
    
    print(f"  Throttling: {throttling}")
    
    # Continuous mode specific
    if mode == 'continuous':
        print(f"\n--- Throughput ---")
        print(f"  Duration: {duration_s:.1f} s")
        print(f"  Images processed: {images_processed}")
        if duration_s > 0:
            throughput = images_processed / (duration_s / 60.0)
            print(f"  Throughput: {throughput:.2f} images/min")
    
    print("\n" + "=" * 80 + "\n")


def save_csv_latency(records: List[Dict[str, Any]], output_path: Path):
    """Save per-image latency records to CSV."""
    if not records:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_name', 'e2e_ms', 'stage1_ms', 'post_ms', 'num_rois',
            'stage2_ms_min', 'stage2_ms_max', 'stage2_ms_mean'
        ])
        writer.writeheader()
        
        for rec in records:
            stage2_list = rec.get('stage2_ms_list', [])
            row = {
                'image_name': rec.get('image_name', ''),
                'e2e_ms': rec.get('e2e_ms', 0),
                'stage1_ms': rec.get('stage1_ms', 0),
                'post_ms': rec.get('post_ms', 0),
                'num_rois': rec.get('num_rois', 0),
                'stage2_ms_min': min(stage2_list) if stage2_list else 0,
                'stage2_ms_max': max(stage2_list) if stage2_list else 0,
                'stage2_ms_mean': sum(stage2_list) / len(stage2_list) if stage2_list else 0,
            }
            writer.writerow(row)
    
    print(f"Saved per-image latency to: {output_path}")


def save_csv_telemetry(samples: List[Dict[str, Any]], output_path: Path):
    """Save telemetry samples to CSV."""
    if not samples:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'power_w', 'temp_c', 'gpu_clock_mhz', 
            'cpu_clock_mhz', 'ram_used_mb', 'ram_total_mb'
        ])
        writer.writeheader()
        writer.writerows(samples)
    
    print(f"Saved telemetry samples to: {output_path}")


def save_json_summary(
    latency_stats: Dict[str, Any],
    memory_stats: Dict[str, float],
    telemetry_stats: Dict[str, Any],
    mode: str,
    images_processed: int,
    duration_s: float,
    output_path: Path
):
    """Save comprehensive summary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'mode': mode,
        'images_processed': images_processed,
        'duration_s': duration_s,
        'latency': latency_stats,
        'memory': memory_stats,
        'telemetry': telemetry_stats,
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved JSON summary to: {output_path}")
