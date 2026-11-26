"""
Analyze and compare benchmark results.

Usage:
    python tools/benchmark/analyze_results.py bench_out_20260105_120000
    python tools/benchmark/analyze_results.py bench_out_*  # Compare multiple runs
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import csv


def load_summary(result_dir: Path) -> Dict[str, Any]:
    """Load summary.json from a benchmark result directory."""
    summary_file = result_dir / 'summary.json'
    if not summary_file.exists():
        raise FileNotFoundError(f"No summary.json found in {result_dir}")
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def format_latency_stats(stats: Dict[str, float]) -> str:
    """Format latency statistics as a compact string."""
    mean = stats.get('mean', 0)
    std = stats.get('std', 0)
    p95 = stats.get('p95', 0)
    return f"{mean:6.1f}±{std:4.1f} (p95={p95:6.1f})"


def print_single_result(result_dir: Path):
    """Print detailed analysis of a single benchmark result."""
    summary = load_summary(result_dir)
    
    print(f"\n{'='*80}")
    print(f"Benchmark Result: {result_dir.name}")
    print(f"{'='*80}")
    
    mode = summary.get('mode', 'unknown')
    images_processed = summary.get('images_processed', 0)
    duration_s = summary.get('duration_s', 0)
    
    print(f"\nMode: {mode}")
    print(f"Images processed: {images_processed}")
    if duration_s > 0:
        print(f"Duration: {duration_s:.1f} s ({duration_s/60:.1f} min)")
        print(f"Throughput: {images_processed/(duration_s/60):.2f} images/min")
    
    # Latency breakdown
    latency = summary.get('latency', {})
    print(f"\n--- Latency Breakdown (ms) ---")
    print(f"End-to-end:      {format_latency_stats(latency.get('e2e', {}))}")
    print(f"  Stage-1:       {format_latency_stats(latency.get('stage1', {}))}")
    print(f"  Stage-2/ROI:   {format_latency_stats(latency.get('stage2_per_roi', {}))}")
    print(f"  Post-proc:     {format_latency_stats(latency.get('post', {}))}")
    
    total_images = latency.get('total_images', 0)
    zero_roi_images = latency.get('images_with_zero_rois', 0)
    total_rois = latency.get('total_rois', 0)
    
    print(f"\nTotal images: {total_images}")
    print(f"Images with 0 ROIs: {zero_roi_images} ({100*zero_roi_images/total_images:.1f}%)")
    print(f"Total ROIs: {total_rois} (avg {total_rois/max(total_images-zero_roi_images,1):.2f} per image)")
    
    # Memory
    memory = summary.get('memory', {})
    print(f"\n--- Memory Usage (GB) ---")
    print(f"Peak VRAM: {memory.get('peak_vram_gb', 0):.3f}")
    print(f"Peak RAM:  {memory.get('peak_ram_gb', 0):.3f}")
    
    # Telemetry
    telemetry = summary.get('telemetry', {})
    print(f"\n--- Power & Thermal ---")
    
    avg_power = telemetry.get('avg_power_w')
    if avg_power is not None:
        print(f"Average power: {avg_power:.2f} W")
    else:
        print(f"Average power: N/A")
    
    max_temp = telemetry.get('max_temp_c')
    if max_temp is not None:
        print(f"Max temperature: {max_temp:.1f} °C")
    else:
        print(f"Max temperature: N/A")
    
    throttling = telemetry.get('throttling', False)
    print(f"Throttling: {throttling}")
    
    sample_count = telemetry.get('sample_count', 0)
    if sample_count > 0:
        print(f"Telemetry samples: {sample_count}")
        max_gpu_clock = telemetry.get('max_gpu_clock_mhz')
        min_gpu_clock = telemetry.get('min_gpu_clock_mhz')
        if max_gpu_clock and min_gpu_clock:
            print(f"GPU clock range: {min_gpu_clock:.0f}-{max_gpu_clock:.0f} MHz")
    
    # Check for CSV files
    latency_csv = result_dir / 'latency.csv'
    if latency_csv.exists():
        with open(latency_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"\n--- Latency CSV ---")
            print(f"Per-image records: {len(rows)}")
            print(f"Available at: {latency_csv}")
    
    print(f"\n{'='*80}\n")


def compare_results(result_dirs: List[Path]):
    """Compare multiple benchmark results in a table."""
    if len(result_dirs) < 2:
        print("Need at least 2 result directories to compare")
        return
    
    print(f"\n{'='*80}")
    print(f"Benchmark Comparison")
    print(f"{'='*80}\n")
    
    summaries = []
    for result_dir in result_dirs:
        try:
            summary = load_summary(result_dir)
            summaries.append((result_dir.name, summary))
        except Exception as e:
            print(f"Warning: Could not load {result_dir}: {e}")
    
    if not summaries:
        print("No valid summaries found")
        return
    
    # Print comparison table
    header = f"{'Metric':<30}"
    for name, _ in summaries:
        header += f" | {name[:20]:>20}"
    print(header)
    print("-" * len(header))
    
    # E2E latency
    row = f"{'E2E mean (ms)':<30}"
    for _, summary in summaries:
        mean = summary.get('latency', {}).get('e2e', {}).get('mean', 0)
        row += f" | {mean:>20.2f}"
    print(row)
    
    row = f"{'E2E p95 (ms)':<30}"
    for _, summary in summaries:
        p95 = summary.get('latency', {}).get('e2e', {}).get('p95', 0)
        row += f" | {p95:>20.2f}"
    print(row)
    
    # Stage-1
    row = f"{'Stage-1 mean (ms)':<30}"
    for _, summary in summaries:
        mean = summary.get('latency', {}).get('stage1', {}).get('mean', 0)
        row += f" | {mean:>20.2f}"
    print(row)
    
    # Stage-2
    row = f"{'Stage-2/ROI mean (ms)':<30}"
    for _, summary in summaries:
        mean = summary.get('latency', {}).get('stage2_per_roi', {}).get('mean', 0)
        row += f" | {mean:>20.2f}"
    print(row)
    
    # Memory
    row = f"{'Peak VRAM (GB)':<30}"
    for _, summary in summaries:
        vram = summary.get('memory', {}).get('peak_vram_gb', 0)
        row += f" | {vram:>20.3f}"
    print(row)
    
    row = f"{'Peak RAM (GB)':<30}"
    for _, summary in summaries:
        ram = summary.get('memory', {}).get('peak_ram_gb', 0)
        row += f" | {ram:>20.3f}"
    print(row)
    
    # Power
    row = f"{'Avg Power (W)':<30}"
    for _, summary in summaries:
        power = summary.get('telemetry', {}).get('avg_power_w')
        if power is not None:
            row += f" | {power:>20.2f}"
        else:
            row += f" | {'N/A':>20}"
    print(row)
    
    # Temperature
    row = f"{'Max Temp (°C)':<30}"
    for _, summary in summaries:
        temp = summary.get('telemetry', {}).get('max_temp_c')
        if temp is not None:
            row += f" | {temp:>20.1f}"
        else:
            row += f" | {'N/A':>20}"
    print(row)
    
    # Throttling
    row = f"{'Throttling':<30}"
    for _, summary in summaries:
        throttling = summary.get('telemetry', {}).get('throttling', False)
        row += f" | {str(throttling):>20}"
    print(row)
    
    # Images processed
    row = f"{'Images processed':<30}"
    for _, summary in summaries:
        images = summary.get('images_processed', 0)
        row += f" | {images:>20}"
    print(row)
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('result_dirs', nargs='+', help='Benchmark result directory/directories')
    parser.add_argument('--compare', action='store_true', 
                        help='Compare multiple results in a table')
    args = parser.parse_args()
    
    # Expand wildcards and collect result directories
    result_dirs = []
    for pattern in args.result_dirs:
        path = Path(pattern)
        if path.is_dir():
            result_dirs.append(path)
        else:
            # Try glob pattern
            parent = path.parent if path.parent.exists() else Path('.')
            matches = list(parent.glob(path.name))
            result_dirs.extend([m for m in matches if m.is_dir()])
    
    if not result_dirs:
        print("Error: No valid result directories found")
        return 1
    
    # Sort by name
    result_dirs = sorted(set(result_dirs))
    
    if args.compare or len(result_dirs) > 1:
        compare_results(result_dirs)
    else:
        print_single_result(result_dirs[0])
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
