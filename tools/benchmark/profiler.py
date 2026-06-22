"""Profiling utilities for timing and quantile computation."""
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import psutil


class Timer:
    """Context manager for timing operations with GPU synchronization."""
    
    def __init__(self, use_cuda: bool = True, sync: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.sync = sync
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
        
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
    
    def __enter__(self):
        if self.use_cuda:
            if self.sync:
                torch.cuda.synchronize()
            self.start_event.record()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda:
            self.end_event.record()
            if self.sync:
                torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.end_time = time.perf_counter()
            self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return False
    
    def elapsed(self) -> float:
        """Return elapsed time in milliseconds."""
        return self.elapsed_ms if self.elapsed_ms is not None else 0.0


class MemoryTracker:
    """Track peak memory usage (VRAM and RAM)."""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.process = psutil.Process()
        self.initial_ram_mb = None
        self.peak_ram_mb = None
        self.initial_vram_mb = None
        self.peak_vram_alloc_mb = None
        self.peak_vram_reserved_mb = None
    
    def reset(self):
        """Reset memory tracking."""
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        self.initial_ram_mb = self.process.memory_info().rss / (1024 ** 2)
        self.peak_ram_mb = self.initial_ram_mb
        if self.use_cuda:
            self.initial_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    
    def update(self):
        """Update peak memory stats."""
        current_ram = self.process.memory_info().rss / (1024 ** 2)
        self.peak_ram_mb = max(self.peak_ram_mb or 0, current_ram)
        
        if self.use_cuda:
            self.peak_vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            self.peak_vram_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    
    def get_peak_memory_gb(self) -> Dict[str, float]:
        """Return peak memory usage in GB."""
        self.update()
        result = {
            'peak_ram_gb': (self.peak_ram_mb or 0) / 1024.0,
        }
        if self.use_cuda:
            result['peak_vram_gb'] = (self.peak_vram_alloc_mb or 0) / 1024.0
            result['peak_vram_reserved_gb'] = (self.peak_vram_reserved_mb or 0) / 1024.0
        return result


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, median, p95 for a list of values."""
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(values)
    }


class LatencyCollector:
    """Collect and compute statistics for various latency measurements."""
    
    def __init__(self):
        self.e2e_latencies: List[float] = []  # end-to-end
        self.stage1_latencies: List[float] = []  # stage-1 inference
        self.stage2_latencies: List[float] = []  # stage-2 inference per ROI
        self.post_latencies: List[float] = []  # post-processing
        self.images_with_zero_rois: int = 0
        self.total_rois: int = 0
        
        # Per-image detailed records
        self.per_image_records: List[Dict[str, Any]] = []
    
    def add_sample(self, 
                   e2e_ms: float,
                   stage1_ms: float, 
                   stage2_ms_list: List[float],
                   post_ms: float,
                   image_name: str = "",
                   num_rois: int = 0):
        """Add a single sample."""
        self.e2e_latencies.append(e2e_ms)
        self.stage1_latencies.append(stage1_ms)
        self.post_latencies.append(post_ms)
        
        if num_rois == 0:
            self.images_with_zero_rois += 1
        else:
            self.stage2_latencies.extend(stage2_ms_list)
            self.total_rois += num_rois
        
        self.per_image_records.append({
            'image_name': image_name,
            'e2e_ms': e2e_ms,
            'stage1_ms': stage1_ms,
            'stage2_ms_list': stage2_ms_list,
            'post_ms': post_ms,
            'num_rois': num_rois
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            'e2e': compute_stats(self.e2e_latencies),
            'stage1': compute_stats(self.stage1_latencies),
            'stage2_per_roi': compute_stats(self.stage2_latencies),
            'post': compute_stats(self.post_latencies),
            'total_images': len(self.e2e_latencies),
            'images_with_zero_rois': self.images_with_zero_rois,
            'total_rois': self.total_rois,
        }
    
    def get_per_image_records(self) -> List[Dict[str, Any]]:
        """Return per-image detailed records."""
        return self.per_image_records
