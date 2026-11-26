#!/usr/bin/env python
"""
Pre-flight validation script for benchmark suite.
Checks that all dependencies and configurations are correct before running benchmarks.

Usage:
    python tools/benchmark/validate_setup.py
"""
import sys
import os
from pathlib import Path
import subprocess


def check_item(name, passed, details=""):
    """Print check result."""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if details:
        print(f"    {details}")
    return passed


def main():
    print("\n" + "="*80)
    print("Benchmark Setup Validation")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Check Python version
    print("1. Python Environment")
    py_version = sys.version_info
    passed = py_version.major == 3 and py_version.minor >= 8
    all_passed &= check_item(
        "Python version >= 3.8",
        passed,
        f"Found: Python {py_version.major}.{py_version.minor}.{py_version.micro}"
    )
    
    # Check PyTorch
    print("\n2. PyTorch")
    try:
        import torch
        passed = True
        all_passed &= check_item(
            "PyTorch installed",
            passed,
            f"Version: {torch.__version__}"
        )
        
        cuda_available = torch.cuda.is_available()
        all_passed &= check_item(
            "CUDA available",
            cuda_available,
            f"CUDA version: {torch.version.cuda if cuda_available else 'N/A'}"
        )
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            check_item(
                "GPU device",
                True,
                f"{device_name}"
            )
    except ImportError:
        all_passed &= check_item("PyTorch installed", False, "Not found")
    
    # Check MMEngine
    print("\n3. OpenMMLab Dependencies")
    try:
        import mmengine
        check_item("mmengine", True, f"Version: {mmengine.__version__}")
    except ImportError:
        all_passed &= check_item("mmengine", False, "Not found")
    
    # Check MMDetection
    try:
        import mmdet
        check_item("mmdet", True, f"Version: {mmdet.__version__}")
    except ImportError:
        all_passed &= check_item("mmdet", False, "Not found")
    
    # Check MMSegmentation
    try:
        import mmseg
        check_item("mmseg", True, f"Version: {mmseg.__version__}")
    except ImportError:
        all_passed &= check_item("mmseg", False, "Not found")
    
    # Check other dependencies
    print("\n4. Other Dependencies")
    
    try:
        import numpy
        check_item("numpy", True, f"Version: {numpy.__version__}")
    except ImportError:
        all_passed &= check_item("numpy", False, "Not found")
    
    try:
        import cv2
        check_item("opencv (cv2)", True, f"Version: {cv2.__version__}")
    except ImportError:
        all_passed &= check_item("opencv (cv2)", False, "Not found")
    
    try:
        import psutil
        check_item("psutil", True, f"Version: {psutil.__version__}")
    except ImportError:
        all_passed &= check_item("psutil", False, "Not found")
    
    # Check project structure
    print("\n5. Project Structure")
    
    root = Path.cwd()
    
    stage1_dir = root / "configs" / "instance"
    passed = stage1_dir.exists() and stage1_dir.is_dir()
    all_passed &= check_item(
        "Stage-1 project directory",
        passed,
        f"{stage1_dir}"
    )
    
    stage2_dir = root / "configs" / "semantic"
    passed = stage2_dir.exists() and stage2_dir.is_dir()
    all_passed &= check_item(
        "Stage-2 project directory",
        passed,
        f"{stage2_dir}"
    )
    
    # Check model configs
    print("\n6. Model Configurations")
    
    stage1_config = root / "configs" / "instance" / "mask2former_leaf.py"
    passed = stage1_config.exists() and stage1_config.is_file()
    check_item(
        "Stage-1 config",
        passed,
        f"{stage1_config}"
    )
    if not passed:
        print(f"    Expected: {stage1_config}")
    
    stage2_config = root / "configs" / "semantic" / "deeplabv3plus_all.py"
    passed = stage2_config.exists() and stage2_config.is_file()
    check_item(
        "Stage-2 config",
        passed,
        f"{stage2_config}"
    )
    if not passed:
        print(f"    Expected: {stage2_config}")
    
    # Check model checkpoints
    print("\n7. Model Checkpoints")
    
    stage1_ckpt = root / "checkpoints" / "mask2former_leaf.pth"
    passed = stage1_ckpt.exists() and stage1_ckpt.is_file()
    check_item(
        "Stage-1 checkpoint",
        passed,
        f"{stage1_ckpt}" if passed else "Not found (use --stage1_checkpoint to specify)"
    )
    
    stage2_ckpt = root / "checkpoints" / "rbnet_deeplabv3plus_all.pth"
    passed = stage2_ckpt.exists() and stage2_ckpt.is_file()
    check_item(
        "Stage-2 checkpoint",
        passed,
        f"{stage2_ckpt}" if passed else "Not found (use --stage2_checkpoint to specify)"
    )
    
    # Check benchmark scripts
    print("\n8. Benchmark Scripts")
    
    bench_dir = root / "tools" / "benchmark"
    passed = bench_dir.exists() and bench_dir.is_dir()
    all_passed &= check_item(
        "Benchmark directory",
        passed,
        f"{bench_dir}"
    )
    
    if bench_dir.exists():
        scripts = [
            "run_benchmark.py",
            "run_benchmark_v3.py",
            "profiler.py",
            "tegrastats.py",
            "report.py"
        ]
        for script in scripts:
            script_path = bench_dir / script
            passed = script_path.exists() and script_path.is_file()
            all_passed &= check_item(
                f"  {script}",
                passed,
                f"{script_path}" if passed else "Not found"
            )
    
    # Check tegrastats availability
    print("\n9. Tegrastats (Optional)")
    
    try:
        result = subprocess.run(
            ['which', 'tegrastats'],
            capture_output=True,
            timeout=2
        )
        tegrastats_found = result.returncode == 0
        check_item(
            "tegrastats available",
            tegrastats_found,
            "Found" if tegrastats_found else "Not found (power/temp monitoring will be disabled)"
        )
        
        if tegrastats_found:
            # Test if we can run it without sudo
            try:
                result = subprocess.run(
                    ['tegrastats', '--help'],
                    capture_output=True,
                    timeout=2
                )
                no_sudo_needed = result.returncode != 1  # Permission denied returns 1
                check_item(
                    "  Can run without sudo",
                    no_sudo_needed,
                    "Yes" if no_sudo_needed else "No (configure sudoless access or run with sudo)"
                )
            except Exception:
                check_item("  Can run without sudo", False, "Could not test")
    
    except Exception as e:
        check_item(
            "tegrastats available",
            False,
            f"Not available: {e}"
        )
    
    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✓ All critical checks passed! You're ready to run benchmarks.")
        print("\nNext steps:")
        print("  1. Prepare test images in a directory")
        print("  2. Run: ./tools/benchmark/quick_bench.sh single-leaf /path/to/images")
        print("  3. Or: python tools/benchmark/run_benchmark.py --images /path/to/images")
    else:
        print("✗ Some checks failed. Please fix the issues above before running benchmarks.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install <package>")
        print("  - Check that you're in the correct directory (RB_REC_TOOL)")
        print("  - Verify model checkpoint paths")
        return 1
    
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
