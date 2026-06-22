#!/usr/bin/env python3
"""
Pre-flight check script for Grad-CAM tool.
Verifies that all dependencies, configs, and checkpoints are accessible.
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_python_packages():
    """Check if required Python packages are installed."""
    print(f"\n{BLUE}=== Checking Python Packages ==={RESET}")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'mmengine': 'MMEngine',
        'mmseg': 'MMSegmentation',
    }
    
    all_ok = True
    for module_name, display_name in packages.items():
        try:
            if module_name == 'cv2':
                import cv2
            elif module_name == 'torch':
                import torch
            elif module_name == 'numpy':
                import numpy
            elif module_name == 'matplotlib':
                import matplotlib
            elif module_name == 'mmengine':
                import mmengine
            elif module_name == 'mmseg':
                import mmseg
            print(f"  {GREEN}✓{RESET} {display_name}")
        except ImportError:
            print(f"  {RED}✗{RESET} {display_name} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_custom_modules():
    """Check if custom decode head is importable."""
    print(f"\n{BLUE}=== Checking Custom Modules ==={RESET}")
    
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import mmseg.models.decode_heads.sep_aspp_head_saetr
        print(f"  {GREEN}✓{RESET} Custom decode head (sep_aspp_head_saetr.py)")
        return True
    except ImportError as e:
        print(f"  {RED}✗{RESET} Custom decode head - IMPORT ERROR")
        print(f"     Error: {e}")
        return False


def check_configs():
    """Check if model configs exist."""
    print(f"\n{BLUE}=== Checking Model Configs ==={RESET}")
    
    project_root = Path(__file__).resolve().parents[2]
    config_dir = project_root / 'configs' / 'my_model_configs'
    
    configs = [
        'deeplabv3plus.py',
        'deeplabv3plus_tr.py',
        'deeplabv3plus_sae.py',
        'deeplabv3plus_all.py',
    ]
    
    all_ok = True
    for config_file in configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  {GREEN}✓{RESET} {config_file}")
        else:
            print(f"  {RED}✗{RESET} {config_file} - NOT FOUND")
            print(f"     Expected: {config_path}")
            all_ok = False
    
    return all_ok


def check_checkpoints():
    """Check if model checkpoints exist."""
    print(f"\n{BLUE}=== Checking Model Checkpoints ==={RESET}")
    
    project_root = Path(__file__).resolve().parents[2]
    work_dir = project_root / 'work_dirs'
    
    checkpoints = {
        'deeplabv3plus': 'Baseline',
        'deeplabv3plus_tr': 'BiFormer (TR)',
        'deeplabv3plus_sae': 'ACE (SAE)',
        'deeplabv3plus_all': 'TR+SAE',
    }
    
    found_count = 0
    for ckpt_dir, display_name in checkpoints.items():
        ckpt_path = work_dir / ckpt_dir
        if ckpt_path.exists():
            # Check for any .pth files
            pth_files = list(ckpt_path.glob('*.pth'))
            if pth_files:
                print(f"  {GREEN}✓{RESET} {display_name}: {len(pth_files)} checkpoint(s) found")
                found_count += 1
            else:
                print(f"  {YELLOW}⚠{RESET} {display_name}: Directory exists but no .pth files")
        else:
            print(f"  {YELLOW}⚠{RESET} {display_name}: Directory not found")
            print(f"     Expected: {ckpt_path}")
    
    if found_count == 0:
        print(f"\n  {RED}Note:{RESET} No checkpoints found. You need to train models first.")
        print(f"        Or update checkpoint paths in example scripts.")
        return False
    elif found_count < 4:
        print(f"\n  {YELLOW}Note:{RESET} Only {found_count}/4 model checkpoints found.")
        print(f"        You can still run Grad-CAM on available models.")
        return True
    else:
        return True


def check_grad_cam_tool():
    """Check if Grad-CAM tool files exist."""
    print(f"\n{BLUE}=== Checking Grad-CAM Tool Files ==={RESET}")
    
    tool_dir = Path(__file__).resolve().parent
    
    files = {
        'run_grad_cam.py': 'Main Grad-CAM script',
        'README.md': 'Documentation',
        'QUICKSTART.md': 'Quick reference',
        'example_run.sh': 'Example script',
    }
    
    all_ok = True
    for filename, description in files.items():
        file_path = tool_dir / filename
        if file_path.exists():
            print(f"  {GREEN}✓{RESET} {filename} ({description})")
        else:
            print(f"  {RED}✗{RESET} {filename} - NOT FOUND")
            all_ok = False
    
    return all_ok


def print_summary(checks):
    """Print summary of all checks."""
    print(f"\n{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}=== Pre-flight Check Summary ==={RESET}")
    print(f"{BLUE}{'='*50}{RESET}")
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"  {check_name}: {status}")
    
    print(f"{BLUE}{'='*50}{RESET}\n")
    
    if all_passed:
        print(f"{GREEN}✓ All checks passed! Ready to run Grad-CAM.{RESET}\n")
        print(f"Next steps:")
        print(f"  1. Check example_run.sh for usage examples")
        print(f"  2. Update checkpoint paths if needed")
        print(f"  3. Run: python tools/grad_cam/run_grad_cam.py --help")
    else:
        print(f"{YELLOW}⚠ Some checks failed. Please address the issues above.{RESET}\n")
        print(f"Common fixes:")
        print(f"  - Install missing packages: pip install opencv-python matplotlib")
        print(f"  - Train models or update checkpoint paths")
        print(f"  - Check config file paths")
    
    return all_passed


def main():
    print(f"{BLUE}{'='*50}{RESET}")
    print(f"{BLUE}Grad-CAM Tool Pre-flight Check{RESET}")
    print(f"{BLUE}{'='*50}{RESET}")
    
    checks = {
        'Python Packages': check_python_packages(),
        'Custom Modules': check_custom_modules(),
        'Model Configs': check_configs(),
        'Model Checkpoints': check_checkpoints(),
        'Grad-CAM Tool Files': check_grad_cam_tool(),
    }
    
    all_passed = print_summary(checks)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
