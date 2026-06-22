#!/usr/bin/env python3
"""Compatibility wrapper for the benchmark entrypoint.

This keeps older shell scripts working while delegating the actual benchmark
implementation to ``run_benchmark_v3.py``.
"""

import argparse
import sys
import warnings
from typing import List


def _build_v3_argv(args: argparse.Namespace) -> List[str]:
    """Translate legacy/compat CLI arguments to the v3 entrypoint."""
    argv = [
        'run_benchmark_v3.py',
        '--images',
        args.images,
        '--stage1-config',
        args.stage1_config,
        '--stage1-checkpoint',
        args.stage1_checkpoint,
        '--stage2-config',
        args.stage2_config,
        '--stage2-checkpoint',
        args.stage2_checkpoint,
        '--score-thr',
        str(args.score_thr),
        '--num-images',
        str(args.num_images),
        '--warmup',
        str(args.warmup),
        '--output',
        args.output,
        '--device',
        args.device,
    ]
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compatibility wrapper for the Jetson benchmark suite')
    parser.add_argument('--images', type=str, required=True,
                        help='Image directory or file list')
    parser.add_argument('--output', '--out_dir', dest='output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='single-leaf',
                        choices=['single-leaf', 'continuous'],
                        help='Legacy mode flag kept for shell compatibility')
    parser.add_argument('--stage1-config', '--stage1_config', dest='stage1_config',
                        type=str, default='2025-11-17-01/my_config/mask2former.py')
    parser.add_argument('--stage1-checkpoint', '--stage1_checkpoint',
                        dest='stage1_checkpoint', type=str,
                        default='2025-11-17-01/work_dirs/mask2former/iter_5000.pth')
    parser.add_argument('--stage2-config', '--stage2_config', dest='stage2_config',
                        type=str,
                        default='2025-11-17-02/configs/my_model_configs/deeplabv3plus_all.py')
    parser.add_argument('--stage2-checkpoint', '--stage2_checkpoint',
                        dest='stage2_checkpoint', type=str,
                        default='2025-11-17-02/work_dirs/deeplabv3plus_all/iter_10000.pth')
    parser.add_argument('--score-thr', '--score_thr', dest='score_thr',
                        type=float, default=0.3)
    parser.add_argument('--num-images', '--num_images', dest='num_images',
                        type=int, default=500)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--continuous-minutes', '--continuous_minutes',
                        dest='continuous_minutes', type=float, default=5.0)
    parser.add_argument('--power-interval-ms', '--power_interval_ms',
                        dest='power_interval_ms', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')

    args = parser.parse_args()

    ignored = []
    if args.mode == 'continuous':
        ignored.append(f'mode={args.mode}')
    if args.continuous_minutes != 5.0:
        ignored.append(f'continuous_minutes={args.continuous_minutes}')
    if args.power_interval_ms != 1000:
        ignored.append(f'power_interval_ms={args.power_interval_ms}')
    if args.seed != 0:
        ignored.append(f'seed={args.seed}')
    if args.deterministic:
        ignored.append('deterministic=True')

    if ignored:
        warnings.warn(
            'The compatibility wrapper forwards to run_benchmark_v3.py and '
            f'ignores legacy options: {", ".join(ignored)}')

    v3_argv = _build_v3_argv(args)
    from run_benchmark_v3 import main as v3_main

    old_argv = sys.argv
    try:
        sys.argv = v3_argv
        return int(v3_main() or 0)
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    raise SystemExit(main())
