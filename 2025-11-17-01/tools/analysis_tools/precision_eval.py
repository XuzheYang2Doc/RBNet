# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
import os
import os.path as osp
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from mmengine import MMLogger
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

# Ensure we use the mmdet in this repo instead of a site-packages install.
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from mmdet.utils.benchmark import InferenceBenchmark  # noqa: E402


def _is_bf16_not_supported_error(err: BaseException) -> bool:
    msg = str(err)
    # Common failures:
    # - RuntimeError: "xxx" not implemented for 'BFloat16'
    # - RuntimeError: ... BFloat16 ... not implemented ...
    return ('BFloat16' in msg) and ('not implemented' in msg)


def _warn(logger: Optional[MMLogger], msg: str) -> None:
    if logger is not None:
        logger.warning(msg)
    else:
        print(f'WARNING: {msg}', flush=True)


def _autocast_ctx(precision: str, device: str):
    if device != 'cuda':
        return torch.autocast(device_type='cpu', enabled=False)

    precision = precision.lower()
    if precision == 'fp16':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    if precision == 'bf16':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return torch.autocast(device_type='cuda', enabled=False)


def _patch_test_step_autocast(model, precision: str, device: str):
    """Wrap model.test_step with autocast.

    This is intentionally lightweight and reversible.
    """
    if not hasattr(model, 'test_step'):
        raise AttributeError('model has no test_step; cannot benchmark')

    orig = model.test_step

    def wrapped(data):
        with _autocast_ctx(precision, device):
            return orig(data)

    model.test_step = wrapped
    return orig


def run_accuracy(cfg: Config,
                 checkpoint: str,
                 precision: str,
                 device: str,
                 logger: Optional[MMLogger] = None) -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    cfg.load_from = checkpoint

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    runner = Runner.from_cfg(cfg)

    # runner.model may already be moved to device by mmengine.
    # We only add autocast wrapper (fp16/bf16) at test_step level.
    orig_test_step = None
    try:
        orig_test_step = _patch_test_step_autocast(runner.model, precision, device)
        start = time.time()
        metrics = runner.test()
        elapsed = time.time() - start
    finally:
        if orig_test_step is not None:
            runner.model.test_step = orig_test_step

    if metrics is None:
        metrics = {}

    return {
        'requested_precision': precision,
        'effective_precision': precision,
        'precision': precision,
        'mode': 'test',
        'metrics': metrics,
        'elapsed_sec': elapsed,
    }


def run_speed(cfg: Config,
              checkpoint: str,
              precision: str,
              device: str,
              max_iter: int,
              log_interval: int,
              num_warmup: int,
              repeat_num: int,
              fuse_conv_bn: bool,
              logger: Optional[MMLogger] = None) -> Dict[str, Any]:
    if device != 'cuda':
        raise ValueError('Speed benchmark currently supports only --device cuda')

    init_default_scope(cfg.get('default_scope', 'mmdet'))

    bench = InferenceBenchmark(
        cfg,
        checkpoint,
        distributed=False,
        is_fuse_conv_bn=fuse_conv_bn,
        max_iter=max_iter,
        log_interval=log_interval,
        num_warmup=num_warmup,
        logger=logger)

    orig_test_step = None
    try:
        orig_test_step = _patch_test_step_autocast(bench.model, precision, device)
        results = bench.run(repeat_num)
    finally:
        if orig_test_step is not None:
            bench.model.test_step = orig_test_step

    return {
        'requested_precision': precision,
        'effective_precision': precision,
        'precision': precision,
        'mode': 'benchmark',
        'results': results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate accuracy + speed under different precisions (fp32/fp16/bf16).')

    parser.add_argument('--config', default='work_dirs/mask2former/mask2former.py')
    parser.add_argument('--checkpoint', default='work_dirs/mask2former/iter_5000.pth')

    parser.add_argument(
        '--precisions',
        default='fp32,fp16,bf16',
        help='Comma-separated list: fp32,fp16,bf16')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    parser.add_argument('--do-accuracy', action='store_true', help='Run runner.test()')
    parser.add_argument('--do-speed', action='store_true', help='Run inference benchmark')

    parser.add_argument('--repeat-num', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=10)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--num-warmup', type=int, default=5)
    parser.add_argument('--fuse-conv-bn', action='store_true')

    parser.add_argument('--work-dir', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='Write summary JSON to this path (default: work_dir/precision_eval.json)')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config (same as tools/test.py).')

    args = parser.parse_args()

    if not args.do_accuracy and not args.do_speed:
        args.do_accuracy = True
        args.do_speed = True

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir or cfg.get('work_dir') or osp.join(
        './work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)

    out_path = args.out or osp.join(work_dir, 'precision_eval.json')

    logger = MMLogger.get_instance('mmdet_precision_eval', log_level='INFO')

    precisions = [p.strip().lower() for p in args.precisions.split(',') if p.strip()]

    summary: Dict[str, Any] = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': args.device,
        'precisions': precisions,
        'runs': [],
    }

    for precision in precisions:
        if precision not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError(f'Unsupported precision: {precision}')

        def _run_with_bf16_fallback(fn, *fn_args, **fn_kwargs) -> Dict[str, Any]:
            try:
                return fn(*fn_args, **fn_kwargs)
            except RuntimeError as e:
                if precision != 'bf16' or not _is_bf16_not_supported_error(e):
                    raise
                if args.device != 'cuda':
                    raise

                _warn(
                    logger,
                    'BF16 autocast failed due to an operator not supporting BFloat16 '
                    '(common for ms_deform_attn). Falling back to FP16 for this run.')

                # First fallback: fp16
                try:
                    out = fn(*fn_args, **{**fn_kwargs, 'precision': 'fp16'})
                    out['requested_precision'] = 'bf16'
                    out['effective_precision'] = 'fp16'
                    out['precision'] = 'fp16'
                    out['fallback_reason'] = str(e)
                    return out
                except RuntimeError as e2:
                    _warn(logger, 'FP16 fallback also failed; falling back to FP32.')
                    out = fn(*fn_args, **{**fn_kwargs, 'precision': 'fp32'})
                    out['requested_precision'] = 'bf16'
                    out['effective_precision'] = 'fp32'
                    out['precision'] = 'fp32'
                    out['fallback_reason'] = str(e2)
                    return out

        # Accuracy
        if args.do_accuracy:
            acc = _run_with_bf16_fallback(
                run_accuracy,
                cfg,
                args.checkpoint,
                precision=precision,
                device=args.device,
                logger=logger)
            summary['runs'].append(acc)

        # Speed
        if args.do_speed:
            spd = _run_with_bf16_fallback(
                run_speed,
                cfg,
                args.checkpoint,
                precision=precision,
                device=args.device,
                max_iter=args.max_iter,
                log_interval=args.log_interval,
                num_warmup=args.num_warmup,
                repeat_num=args.repeat_num,
                fuse_conv_bn=args.fuse_conv_bn,
                logger=logger)
            summary['runs'].append(spd)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'Wrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
