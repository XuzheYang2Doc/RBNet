#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像批量resize脚本

用法示例:
    python tools/resize_images.py --input-dir datasets/test/images --output-dir datasets/test/images_resized --size 512 512
    python tools/resize_images.py --input-dir datasets/test/images --output-dir datasets/test/images_resized --size 1024 1024 --keep-ratio
"""

import argparse
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='批量resize图像')
    parser.add_argument(
        '--input-dir',
        '-i',
        type=str,
        required=True,
        help='输入图像文件夹路径')
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        required=True,
        help='输出图像文件夹路径')
    parser.add_argument(
        '--size',
        '-s',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('WIDTH', 'HEIGHT'),
        help='目标尺寸，格式: 宽 高 (默认: 512 512)')
    parser.add_argument(
        '--keep-ratio',
        '-k',
        action='store_true',
        help='是否保持宽高比（使用padding填充）')
    parser.add_argument(
        '--padding-color',
        '-p',
        type=int,
        nargs=3,
        default=[0, 0, 0],
        metavar=('R', 'G', 'B'),
        help='保持宽高比时的填充颜色 (默认: 0 0 0 黑色)')
    parser.add_argument(
        '--interpolation',
        type=str,
        default='bilinear',
        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
        help='插值方法 (默认: bilinear)')
    parser.add_argument(
        '--ext',
        '-e',
        type=str,
        nargs='+',
        default=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        help='要处理的图像扩展名 (默认: jpg jpeg png bmp tif tiff)')
    parser.add_argument(
        '--output-format',
        '-f',
        type=str,
        default=None,
        help='输出图像格式，如 png, jpg (默认: 保持原格式)')
    parser.add_argument(
        '--quality',
        '-q',
        type=int,
        default=95,
        help='JPEG保存质量 (默认: 95)')
    args = parser.parse_args()
    return args


def get_interpolation(method):
    """获取PIL插值方法"""
    interpolation_dict = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    return interpolation_dict.get(method, Image.BILINEAR)


def resize_image(img, target_size, keep_ratio=False, padding_color=(0, 0, 0), interpolation=Image.BILINEAR):
    """
    Resize图像
    
    Args:
        img: PIL Image对象
        target_size: 目标尺寸 (width, height)
        keep_ratio: 是否保持宽高比
        padding_color: 填充颜色 (R, G, B)
        interpolation: 插值方法
    
    Returns:
        resized PIL Image对象
    """
    target_w, target_h = target_size
    
    if not keep_ratio:
        # 直接resize
        return img.resize((target_w, target_h), interpolation)
    
    # 保持宽高比resize
    orig_w, orig_h = img.size
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    
    # 先resize
    resized_img = img.resize((new_w, new_h), interpolation)
    
    # 创建目标尺寸的画布并粘贴
    if img.mode == 'RGBA':
        new_img = Image.new('RGBA', (target_w, target_h), (*padding_color, 255))
    elif img.mode == 'L':
        # 灰度图
        new_img = Image.new('L', (target_w, target_h), padding_color[0])
    else:
        new_img = Image.new('RGB', (target_w, target_h), tuple(padding_color))
    
    # 居中粘贴
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized_img, (paste_x, paste_y))
    
    return new_img


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in args.ext:
        image_files.extend(glob(osp.join(args.input_dir, f'*.{ext}')))
        image_files.extend(glob(osp.join(args.input_dir, f'*.{ext.upper()}')))
    
    # 去重
    image_files = list(set(image_files))
    image_files.sort()
    
    if len(image_files) == 0:
        print(f'错误: 在 {args.input_dir} 中没有找到图像文件')
        return
    
    print(f'找到 {len(image_files)} 个图像文件')
    print(f'目标尺寸: {args.size[0]} x {args.size[1]}')
    print(f'保持宽高比: {args.keep_ratio}')
    print(f'插值方法: {args.interpolation}')
    print(f'输出目录: {args.output_dir}')
    print('-' * 50)
    
    interpolation = get_interpolation(args.interpolation)
    target_size = tuple(args.size)
    padding_color = tuple(args.padding_color)
    
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_files, desc='Resizing'):
        try:
            # 读取图像
            img = Image.open(img_path)
            
            # 如果是调色板模式，转换为RGB
            if img.mode == 'P':
                img = img.convert('RGB')
            
            # Resize
            resized_img = resize_image(
                img, 
                target_size, 
                keep_ratio=args.keep_ratio,
                padding_color=padding_color,
                interpolation=interpolation
            )
            
            # 确定输出文件名
            basename = osp.basename(img_path)
            if args.output_format:
                name, _ = osp.splitext(basename)
                basename = f'{name}.{args.output_format}'
            
            output_path = osp.join(args.output_dir, basename)
            
            # 保存图像
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                # JPEG不支持RGBA
                if resized_img.mode == 'RGBA':
                    resized_img = resized_img.convert('RGB')
                resized_img.save(output_path, quality=args.quality)
            else:
                resized_img.save(output_path)
            
            success_count += 1
            
        except Exception as e:
            print(f'\n处理 {img_path} 时出错: {e}')
            fail_count += 1
    
    print('-' * 50)
    print(f'处理完成! 成功: {success_count}, 失败: {fail_count}')


if __name__ == '__main__':
    main()
