import os
import cv2
import numpy as np
import random

# 路径设置
img_dir = '/data/code2025/Q1/2025-04-10-01/datasets/img/'       # 原图路径
mask_dir = '/data/code2025/Q1/2025-04-10-01/datasets/gt/'       # mask路径
out_img_dir = '/data/code2025/Q1/2025-04-10-01/datasets/cropped_img/'  # 裁剪后原图保存路径
out_mask_dir = '/data/code2025/Q1/2025-04-10-01/datasets/cropped_gt/'  # 裁剪后mask保存路径

# 参数
n = 20  # 每张图生成的增强图对数量
expand_range = (0.02, 0.2)  # 外扩比例范围
flip_prob = 0.5  # 水平翻转概率

# 创建输出目录
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)

# 文件名列表
img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

for img_file, mask_file in zip(img_files, mask_files):
    img_path = os.path.join(img_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        print(f"读取失败：{img_file} 或 {mask_file}")
        continue

    coords = cv2.findNonZero(mask)
    if coords is None:
        print(f"未检测到目标区域：{mask_file}")
        continue

    x, y, w, h = cv2.boundingRect(coords)
    base_crop_width = max(w, h)
    crop_height = img.shape[0]

    for i in range(n):
        # 随机外扩比例
        expand_ratio = random.uniform(*expand_range)
        expand_len = int(base_crop_width * expand_ratio)
        crop_width = base_crop_width + 2 * expand_len

        # 计算中心点位置和裁剪区域
        center_x = x + w // 2
        start_x = max(center_x - crop_width // 2, 0)
        end_x = min(start_x + crop_width, img.shape[1])
        start_x = max(end_x - crop_width, 0)

        cropped_img = img[:, start_x:end_x]
        cropped_mask = mask[:, start_x:end_x]

        # 是否水平翻转
        if random.random() < flip_prob:
            cropped_img = cv2.flip(cropped_img, 1)
            cropped_mask = cv2.flip(cropped_mask, 1)

        # 新文件名（包含序号）
        base_name = os.path.splitext(img_file)[0]
        out_img_path = os.path.join(out_img_dir, f"{base_name}_aug{i}.png")
        out_mask_path = os.path.join(out_mask_dir, f"{base_name}_aug{i}.png")

        cv2.imwrite(out_img_path, cropped_img)
        cv2.imwrite(out_mask_path, cropped_mask)

    print(f"已增强并保存：{img_file}")

print("全部增强裁剪完成！")