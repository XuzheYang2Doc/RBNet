#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2025-12-07
"""
数据集转换脚本
将 extra_image 文件夹中的单张图标注文件转换为 COCO 格式的测试集标注文件
"""

import json
import os
from pathlib import Path
from PIL import Image


def convert_labelme_to_coco(extra_image_dir, output_dir):
    """
    将 LabelMe 格式的标注转换为 COCO 格式
    
    Args:
        extra_image_dir: 包含图片和标注文件的目录
        output_dir: 输出目录
    """
    
    # 初始化 COCO 格式数据结构（包含所有类别）
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Leaf"},
            {"id": 2, "name": "Lesion"}
        ]
    }
    
    # 初始化 COCO 格式数据结构（仅包含 leaf 类别）
    coco_data_no_lesion = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Leaf"}
        ]
    }
    
    # 获取所有 JSON 标注文件
    json_files = sorted(Path(extra_image_dir).glob("*.json"))
    
    image_id = 1
    annotation_id = 1
    annotation_id_no_lesion = 1
    
    print(f"找到 {len(json_files)} 个标注文件")
    
    for json_file in json_files:
        # 读取 JSON 标注
        with open(json_file, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        
        # 获取对应的图片文件
        image_file = json_file.with_suffix('.jpg')
        if not image_file.exists():
            print(f"警告: 图片文件不存在 {image_file}")
            continue
        
        # 获取图片尺寸
        try:
            with Image.open(image_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"警告: 无法读取图片 {image_file}: {e}")
            continue
        
        # 添加图片信息
        image_info = {
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        }
        
        coco_data["images"].append(image_info)
        coco_data_no_lesion["images"].append(image_info)
        
        # 处理标注
        for shape in labelme_data.get("shapes", []):
            label = shape["label"]
            points = shape["points"]
            
            # 将 points 展平为 COCO segmentation 格式
            segmentation = []
            for point in points:
                segmentation.extend(point)
            
            # 计算 bbox (x, y, width, height)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            # 计算面积
            area = (x_max - x_min) * (y_max - y_min)
            
            # 确定类别ID
            if label.lower() == "leaf":
                category_id = 1
            elif label.lower() == "lesion":
                category_id = 2
            else:
                print(f"警告: 未知类别 {label} 在文件 {json_file}")
                continue
            
            # 创建标注信息（完整版本）
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
            
            # 如果是 leaf 类别，也添加到 no_lesion 版本
            if category_id == 1:
                annotation_no_lesion = {
                    "id": annotation_id_no_lesion,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_data_no_lesion["annotations"].append(annotation_no_lesion)
                annotation_id_no_lesion += 1
        
        image_id += 1
        
        if image_id % 10 == 0:
            print(f"已处理 {image_id - 1} 张图片...")
    
    # 保存输出文件
    output_file_all = Path(output_dir) / "instances_test2017.json"
    output_file_no_lesion = Path(output_dir) / "instances_test2017_no_lesion.json"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存完整版本
    with open(output_file_all, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    print(f"\n已保存完整标注文件: {output_file_all}")
    print(f"  - 图片数量: {len(coco_data['images'])}")
    print(f"  - 标注数量: {len(coco_data['annotations'])}")
    print(f"  - 类别: Leaf, Lesion")
    
    # 保存 no_lesion 版本
    with open(output_file_no_lesion, 'w', encoding='utf-8') as f:
        json.dump(coco_data_no_lesion, f, indent=2)
    print(f"\n已保存 no_lesion 标注文件: {output_file_no_lesion}")
    print(f"  - 图片数量: {len(coco_data_no_lesion['images'])}")
    print(f"  - 标注数量: {len(coco_data_no_lesion['annotations'])}")
    print(f"  - 类别: Leaf only")
    
    print("\n转换完成！")


if __name__ == "__main__":
    # 设置路径
    extra_image_dir = "data/coco/extra_image"
    output_dir = "data/coco/annotations"
    
    # 执行转换
    convert_labelme_to_coco(extra_image_dir, output_dir)
