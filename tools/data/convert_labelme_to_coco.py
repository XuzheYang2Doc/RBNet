#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LabelMe rice blast annotations to COCO instance format."""

import json
import os
from pathlib import Path
from PIL import Image


def convert_labelme_to_coco(extra_image_dir, output_dir):
    """Convert LabelMe polygon annotations to COCO JSON files."""

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Leaf"},
            {"id": 2, "name": "Lesion"}
        ]
    }
    
    coco_data_no_lesion = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Leaf"}
        ]
    }
    
    json_files = sorted(Path(extra_image_dir).glob("*.json"))
    
    image_id = 1
    annotation_id = 1
    annotation_id_no_lesion = 1
    
    print(f"Found {len(json_files)} annotation files")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)

        image_file = json_file.with_suffix('.jpg')
        if not image_file.exists():
            print(f"Warning: image file does not exist: {image_file}")
            continue

        try:
            with Image.open(image_file) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: failed to read image {image_file}: {e}")
            continue

        image_info = {
            "id": image_id,
            "file_name": image_file.name,
            "width": width,
            "height": height
        }
        
        coco_data["images"].append(image_info)
        coco_data_no_lesion["images"].append(image_info)
        
        for shape in labelme_data.get("shapes", []):
            label = shape["label"]
            points = shape["points"]
            
            segmentation = []
            for point in points:
                segmentation.extend(point)

            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            area = (x_max - x_min) * (y_max - y_min)

            if label.lower() == "leaf":
                category_id = 1
            elif label.lower() == "lesion":
                category_id = 2
            else:
                print(f"Warning: unknown class {label} in {json_file}")
                continue

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
            print(f"Processed {image_id - 1} images...")

    output_file_all = Path(output_dir) / "instances_test2017.json"
    output_file_no_lesion = Path(output_dir) / "instances_test2017_no_lesion.json"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_file_all, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    print(f"\nSaved full annotation file: {output_file_all}")
    print(f"  - images: {len(coco_data['images'])}")
    print(f"  - annotations: {len(coco_data['annotations'])}")
    print("  - classes: Leaf, Lesion")

    with open(output_file_no_lesion, 'w', encoding='utf-8') as f:
        json.dump(coco_data_no_lesion, f, indent=2)
    print(f"\nSaved leaf-only annotation file: {output_file_no_lesion}")
    print(f"  - images: {len(coco_data_no_lesion['images'])}")
    print(f"  - annotations: {len(coco_data_no_lesion['annotations'])}")
    print("  - classes: Leaf only")

    print("\nConversion complete")


if __name__ == "__main__":
    extra_image_dir = "data/instance/labelme"
    output_dir = "data/instance/annotations"

    convert_labelme_to_coco(extra_image_dir, output_dir)
