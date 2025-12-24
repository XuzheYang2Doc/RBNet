import os
import json
import numpy as np
import cv2
from tqdm import tqdm


def create_mask_from_json(json_path, image_shape=(512, 512), line_thickness=20):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img = json_path.replace('/json', '/img').replace('.json', '.jpg')
    img = cv2.imread(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        if len(points) < 2 or label != 'depth':
            continue

        points = np.array(points, dtype=np.int32)
        for i in range(len(points) - 1):
            pt1 = tuple(points[i])
            pt2 = tuple(points[i + 1])
            cv2.line(mask, pt1, pt2, 255, thickness=line_thickness)

    return mask


def process_json_folder(input_dir, output_dir, image_shape=(512, 512), line_thickness=20):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Converting JSON to masks"):
        json_path = os.path.join(input_dir, json_file)
        mask = create_mask_from_json(json_path, image_shape=image_shape, line_thickness=line_thickness)
        mask_filename = os.path.splitext(json_file)[0] + '.png'
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, mask)


if __name__ == "__main__":
    images_json_folder = "/data/code2025/Q1/2025-04-10-01/datasets/img"  
    input_json_folder = "/data/code2025/Q1/2025-04-10-01/datasets/json"  
    output_mask_folder = "/data/code2025/Q1/2025-04-10-01/datasets/gt"
    os.makedirs(output_mask_folder, exist_ok=True)
    process_json_folder(input_json_folder, output_mask_folder)
