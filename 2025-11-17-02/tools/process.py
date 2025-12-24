import mmcv
import os
import cv2
from PIL import Image
import numpy as np
import os.path as osp

root = '/data/code2025/Q1/2025-04-10-01/datasets/dataset/masks'
save = '/data/code2025/Q1/2025-04-10-01/datasets/dataset/labels'

os.makedirs(save, exist_ok=True)

PALETTE = [[0, 0, 0],[1, 1, 1]]
count = 0
for file in mmcv.scandir(osp.join(root), suffix='.png'):
    count += 1
    seg_map = cv2.imread(osp.join(root, file), cv2.IMREAD_GRAYSCALE)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(PALETTE, dtype=np.uint8))
    seg_img.save(osp.join(save, file))
    print(count)

