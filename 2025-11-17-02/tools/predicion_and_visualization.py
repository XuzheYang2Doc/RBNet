import os
from mmseg.apis import init_segmentor, inference_segmentor
import mmcv
import numpy as np

# 配置文件和模型文件路径
config_file = r'E:\BaiduNetdiskDownload\2025-11-17-02\my_config\upernet.py'
checkpoint_file = r'E:\BaiduNetdiskDownload\2025-11-17-02\work_dirs\upernet\iter_10000.pth'
device = 'cuda:0'

# 初始化模型
model = init_segmentor(config_file, checkpoint_file, device=device)

input_folder = r'D:\scientific_research\paper_RB_rec_tool\compare_model_prediction_visualization\input'
output_folder = r'D:\scientific_research\paper_RB_rec_tool\compare_model_prediction_visualization\upernet'

os.makedirs(output_folder, exist_ok=True)

mask_color = np.array([0, 0, 255], dtype=np.uint8)  # 红色
alpha = 0.3

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing {img_name}...")

        img = mmcv.imread(img_path)
        H, W = img.shape[:2]

        # 推理
        result = inference_segmentor(model, img_path)
        pred_mask = result[0].astype(np.uint8)

        # 注意这里：size = (W, H)，并且用最近邻插值
        pred_mask_resized = mmcv.imresize(
            pred_mask,
            (W, H),
            interpolation='nearest'
        )

        # 保存灰度 mask
        mask_output_path = os.path.join(
            output_folder, f"{os.path.splitext(img_name)[0]}_mask.png")
        mmcv.imwrite(pred_mask_resized, mask_output_path)

        # 生成红色叠加图
        vis_img = img.copy()
        mask_region = pred_mask_resized > 0  # True/False，形状 (H, W)

        color_layer = np.zeros_like(vis_img)  # (H, W, 3)
        color_layer[mask_region] = mask_color

        vis_img = vis_img * (1 - alpha) + color_layer * alpha
        vis_img = vis_img.astype(np.uint8)

        vis_output_path = os.path.join(
            output_folder, f"{os.path.splitext(img_name)[0]}_vis.png")
        mmcv.imwrite(vis_img, vis_output_path)

        print(f"Saved mask and visualized result for {img_name}.")

print("Batch prediction complete.")
