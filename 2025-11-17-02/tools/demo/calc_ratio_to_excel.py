import os

import mmcv
import numpy as np
import pandas as pd

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.utils import get_palette


def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def resolve_palette(palette_arg):
    """Support mmseg palette name, npy palette file, or 'my' custom palette."""
    if isinstance(palette_arg, str):
        if palette_arg == 'my':
            # background = black, target = white
            return [[0, 0, 0], [255, 255, 255]]
        if palette_arg.endswith('.npy') and os.path.isfile(palette_arg):
            return mmcv.load(palette_arg).tolist()
        return get_palette(palette_arg)
    return palette_arg


def save_segmentation(seg_model, image_path, result, palette, opacity, out_path):
    show_result_pyplot(
        seg_model,
        image_path,
        result,
        palette=palette,
        opacity=opacity,
        out_file=out_path
    )


def compute_leaf_mask(image: np.ndarray) -> np.ndarray:
    """Leaf area mask: any non-zero RGB pixel is considered leaf."""
    return np.any(image != 0, axis=-1)


def run_inference_and_export(
    config: str,
    work_dirs: str,
    checkpoint: str,
    device: str,
    input_dir: str,
    seg_output_dir: str,
    excel_output_path: str,
    target_class: int = 1,
    opacity: float = 1.0,
    palette: str = "my",
):
    # 1) init model
    config_path = os.path.join("../my_config", config + ".py")
    checkpoint_path = os.path.join(work_dirs, config, checkpoint)
    seg_model = init_model(config_path, checkpoint_path, device=device)

    # 2) prepare dirs / palette
    ensure_dir(seg_output_dir)
    ensure_dir(os.path.dirname(excel_output_path))
    palette = resolve_palette(palette)

    # 3) list images
    image_files = [f for f in os.listdir(input_dir) if not f.startswith(".")]
    image_files.sort()

    # 4) loop & compute
    rows = []
    total_leaf_area = 0
    total_disease_area = 0

    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        result = inference_model(seg_model, image_path)
        seg_map = np.array(result[0], dtype=np.int32)

        # save visualization
        vis_path = os.path.join(seg_output_dir, file_name)
        save_segmentation(seg_model, image_path, result, palette, opacity, vis_path)

        # compute ratio on leaf pixels
        image = mmcv.imread(image_path)
        leaf_mask = compute_leaf_mask(image)
        disease_mask = (seg_map == target_class)
        disease_on_leaf = disease_mask & leaf_mask

        disease_area = int(np.count_nonzero(disease_on_leaf))
        leaf_area = int(np.count_nonzero(leaf_mask))
        ratio = (disease_area / leaf_area)*100 if leaf_area > 0 else 0.0

        total_leaf_area += leaf_area
        total_disease_area += disease_area

        print(f"{file_name}: disease area {disease_area} px, leaf area {leaf_area} px, ratio {ratio:.4f}")
        rows.append({"image_name": file_name, "ratio": ratio})

    # 5) total print
    if total_leaf_area > 0:
        total_ratio = (total_disease_area / total_leaf_area)*100
        print(
            f"Total disease area {total_disease_area} px, total leaf area {total_leaf_area} px, "
            f"overall ratio {total_ratio:.4f}"
        )
    else:
        print("No leaf pixels detected in the dataset.")

    # 6) export excel
    df = pd.DataFrame(rows, columns=["image_name", "ratio"])
    df.to_excel(excel_output_path, index=False)
    print(f"\nExcel saved to: {excel_output_path}")


# =========================
# 只需要在这里改路径/文件名
# =========================
if __name__ == "__main__":
    USER_CONFIG = {
        "config": "upernet",
        "work_dirs": "../work_dirs",
        "checkpoint": "iter_10000.pth",

        # 可选：如果你也想在这儿改输入/输出目录
        "input_dir": "../datasets/test/images",
        "seg_output_dir": "../datasets/test/pred_mask",

        # Excel 输出路径（含文件名）
        "excel_output_path": r"D:\scientific_research\paper_RB_rec_tool\Model_artificial_compare\model_calc_results\upernet.xlsx",

        # 其他参数一般不需要改（需要的话再改）
        "device": "cuda:0",
        "target_class": 1,
        "opacity": 1.0,
        "palette": "my",
    }

    run_inference_and_export(**USER_CONFIG)
