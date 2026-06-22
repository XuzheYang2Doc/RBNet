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
    work_dir: str,
    checkpoint: str,
    device: str,
    input_dir: str,
    seg_output_dir: str,
    excel_output_path: str,
    target_class: int = 1,
    opacity: float = 1.0,
    palette: str = "my",
):
    config_path = os.path.join("configs/semantic", config + ".py")
    checkpoint_path = os.path.join(work_dir, config, checkpoint)
    seg_model = init_model(config_path, checkpoint_path, device=device)

    ensure_dir(seg_output_dir)
    ensure_dir(os.path.dirname(excel_output_path))
    palette = resolve_palette(palette)

    image_files = [f for f in os.listdir(input_dir) if not f.startswith(".")]
    image_files.sort()

    rows = []
    total_leaf_area = 0
    total_disease_area = 0

    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        result = inference_model(seg_model, image_path)
        seg_map = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.int32)

        vis_path = os.path.join(seg_output_dir, file_name)
        save_segmentation(seg_model, image_path, result, palette, opacity, vis_path)

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

    if total_leaf_area > 0:
        total_ratio = (total_disease_area / total_leaf_area)*100
        print(
            f"Total disease area {total_disease_area} px, total leaf area {total_leaf_area} px, "
            f"overall ratio {total_ratio:.4f}"
        )
    else:
        print("No leaf pixels detected in the dataset.")

    df = pd.DataFrame(rows, columns=["image_name", "ratio"])
    df.to_excel(excel_output_path, index=False)
    print(f"\nExcel saved to: {excel_output_path}")


if __name__ == "__main__":
    USER_CONFIG = {
        "config": "deeplabv3plus_all",
        "work_dir": "work_dirs",
        "checkpoint": "iter_10000.pth",
        "input_dir": "outputs/leaf_crops/images",
        "seg_output_dir": "outputs/semantic_masks",
        "excel_output_path": "outputs/severity/semantic_results.xlsx",
        "device": "cuda:0",
        "target_class": 1,
        "opacity": 1.0,
        "palette": "my",
    }

    run_inference_and_export(**USER_CONFIG)
