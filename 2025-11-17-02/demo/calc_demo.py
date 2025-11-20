import os
from argparse import ArgumentParser

import mmcv
import numpy as np

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def parse_args():
    parser = ArgumentParser(description='Segmentation inference with area calculation')
    parser.add_argument('--config', default='deeplabv3plus_all', help='Config file name without extension')
    parser.add_argument('--work_dirs', default='../work_dirs', help='Directory that stores checkpoints')
    parser.add_argument('--checkpoint', default='iter_10000.pth', help='Checkpoint file name')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--input-dir', default='../datasets/test/images', help='Directory with input images')
    parser.add_argument(
        '--seg-output-dir',
        default='../datasets/test/pred_mask',
        help='Directory to save segmentation visualizations')
    parser.add_argument(
        '--palette',
        default='my',
        help='Color palette used for visualization (mmseg palette name or comma separated RGB values)')
    parser.add_argument('--opacity', type=float, default=1.0, help='Opacity of painted segmentation map')
    parser.add_argument('--target-class', type=int, default=1, help='Class index that represents the disease area')
    return parser.parse_args()


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def resolve_palette(palette_arg):
    if isinstance(palette_arg, str):
        if palette_arg == 'my':
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
        out_file=out_path)


def compute_leaf_mask(image: np.ndarray) -> np.ndarray:
    return np.any(image != 0, axis=-1)


def main():
    args = parse_args()
    seg_model = init_segmentor(
        os.path.join('../my_config', args.config + '.py'),
        os.path.join(args.work_dirs, args.config, args.checkpoint),
        device=args.device)

    input_dir = args.input_dir
    seg_output_dir = args.seg_output_dir
    ensure_dir(seg_output_dir)
    palette = resolve_palette(args.palette)

    image_files = [f for f in os.listdir(input_dir) if not f.startswith('.')]
    image_files.sort()

    total_leaf_area = 0
    total_disease_area = 0

    for file_name in image_files:
        image_path = os.path.join(input_dir, file_name)
        result = inference_segmentor(seg_model, image_path)

        seg_map = np.array(result[0], dtype=np.int32)

        vis_path = os.path.join(seg_output_dir, file_name)
        save_segmentation(seg_model, image_path, result, palette, args.opacity, vis_path)

        image = mmcv.imread(image_path)
        leaf_mask = compute_leaf_mask(image)
        disease_mask = seg_map == args.target_class
        disease_on_leaf = disease_mask & leaf_mask

        disease_area = int(np.count_nonzero(disease_on_leaf))
        leaf_area = int(np.count_nonzero(leaf_mask))
        ratio = disease_area / leaf_area if leaf_area > 0 else 0.0

        total_leaf_area += leaf_area
        total_disease_area += disease_area

        print(f'{file_name}: disease area {disease_area} px, leaf area {leaf_area} px, ratio {ratio:.4f}')

    if total_leaf_area > 0:
        total_ratio = total_disease_area / total_leaf_area
        print(
            f'Total disease area {total_disease_area} px, total leaf area {total_leaf_area} px, '
            f'overall ratio {total_ratio:.4f}')
    else:
        print('No leaf pixels detected in the dataset.')


if __name__ == '__main__':
    main()

