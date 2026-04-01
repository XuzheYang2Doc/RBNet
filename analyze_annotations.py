import os
import json
import glob
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import matplotlib.pyplot as plt


def collect_stats(json_pattern: str = "*.json"):
    json_files = glob.glob(json_pattern)
    json_files = [f for f in json_files if os.path.isfile(f)]

    class_counts = Counter()
    bbox_widths = []
    bbox_heights = []
    center_x_norm = []
    center_y_norm = []
    rel_w = []
    rel_h = []
    image_label_sets = []
    all_labels = set()

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        shapes = data.get("shapes", [])
        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")

        labels_in_image = set()

        for s in shapes:
            label = s.get("label", "unknown")
            points = np.array(s.get("points", []), dtype=float)
            if points.size == 0:
                continue

            xs = points[:, 0]
            ys = points[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            bw = x_max - x_min
            bh = y_max - y_min
            if bw <= 0 or bh <= 0:
                continue

            # 掩码/目标实例数量（按类别）
            class_counts[label] += 1
            labels_in_image.add(label)

            # 框尺寸
            bbox_widths.append(bw)
            bbox_heights.append(bh)

            if img_w and img_h:
                # 中心点相对位置（归一化到 [0,1]）
                cx = (x_min + x_max) / 2.0 / img_w
                cy = (y_min + y_max) / 2.0 / img_h
                center_x_norm.append(cx)
                center_y_norm.append(cy)

                # 目标相对于整幅图的宽高比例（归一化宽、高）
                rel_w.append(bw / img_w)
                rel_h.append(bh / img_h)

        if labels_in_image:
            image_label_sets.append(labels_in_image)
            all_labels.update(labels_in_image)

    return {
        "class_counts": class_counts,
        "bbox_widths": np.array(bbox_widths, dtype=float),
        "bbox_heights": np.array(bbox_heights, dtype=float),
        "center_x": np.array(center_x_norm, dtype=float),
        "center_y": np.array(center_y_norm, dtype=float),
        "rel_w": np.array(rel_w, dtype=float),
        "rel_h": np.array(rel_h, dtype=float),
        "image_label_sets": image_label_sets,
        "all_labels": sorted(all_labels),
        "num_json_files": len(json_files),
    }


def plot_overview(stats, output_path: str = "annotation_overview.png"):
    class_counts = stats["class_counts"]
    bbox_widths = stats["bbox_widths"]
    bbox_heights = stats["bbox_heights"]
    center_x = stats["center_x"]
    center_y = stats["center_y"]
    rel_w = stats["rel_w"]
    rel_h = stats["rel_h"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 宫格1：训练集的数据量，每个类别包含的样本（实例）数量
    labels = list(class_counts.keys())
    counts = [class_counts[l] for l in labels]
    axes[0].bar(range(len(labels)), counts)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("实例数量")
    axes[0].set_title("宫格1：每个类别的掩码数量（训练集）")

    # 宫格2：框的尺寸和数量（宽-高二维直方图）
    if len(bbox_widths) > 0:
        h2 = axes[1].hist2d(
            bbox_widths,
            bbox_heights,
            bins=50,
        )
        fig.colorbar(h2[3], ax=axes[1])
        axes[1].set_xlabel("框宽度 (像素)")
        axes[1].set_ylabel("框高度 (像素)")
        axes[1].set_title("宫格2：边界框尺寸分布（宽×高，颜色表示数量）")
    else:
        axes[1].set_title("宫格2：无有效边界框数据")

    # 宫格3：中心点相对于整幅图的位置（二维直方图，归一化坐标）
    if len(center_x) > 0:
        h3 = axes[2].hist2d(
            center_x,
            center_y,
            bins=50,
            range=[[0, 1], [0, 1]],
        )
        fig.colorbar(h3[3], ax=axes[2])
        axes[2].set_xlabel("归一化中心点 X (0~1)")
        axes[2].set_ylabel("归一化中心点 Y (0~1)")
        axes[2].set_title("宫格3：目标中心点在图像中的位置分布")
    else:
        axes[2].set_title("宫格3：无中心点数据")

    # 宫格4：目标相对于整幅图的高宽比例（归一化宽、高二维直方图）
    if len(rel_w) > 0:
        h4 = axes[3].hist2d(
            rel_w,
            rel_h,
            bins=50,
            range=[[0, 1], [0, 1]],
        )
        fig.colorbar(h4[3], ax=axes[3])
        axes[3].set_xlabel("宽度 / 图像宽度")
        axes[3].set_ylabel("高度 / 图像高度")
        axes[3].set_title("宫格4：目标相对图像的高宽比例分布")
    else:
        axes[3].set_title("宫格4：无目标尺寸数据")

    fig.suptitle("训练集标注统计四宫格", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_labels_correlogram(stats, output_path: str = "labels_correlogram.png"):
    all_labels = stats["all_labels"]
    image_label_sets = stats["image_label_sets"]

    if not all_labels:
        return

    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n = len(all_labels)
    co_mat = np.zeros((n, n), dtype=int)

    # 统计每张图像中标签的共现情况
    for label_set in image_label_sets:
        idxs = [label_to_idx[l] for l in label_set if l in label_to_idx]
        for i, j in combinations_with_replacement(idxs, 2):
            co_mat[i, j] += 1
            if i != j:
                co_mat[j, i] += 1

    fig, ax = plt.subplots(figsize=(6 + n * 0.3, 6 + n * 0.3))
    im = ax.imshow(co_mat, cmap="viridis")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right")
    ax.set_yticklabels(all_labels)
    ax.set_xlabel("标签")
    ax.set_ylabel("标签")
    ax.set_title("标签共现矩阵（labels correlogram）")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    # 默认统计当前文件夹下的 JSON 标注（如 LabelMe 导出的 *.json）
    stats = collect_stats("*.json")

    print("共找到 JSON 标注文件数量：", stats["num_json_files"])
    print("各类别掩码数量：")
    for label, cnt in stats["class_counts"].items():
        print(f"  {label}: {cnt}")

    plot_overview(stats, output_path="annotation_overview.png")
    plot_labels_correlogram(stats, output_path="labels_correlogram.png")
    print("已生成图像文件：annotation_overview.png, labels_correlogram.png")


if __name__ == "__main__":
    main()

