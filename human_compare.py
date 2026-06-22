from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


RESULTS_FILE = Path("results") / "lesion_leaf_results.json"
IDENTITIES = [
    {"key": "pro_expert", "label": "专业植保人员"},
    {"key": "student", "label": "接触过病害调查的学生"},
    {"key": "novice", "label": "完全不了解病害调查工作的人员"},
]
COCO_TARGET_CATEGORIES = {"Leaf", "Lesion"}


@dataclass
class ImageEntry:
    image_id: int
    name: str
    image_path: Path
    ratio: float  # 0-1 区间


def load_coco_data(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "images" not in data or "annotations" not in data or "categories" not in data:
        raise ValueError("JSON 结构缺少 images / annotations / categories 字段")
    return data


def segmentation_to_polygon(seg: Sequence[float]) -> List[Tuple[float, float]]:
    coords = list(seg)
    if len(coords) < 6 or len(coords) % 2 != 0:
        return []
    return [
        (float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)
    ]


def build_mask_from_polygons(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    size: Tuple[int, int],
) -> np.ndarray:
    mask = Image.new("1", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        draw.polygon(polygon, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def compute_ratio_from_polygons(
    image_info: dict,
    polygons: Dict[str, List[Sequence[Tuple[float, float]]]],
) -> float:
    width, height = int(image_info["width"]), int(image_info["height"])
    leaf_polygons = polygons.get("Leaf", [])
    if not leaf_polygons:
        raise ValueError("Leaf 区域为空")

    leaf_mask = build_mask_from_polygons(leaf_polygons, (width, height))
    lesion_polygons = polygons.get("Lesion", [])
    if lesion_polygons:
        lesion_mask = build_mask_from_polygons(lesion_polygons, (width, height))
    else:
        lesion_mask = np.zeros_like(leaf_mask, dtype=np.uint8)

    leaf_pixels = int(leaf_mask.sum())
    if leaf_pixels == 0:
        raise ValueError("Leaf 像素为 0")

    lesion_on_leaf = int(np.logical_and(leaf_mask, lesion_mask).sum())
    return lesion_on_leaf / leaf_pixels


def collect_entries(image_folder: Path, annotation_path: Path) -> List[ImageEntry]:
    data = load_coco_data(annotation_path)
    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_map = {img["id"]: img for img in data["images"]}
    polygons_by_image: Dict[int, Dict[str, List[List[Tuple[float, float]]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for ann in data["annotations"]:
        cat_name = category_map.get(ann.get("category_id"))
        if cat_name not in COCO_TARGET_CATEGORIES:
            continue
        segmentation = ann.get("segmentation") or []
        if isinstance(segmentation, dict):
            print(f"[警告] 跳过 RLE 标注 image_id={ann.get('image_id')}")
            continue
        for seg in segmentation:
            polygon = segmentation_to_polygon(seg)
            if len(polygon) >= 3:
                polygons_by_image[ann["image_id"]][cat_name].append(polygon)

    entries: List[ImageEntry] = []
    for image_id, polygon_map in polygons_by_image.items():
        image_info = image_map.get(image_id)
        if not image_info:
            print(f"[警告] 找不到 image_id={image_id} 的图像信息")
            continue
        image_path = (image_folder / image_info["file_name"]).resolve()
        if not image_path.exists():
            print(f"[警告] 图像文件缺失：{image_path}")
            continue
        try:
            ratio = compute_ratio_from_polygons(image_info, polygon_map)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] {image_info['file_name']} 计算失败：{exc}")
            continue
        entries.append(
            ImageEntry(
                image_id=image_id,
                name=image_info["file_name"],
                image_path=image_path,
                ratio=ratio,
            )
        )

    entries.sort(key=lambda entry: entry.name.lower())
    return entries


def load_results() -> Dict[str, dict]:
    if RESULTS_FILE.exists():
        with RESULTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(data: Dict[str, dict]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_store(
    store: Dict[str, dict],
    entry: ImageEntry,
    identity: dict,
    estimation_percent: float,
    *,
    session_index: int,
) -> None:
    record = store.setdefault(
        entry.name,
        {
            "image_name": entry.name,
            "computed_ratio": round(entry.ratio, 6),
            "evaluations": {},
        },
    )
    record["computed_ratio"] = round(entry.ratio, 6)

    identity_rec = record["evaluations"].setdefault(identity["key"], [])
    identity_rec.append(
        {
            "value_percent": round(estimation_percent, 2),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "identity_label": identity["label"],
            "session_index": session_index,
        }
    )


def export_results(output_path: Path, store: Optional[Dict[str, dict]] = None) -> Path:
    data = store if store is not None else load_results()
    if not data:
        raise ValueError("暂无可导出的结果。")

    # 统计每个身份的最大录入次数
    max_sessions: Dict[str, int] = {}
    for record in data.values():
        for identity in IDENTITIES:
            evaluations = record.get("evaluations", {}).get(identity["key"], [])
            if evaluations:
                max_idx = 0
                for eval_item in evaluations:
                    session_idx = eval_item.get("session_index")
                    if session_idx is not None:
                        max_idx = max(max_idx, int(session_idx))
                    else:
                        # 如果没有session_index，按顺序编号
                        max_idx = max(max_idx, len(evaluations))
                identity_key = identity["key"]
                max_sessions[identity_key] = max(
                    max_sessions.get(identity_key, 0), max_idx
                )

    # 构建列名：图像名称、JSON计算比例、各身份的多列
    columns = ["图像名称", "JSON计算比例(%)"]
    column_map: Dict[Tuple[str, int], str] = {}  # (identity_key, session_index) -> column_name
    for identity in IDENTITIES:
        identity_key = identity["key"]
        max_count = max_sessions.get(identity_key, 0)
        for session_idx in range(1, max_count + 1):
            col_name = f"{identity['label']}_{session_idx}"
            columns.append(col_name)
            column_map[(identity_key, session_idx)] = col_name

    # 构建行数据
    rows = []
    for record in data.values():
        row = {
            "图像名称": record["image_name"],
            "JSON计算比例(%)": round(record["computed_ratio"] * 100, 2),
        }
        # 初始化所有身份列为空
        for col in columns[2:]:
            row[col] = None

        # 填充评估值
        for identity in IDENTITIES:
            identity_key = identity["key"]
            evaluations = record.get("evaluations", {}).get(identity_key, [])
            # 按session_index分组，每个session只取最后一次评估
            session_values: Dict[int, float] = {}
            for fallback_idx, evaluation in enumerate(evaluations, start=1):
                session_idx = int(evaluation.get("session_index") or fallback_idx)
                value = evaluation.get("value_percent")
                if value is not None:
                    # 如果同一session有多次评估，保留最后一次
                    session_values[session_idx] = value
            # 将值填入对应列
            for session_idx, value in session_values.items():
                col_name = column_map.get((identity_key, session_idx))
                if col_name:
                    row[col_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    return output_path


class LesionApp(QMainWindow):
    def __init__(
        self,
        initial_images: Optional[Path] = None,
        initial_annotations: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Lesion/Leaf 评估软件")
        self.resize(1280, 800)

        self.entries: List[ImageEntry] = []
        self.current_index: int = -1
        self.identity: Optional[dict] = None
        self.image_folder: Optional[Path] = None
        self.annotation_path: Optional[Path] = None
        self.store: Dict[str, dict] = load_results()
        self._current_pixmap: Optional[QPixmap] = None
        self.session_entries: Dict[str, float] = {}
        self.current_session_index: int = 1

        self._build_ui()
        if initial_images and initial_images.is_dir():
            self.image_folder = initial_images
            self.folder_label.setText(str(initial_images))
        if initial_annotations and initial_annotations.is_file():
            self.annotation_path = initial_annotations
            self.annotation_label.setText(str(initial_annotations))
        self.try_load_entries()
        self.update_session_label()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # 左侧
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        identity_layout = QHBoxLayout()
        identity_layout.addWidget(QLabel("身份："))
        self.identity_combo = QComboBox()
        self.identity_combo.addItem("请选择身份", userData=None)
        for identity in IDENTITIES:
            self.identity_combo.addItem(identity["label"], userData=identity)
        self.identity_combo.currentIndexChanged.connect(self.on_identity_changed)
        identity_layout.addWidget(self.identity_combo)
        left_layout.addLayout(identity_layout)

        self.identity_session_label = QLabel("当前录入编号：-")
        self.identity_session_label.setWordWrap(True)
        left_layout.addWidget(self.identity_session_label)

        folder_layout = QHBoxLayout()
        folder_btn = QPushButton("选择图像文件夹")
        folder_btn.clicked.connect(self.select_folder)
        self.folder_label = QLabel("未选择")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(folder_btn)
        folder_layout.addWidget(self.folder_label)
        left_layout.addLayout(folder_layout)

        annotation_layout = QHBoxLayout()
        annotation_btn = QPushButton("选择标注 JSON")
        annotation_btn.clicked.connect(self.select_annotation)
        self.annotation_label = QLabel("未选择")
        self.annotation_label.setWordWrap(True)
        annotation_layout.addWidget(annotation_btn)
        annotation_layout.addWidget(self.annotation_label)
        left_layout.addLayout(annotation_layout)

        self.progress_label = QLabel("未加载数据")
        left_layout.addWidget(self.progress_label)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_current_row_changed)
        left_layout.addWidget(self.list_widget, stretch=1)

        left_widget.setMinimumWidth(320)
        splitter.addWidget(left_widget)

        # 右侧
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        self.image_label = QLabel("图像预览区")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background: #f8f8f8;")
        right_layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel("请先选择身份与数据")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        estimation_layout = QHBoxLayout()
        estimation_layout.addWidget(QLabel("目视评估："))
        self.estimate_spin = QDoubleSpinBox()
        self.estimate_spin.setRange(0.0, 100.0)
        self.estimate_spin.setDecimals(2)
        self.estimate_spin.setSingleStep(1.0)
        self.estimate_spin.setSuffix(" %")
        estimation_layout.addWidget(self.estimate_spin)
        right_layout.addLayout(estimation_layout)

        self.last_eval_label = QLabel("当前身份尚无评估记录")
        self.last_eval_label.setWordWrap(True)
        right_layout.addWidget(self.last_eval_label)

        self.session_progress_label = QLabel("本次录入进度：0/0")
        self.session_progress_label.setWordWrap(True)
        right_layout.addWidget(self.session_progress_label)

        button_layout = QHBoxLayout()
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.go_to_next_image)
        button_layout.addWidget(self.next_btn)

        self.submit_btn = QPushButton("提交当前结果")
        self.submit_btn.clicked.connect(self.submit_current)
        button_layout.addWidget(self.submit_btn)

        self.export_btn = QPushButton("导出Excel")
        self.export_btn.clicked.connect(self.export_dialog)
        button_layout.addWidget(self.export_btn)

        right_layout.addLayout(button_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # 回车提交
        self.estimate_spin.lineEdit().returnPressed.connect(self.submit_current)

    # -------- 数据加载 -------- #
    def select_folder(self) -> None:
        start_dir = str(self.image_folder or Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", start_dir)
        if folder:
            self.image_folder = Path(folder)
            self.folder_label.setText(folder)
            self.try_load_entries()

    def select_annotation(self) -> None:
        start_dir = (
            str(self.annotation_path.parent) if self.annotation_path else str(Path.cwd())
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 COCO 标注 JSON", start_dir, "JSON 文件 (*.json)"
        )
        if file_path:
            self.annotation_path = Path(file_path)
            self.annotation_label.setText(file_path)
            self.try_load_entries()

    def try_load_entries(self) -> None:
        if not self.image_folder or not self.annotation_path:
            return
        try:
            entries = collect_entries(self.image_folder, self.annotation_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "加载失败", str(exc))
            return

        if not entries:
            QMessageBox.warning(self, "提示", "未在标注中找到符合要求的图像。")
            return

        self.entries = entries
        self.progress_label.setText(f"共加载 {len(entries)} 张图像")
        self.session_entries.clear()
        self.populate_list()
        self.list_widget.setCurrentRow(0)
        if self.identity:
            self.reset_session_state()
        else:
            self.refresh_session_progress()
        self.statusBar().showMessage("数据加载完成", 4000)

    def populate_list(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for entry in self.entries:
            self.list_widget.addItem(self.entry_status_text(entry))
        self.list_widget.blockSignals(False)

    def entry_status_text(self, entry: ImageEntry) -> str:
        record = self.store.get(entry.name)
        status_parts = []
        if record:
            total = sum(len(v) for v in record.get("evaluations", {}).values())
            if total:
                status_parts.append(f"总评估{total}次")
            if self.identity:
                count = len(record.get("evaluations", {}).get(self.identity["key"], []))
                if count:
                    status_parts.append(f"当前身份{count}次")
        if entry.name in self.session_entries:
            status_parts.append("本次已填")
        status = " | ".join(status_parts)
        suffix = f" - {status}" if status else ""
        return f"{entry.name} (计算{entry.ratio * 100:.2f}%) {suffix}"

    # -------- 交互逻辑 -------- #
    def on_identity_changed(self, _: int) -> None:
        new_identity = self.identity_combo.currentData()
        if new_identity is None:
            self.identity = None
            self.reset_session_state()
            self.populate_list()
            self.update_detail_panel()
            return

        if self.session_entries and new_identity != self.identity:
            reply = QMessageBox.question(
                self,
                "确认切换身份",
                "当前身份有未提交的录入，切换身份将丢弃这些数据，是否继续？",
            )
            if reply != QMessageBox.Yes:
                self.identity_combo.blockSignals(True)
                index = self.identity_combo.findData(self.identity)
                self.identity_combo.setCurrentIndex(index if index >= 0 else 0)
                self.identity_combo.blockSignals(False)
                return

        self.identity = new_identity
        self.reset_session_state()
        self.populate_list()
        self.update_detail_panel()

    def on_current_row_changed(self, row: int) -> None:
        self.current_index = row
        self.update_detail_panel()

    def update_detail_panel(self) -> None:
        if not (0 <= self.current_index < len(self.entries)):
            self.image_label.setText("请选择图像")
            self.info_label.setText("无图像")
            self.last_eval_label.setText("当前身份尚无评估记录")
            self.prepare_input_state()
            return

        entry = self.entries[self.current_index]
        self.info_label.setText(
            f"图像：{entry.image_path}\nJSON 计算比例：{entry.ratio * 100:.2f}%"
        )

        pixmap = QPixmap(str(entry.image_path))
        if pixmap.isNull():
            self.image_label.setText("无法加载图像")
            self._current_pixmap = None
        else:
            self._current_pixmap = pixmap
            self._update_image_pixmap()

        if self.identity:
            latest = self.latest_estimation(entry.name, self.identity["key"])
            if latest is not None:
                session_idx = self.latest_estimation_session(entry.name, self.identity["key"])
                label_text = (
                    f"当前身份上次评估：{latest:.2f}% "
                    f"(编号#{session_idx})" if session_idx else f"当前身份上次评估：{latest:.2f}%"
                )
                self.last_eval_label.setText(label_text)
            else:
                self.last_eval_label.setText("当前身份尚无评估记录")
        else:
            self.last_eval_label.setText("请先选择身份")

        if entry.name in self.session_entries:
            current_value = self.session_entries[entry.name]
            self.last_eval_label.setText(
                f"{self.last_eval_label.text()} | 本次已填：{current_value:.2f}%"
            )

        self.prepare_input_state()

    def latest_estimation(self, image_name: str, identity_key: str) -> Optional[float]:
        record = self.store.get(image_name)
        if not record:
            return None
        evaluations = record.get("evaluations", {}).get(identity_key, [])
        if not evaluations:
            return None
        return evaluations[-1]["value_percent"]

    def latest_estimation_session(self, image_name: str, identity_key: str) -> Optional[int]:
        record = self.store.get(image_name)
        if not record:
            return None
        evaluations = record.get("evaluations", {}).get(identity_key, [])
        if not evaluations:
            return None
        return evaluations[-1].get("session_index")

    def compute_next_session_index(self, identity_key: str) -> int:
        max_idx = 0
        for record in self.store.values():
            for item in record.get("evaluations", {}).get(identity_key, []):
                max_idx = max(max_idx, int(item.get("session_index", 1)))
        return max_idx + 1

    def reset_session_state(self) -> None:
        self.session_entries.clear()
        if self.identity:
            self.current_session_index = self.compute_next_session_index(
                self.identity["key"]
            )
        else:
            self.current_session_index = 1
        self.update_session_label()
        self.refresh_session_progress()

    def update_session_label(self) -> None:
        if self.identity:
            self.identity_session_label.setText(
                f"当前录入编号：{self.identity['label']} #{self.current_session_index}"
            )
        else:
            self.identity_session_label.setText("当前录入编号：-")

    def refresh_session_progress(self) -> None:
        total = len(self.entries)
        finished = len(self.session_entries)
        if total == 0:
            self.session_progress_label.setText("本次录入进度：0/0")
        else:
            self.session_progress_label.setText(
                f"本次录入进度：{finished}/{total}"
            )

    def prepare_input_state(self) -> None:
        self.estimate_spin.blockSignals(True)
        self.estimate_spin.setValue(0.0)
        self.estimate_spin.blockSignals(False)
        self.focus_input()

    def focus_input(self) -> None:
        self.estimate_spin.setFocus(Qt.OtherFocusReason)
        self.estimate_spin.lineEdit().selectAll()

    def go_to_next_image(self) -> None:
        if not self.entries:
            return
        next_index = min(self.current_index + 1, len(self.entries) - 1)
        if next_index == self.current_index:
            self.statusBar().showMessage("已经是最后一张。", 3000)
            return
        self.list_widget.setCurrentRow(next_index)

    def advance_to_next_image(self) -> None:
        if self.current_index < len(self.entries) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)
        else:
            self.statusBar().showMessage("已到最后一张。", 3000)
            self.prepare_input_state()

    def finalize_session(self) -> None:
        if not self.identity or not self.session_entries:
            return
        for entry in self.entries:
            if entry.name not in self.session_entries:
                continue
            update_store(
                self.store,
                entry,
                self.identity,
                self.session_entries[entry.name],
                session_index=self.current_session_index,
            )
        save_results(self.store)
        self.store = load_results()
        QMessageBox.information(
            self,
            "提交成功",
            f"{self.identity['label']} 第 {self.current_session_index} 次录入已保存。",
        )
        self.populate_list()
        self.reset_session_state()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_image_pixmap()

    def _update_image_pixmap(self) -> None:
        if self._current_pixmap is None:
            return
        target_rect = self.image_label.contentsRect()
        if target_rect.width() <= 0 or target_rect.height() <= 0:
            return
        scaled = self._current_pixmap.scaled(
            target_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def submit_current(self) -> None:
        if self.identity is None:
            QMessageBox.warning(self, "提示", "请先选择身份。")
            return
        if not (0 <= self.current_index < len(self.entries)):
            QMessageBox.warning(self, "提示", "请先选择图像。")
            return

        entry = self.entries[self.current_index]
        value = self.estimate_spin.value()
        self.session_entries[entry.name] = value
        self.list_widget.item(self.current_index).setText(self.entry_status_text(entry))
        self.refresh_session_progress()
        self.statusBar().showMessage("已记录，待整批提交。", 3000)
        self.advance_to_next_image()

        if len(self.session_entries) == len(self.entries):
            self.finalize_session()

    def export_dialog(self) -> None:
        default_path = (Path.cwd() / "results" / "lesion_leaf_report.xlsx").resolve()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Excel",
            str(default_path),
            "Excel 文件 (*.xlsx)",
        )
        if not path:
            return
        try:
            output = export_results(Path(path), self.store)
            QMessageBox.information(self, "导出成功", f"已导出：\n{output}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lesion / Leaf 比例评估 GUI 软件")
    parser.add_argument(
        "--images",
        type=Path,
        help="启动时自动加载的图像文件夹路径",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="启动时自动加载的 COCO 标注 JSON 路径",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = LesionApp(args.images, args.annotations)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

