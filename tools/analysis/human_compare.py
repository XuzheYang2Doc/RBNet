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
    {"key": "pro_expert", "label": "Plant protection expert"},
    {"key": "student", "label": "Student with survey experience"},
    {"key": "novice", "label": "Novice without survey experience"},
]
COCO_TARGET_CATEGORIES = {"Leaf", "Lesion"}


@dataclass
class ImageEntry:
    image_id: int
    name: str
    image_path: Path
    ratio: float  # 0-1 range


def load_coco_data(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "images" not in data or "annotations" not in data or "categories" not in data:
        raise ValueError("JSON must contain images, annotations, and categories")
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
        raise ValueError("Leaf region is empty")

    leaf_mask = build_mask_from_polygons(leaf_polygons, (width, height))
    lesion_polygons = polygons.get("Lesion", [])
    if lesion_polygons:
        lesion_mask = build_mask_from_polygons(lesion_polygons, (width, height))
    else:
        lesion_mask = np.zeros_like(leaf_mask, dtype=np.uint8)

    leaf_pixels = int(leaf_mask.sum())
    if leaf_pixels == 0:
        raise ValueError("Leaf pixel count is zero")

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
            print(f"[Warning] skipped RLE annotation image_id={ann.get('image_id')}")
            continue
        for seg in segmentation:
            polygon = segmentation_to_polygon(seg)
            if len(polygon) >= 3:
                polygons_by_image[ann["image_id"]][cat_name].append(polygon)

    entries: List[ImageEntry] = []
    for image_id, polygon_map in polygons_by_image.items():
        image_info = image_map.get(image_id)
        if not image_info:
            print(f"[Warning] missing image_id={image_id} image metadata")
            continue
        image_path = (image_folder / image_info["file_name"]).resolve()
        if not image_path.exists():
            print(f"[Warning] missing image file: {image_path}")
            continue
        try:
            ratio = compute_ratio_from_polygons(image_info, polygon_map)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] {image_info['file_name']} failed: {exc}")
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
        raise ValueError("No results to export.")

    # Count the maximum number of sessions for each identity
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
                        # Fall back to sequential numbering when session_index is missing
                        max_idx = max(max_idx, len(evaluations))
                identity_key = identity["key"]
                max_sessions[identity_key] = max(
                    max_sessions.get(identity_key, 0), max_idx
                )

    # Build columns for image name, computed ratio, and each evaluator group
    columns = ["Image name", "Computed ratio (%)"]
    column_map: Dict[Tuple[str, int], str] = {}  # (identity_key, session_index) -> column_name
    for identity in IDENTITIES:
        identity_key = identity["key"]
        max_count = max_sessions.get(identity_key, 0)
        for session_idx in range(1, max_count + 1):
            col_name = f"{identity['label']}_{session_idx}"
            columns.append(col_name)
            column_map[(identity_key, session_idx)] = col_name

    # Build row records
    rows = []
    for record in data.values():
        row = {
            "Image name": record["image_name"],
            "Computed ratio (%)": round(record["computed_ratio"] * 100, 2),
        }
        # Initialize evaluator columns
        for col in columns[2:]:
            row[col] = None

        # Fill evaluation values
        for identity in IDENTITIES:
            identity_key = identity["key"]
            evaluations = record.get("evaluations", {}).get(identity_key, [])
            # Group by session_index and keep the last value per session
            session_values: Dict[int, float] = {}
            for fallback_idx, evaluation in enumerate(evaluations, start=1):
                session_idx = int(evaluation.get("session_index") or fallback_idx)
                value = evaluation.get("value_percent")
                if value is not None:
                    # Keep the last value if a session has multiple entries
                    session_values[session_idx] = value
            # Fill corresponding columns
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
        self.setWindowTitle("Lesion/Leaf Evaluation Tool")
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

        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        identity_layout = QHBoxLayout()
        identity_layout.addWidget(QLabel("Identity:"))
        self.identity_combo = QComboBox()
        self.identity_combo.addItem("Select identity", userData=None)
        for identity in IDENTITIES:
            self.identity_combo.addItem(identity["label"], userData=identity)
        self.identity_combo.currentIndexChanged.connect(self.on_identity_changed)
        identity_layout.addWidget(self.identity_combo)
        left_layout.addLayout(identity_layout)

        self.identity_session_label = QLabel("Current session: -")
        self.identity_session_label.setWordWrap(True)
        left_layout.addWidget(self.identity_session_label)

        folder_layout = QHBoxLayout()
        folder_btn = QPushButton("Select image folder")
        folder_btn.clicked.connect(self.select_folder)
        self.folder_label = QLabel("Not selected")
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(folder_btn)
        folder_layout.addWidget(self.folder_label)
        left_layout.addLayout(folder_layout)

        annotation_layout = QHBoxLayout()
        annotation_btn = QPushButton("Select annotation JSON")
        annotation_btn.clicked.connect(self.select_annotation)
        self.annotation_label = QLabel("Not selected")
        self.annotation_label.setWordWrap(True)
        annotation_layout.addWidget(annotation_btn)
        annotation_layout.addWidget(self.annotation_label)
        left_layout.addLayout(annotation_layout)

        self.progress_label = QLabel("No data loaded")
        left_layout.addWidget(self.progress_label)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_current_row_changed)
        left_layout.addWidget(self.list_widget, stretch=1)

        left_widget.setMinimumWidth(320)
        splitter.addWidget(left_widget)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        self.image_label = QLabel("Image preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("border: 1px solid #cccccc; background: #f8f8f8;")
        right_layout.addWidget(self.image_label, stretch=1)

        self.info_label = QLabel("Select identity and data first")
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        estimation_layout = QHBoxLayout()
        estimation_layout.addWidget(QLabel("Visual estimate:"))
        self.estimate_spin = QDoubleSpinBox()
        self.estimate_spin.setRange(0.0, 100.0)
        self.estimate_spin.setDecimals(2)
        self.estimate_spin.setSingleStep(1.0)
        self.estimate_spin.setSuffix(" %")
        estimation_layout.addWidget(self.estimate_spin)
        right_layout.addLayout(estimation_layout)

        self.last_eval_label = QLabel("No records for current identity")
        self.last_eval_label.setWordWrap(True)
        right_layout.addWidget(self.last_eval_label)

        self.session_progress_label = QLabel("Current session progress: 0/0")
        self.session_progress_label.setWordWrap(True)
        right_layout.addWidget(self.session_progress_label)

        button_layout = QHBoxLayout()
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.go_to_next_image)
        button_layout.addWidget(self.next_btn)

        self.submit_btn = QPushButton("Submit current result")
        self.submit_btn.clicked.connect(self.submit_current)
        button_layout.addWidget(self.submit_btn)

        self.export_btn = QPushButton("Export Excel")
        self.export_btn.clicked.connect(self.export_dialog)
        button_layout.addWidget(self.export_btn)

        right_layout.addLayout(button_layout)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        # Submit with Enter
        self.estimate_spin.lineEdit().returnPressed.connect(self.submit_current)

    # -------- Data loading -------- #
    def select_folder(self) -> None:
        start_dir = str(self.image_folder or Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", start_dir)
        if folder:
            self.image_folder = Path(folder)
            self.folder_label.setText(folder)
            self.try_load_entries()

    def select_annotation(self) -> None:
        start_dir = (
            str(self.annotation_path.parent) if self.annotation_path else str(Path.cwd())
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select COCO annotation JSON", start_dir, "JSON files (*.json)"
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
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        if not entries:
            QMessageBox.warning(self, "Notice", "No valid images found in the annotations.")
            return

        self.entries = entries
        self.progress_label.setText(f"Loaded {len(entries)} images")
        self.session_entries.clear()
        self.populate_list()
        self.list_widget.setCurrentRow(0)
        if self.identity:
            self.reset_session_state()
        else:
            self.refresh_session_progress()
        self.statusBar().showMessage("Data loaded", 4000)

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
                status_parts.append(f"total evaluations: {total}")
            if self.identity:
                count = len(record.get("evaluations", {}).get(self.identity["key"], []))
                if count:
                    status_parts.append(f"current identity: {count}")
        if entry.name in self.session_entries:
            status_parts.append("filled in this session")
        status = " | ".join(status_parts)
        suffix = f" - {status}" if status else ""
        return f"{entry.name} (computed {entry.ratio * 100:.2f}%) {suffix}"

    # -------- Interaction logic -------- #
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
                "Confirm identity change",
                "The current identity has unsubmitted entries. Changing identity will discard them. Continue?",
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
            self.image_label.setText("Select an image")
            self.info_label.setText("No image")
            self.last_eval_label.setText("No records for current identity")
            self.prepare_input_state()
            return

        entry = self.entries[self.current_index]
        self.info_label.setText(
            f"Image: {entry.image_path}\nComputed JSON ratio: {entry.ratio * 100:.2f}%"
        )

        pixmap = QPixmap(str(entry.image_path))
        if pixmap.isNull():
            self.image_label.setText("Failed to load image")
            self._current_pixmap = None
        else:
            self._current_pixmap = pixmap
            self._update_image_pixmap()

        if self.identity:
            latest = self.latest_estimation(entry.name, self.identity["key"])
            if latest is not None:
                session_idx = self.latest_estimation_session(entry.name, self.identity["key"])
                label_text = (
                    f"Last estimate for current identity: {latest:.2f}% "
                    f"(session #{session_idx})" if session_idx else f"Last estimate for current identity: {latest:.2f}%"
                )
                self.last_eval_label.setText(label_text)
            else:
                self.last_eval_label.setText("No records for current identity")
        else:
            self.last_eval_label.setText("Select identity first")

        if entry.name in self.session_entries:
            current_value = self.session_entries[entry.name]
            self.last_eval_label.setText(
                f"{self.last_eval_label.text()} | filled in this session：{current_value:.2f}%"
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
                f"Current session: {self.identity['label']} #{self.current_session_index}"
            )
        else:
            self.identity_session_label.setText("Current session: -")

    def refresh_session_progress(self) -> None:
        total = len(self.entries)
        finished = len(self.session_entries)
        if total == 0:
            self.session_progress_label.setText("Current session progress: 0/0")
        else:
            self.session_progress_label.setText(
                f"Current session progress: {finished}/{total}"
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
            self.statusBar().showMessage("Already at the last image.", 3000)
            return
        self.list_widget.setCurrentRow(next_index)

    def advance_to_next_image(self) -> None:
        if self.current_index < len(self.entries) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)
        else:
            self.statusBar().showMessage("Reached the last image.", 3000)
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
            "Submitted",
            f"{self.identity['label']} session {self.current_session_index} saved.",
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
            QMessageBox.warning(self, "Notice", "Select identity first。")
            return
        if not (0 <= self.current_index < len(self.entries)):
            QMessageBox.warning(self, "Notice", "Select an image first.")
            return

        entry = self.entries[self.current_index]
        value = self.estimate_spin.value()
        self.session_entries[entry.name] = value
        self.list_widget.item(self.current_index).setText(self.entry_status_text(entry))
        self.refresh_session_progress()
        self.statusBar().showMessage("Recorded and waiting for batch submission.", 3000)
        self.advance_to_next_image()

        if len(self.session_entries) == len(self.entries):
            self.finalize_session()

    def export_dialog(self) -> None:
        default_path = (Path.cwd() / "results" / "lesion_leaf_report.xlsx").resolve()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            str(default_path),
            "Excel files (*.xlsx)",
        )
        if not path:
            return
        try:
            output = export_results(Path(path), self.store)
            QMessageBox.information(self, "Export complete", f"Exported:\n{output}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lesion/leaf ratio evaluation GUI")
    parser.add_argument(
        "--images",
        type=Path,
        help="Image folder loaded at startup",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="COCO annotation JSON loaded at startup",
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

