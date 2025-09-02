import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLOWorld

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
)


def np_to_qimage(img: np.ndarray) -> QImage:
    """Convert a HxWxC RGB numpy array to QImage."""
    if img is None:
        raise ValueError("Image array is None")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB image array")
    h, w, _ = img.shape
    # Ensure contiguous memory
    img_rgb = np.ascontiguousarray(img)
    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    # Make a deep copy so data is owned by QImage
    return qimg.copy()


class ImageLabel(QLabel):
    """A QLabel that keeps a reference to the original QPixmap and scales it preserving aspect ratio."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap_original: Optional[QPixmap] = None

    def setPixmap(self, pm: QPixmap):  # type: ignore[override]
        self._pixmap_original = pm
        super().setPixmap(self.scaled_pixmap())

    def scaled_pixmap(self) -> Optional[QPixmap]:
        if not self._pixmap_original:
            return None
        if self.width() <= 0 or self.height() <= 0:
            return self._pixmap_original
        return self._pixmap_original.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def resizeEvent(self, event):  # noqa: N802
        if self._pixmap_original:
            super().setPixmap(self.scaled_pixmap())
        super().resizeEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Label - YOLOWorld (PySide)")
        self.model: Optional[YOLOWorld] = None
        self.weights_path = "yolov8s-world.pt"
        self.selected_image_path: Optional[str] = None

        # UI elements
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("e.g., person, car, traffic light")

        self.select_button = QPushButton("Select Image…")
        self.run_button = QPushButton("Run Inference")
        self.run_button.setEnabled(False)

        self.info_label = QLabel("No image selected.")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Image views (with scroll areas)
        self.input_image_label = ImageLabel("Input image will appear here")
        self.ann_image_label = ImageLabel("Annotated image will appear here")

        input_scroll = QScrollArea()
        input_scroll.setWidgetResizable(True)
        input_scroll.setWidget(self.input_image_label)

        ann_scroll = QScrollArea()
        ann_scroll.setWidgetResizable(True)
        ann_scroll.setWidget(self.ann_image_label)

        # Layouts
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Classes (comma-separated):"))
        top_row.addWidget(self.classes_input, 1)
        top_row.addWidget(self.select_button)
        top_row.addWidget(self.run_button)

        images_row = QHBoxLayout()
        images_row.addWidget(input_scroll, 1)
        images_row.addWidget(ann_scroll, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_row)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(images_row)
        self.setLayout(main_layout)

        # Signals
        self.select_button.clicked.connect(self.on_select_image)
        self.run_button.clicked.connect(self.on_run_inference)

        # Initial hint
        if not os.path.exists(self.weights_path):
            self.info_label.setText(
                "Weights file 'yolov8s-world.pt' not found. Place it in project root."
            )

    def ensure_model(self) -> YOLOWorld:
        if self.model is None:
            try:
                self.model = YOLOWorld(self.weights_path)
            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", str(e))
                raise
        return self.model

    def on_select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            os.getcwd(),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not file_path:
            return
        self.selected_image_path = file_path
        self.info_label.setText(f"Selected: {file_path}")
        self.run_button.setEnabled(True)

        try:
            img = Image.open(file_path).convert("RGB")
            np_img = np.array(img)
            qimg = np_to_qimage(np_img)
            self.input_image_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            QMessageBox.warning(self, "Image Load Error", str(e))

    @staticmethod
    def parse_classes(text: str) -> List[str]:
        # Split by comma, strip whitespace, remove empties
        classes = [c.strip() for c in text.split(",") if c.strip()]
        return classes

    def on_run_inference(self):
        if not self.selected_image_path:
            QMessageBox.information(self, "No Image", "Please select an image first.")
            return

        classes_text = self.classes_input.text()
        classes = self.parse_classes(classes_text)
        if not classes:
            # Provide a gentle hint but proceed with defaults (open-vocabulary)
            ret = QMessageBox.question(
                self,
                "No Classes Provided",
                "No classes specified. Proceed with default model prompts?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.No:
                return

        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        QApplication.processEvents()

        try:
            model = self.ensure_model()
            if classes:
                model.set_classes(classes)
            results_list = model.predict(self.selected_image_path, conf=0.25)
            if not results_list:
                QMessageBox.warning(self, "Inference", "Model returned no results.")
                return
            result = results_list[0]

            # Ultralytics returns BGR image from .plot(); convert to RGB for Qt
            ann_bgr = result.plot()
            ann_rgb = ann_bgr[..., ::-1]
            qimg_ann = np_to_qimage(ann_rgb)
            self.ann_image_label.setPixmap(QPixmap.fromImage(qimg_ann))
            self.info_label.setText(
                f"Done. Classes: {', '.join(classes) if classes else '(default)'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Inference Error", str(e))
        finally:
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Inference")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(QSize(1200, 700))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()