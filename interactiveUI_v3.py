import sys
import os
import json
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, QFileDialog,
                             QListWidget, QVBoxLayout, QWidget, QHBoxLayout, QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageAnnotationApp(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.setWindowTitle("Select Positive and Negative Points")
        self.setGeometry(100, 100, 1000, 600)

        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.image_files.sort()
        self.current_image_index = 0

        self.points_dict = {}

        # Initialize GUI components
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setFixedSize(780, 500)
        self.label.mousePressEvent = self.get_points
        left_layout.addWidget(self.label)

        self.image_selector = QComboBox()
        self.image_selector.addItems(self.image_files)
        self.image_selector.currentIndexChanged.connect(self.change_image)
        left_layout.addWidget(self.image_selector)

        self.save_button = QPushButton("Save Points")
        self.save_button.clicked.connect(self.save_points)
        left_layout.addWidget(self.save_button)

        self.point_list = QListWidget()
        self.point_list.itemClicked.connect(self.delete_point)
        right_layout.addWidget(QLabel("Selected Points"))
        right_layout.addWidget(self.point_list)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Load the first image (ensures `display_image` is initialized)
        self.load_image()

    def load_image(self):
        if not hasattr(self, "point_list"):  # Ensure the widget exists
            print("Error: self.point_list is not initialized.")
            return

        self.current_image = self.image_files[self.current_image_index]
        self.image_path = os.path.join(self.folder_path, self.current_image)
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()
        self.points = self.points_dict.get(self.current_image, [])
        self.labels = [p[2] for p in self.points]
        self.update_display()
        self.update_point_list()

    def update_display(self):
        if not hasattr(self, "display_image"):
            print("Error: display_image is not initialized.")
            return

        height, width, channel = self.display_image.shape
        bytes_per_line = 3 * width
        qimg = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def get_points(self, event):
        x, y = event.pos().x(), event.pos().y()
        if y >= self.label.height():
            return

        if event.button() == Qt.LeftButton:
            color = (0, 255, 0)  # Green for positive points
            label = 1
        elif event.button() == Qt.RightButton:
            color = (255, 0, 0)  # Red for negative points
            label = 0
        else:
            return

        self.points.append((x, y, label))
        self.points_dict[self.current_image] = self.points

        cv2.circle(self.display_image, (x, y), 5, color, -1)
        self.point_list.addItem(f"{'Positive' if label == 1 else 'Negative'}: ({x}, {y})")
        self.update_display()

    def delete_point(self, item):
        index = self.point_list.row(item)
        if index >= 0:
            del self.points[index]
            self.points_dict[self.current_image] = self.points
            self.point_list.takeItem(index)
            self.display_image = self.image.copy()
            for x, y, label in self.points:
                color = (0, 255, 0) if label == 1 else (255, 0, 0)
                cv2.circle(self.display_image, (x, y), 5, color, -1)
            self.update_display()

    def update_point_list(self):
        self.point_list.clear()
        for x, y, label in self.points:
            self.point_list.addItem(f"{'Positive' if label == 1 else 'Negative'}: ({x}, {y})")

    def change_image(self, index):
        self.current_image_index = index
        self.load_image()

    def save_points(self):
        data = {img: pts for img, pts in self.points_dict.items() if pts}
        with open("selected_points.json", "w") as f:
            json.dump(data, f, indent=4)
        print("Points saved successfully!")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    folder_path = QFileDialog.getExistingDirectory(None, "Select Image Folder")
    if folder_path:
        window = ImageAnnotationApp(folder_path)
        window.show()
        sys.exit(app.exec_())
