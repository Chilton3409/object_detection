#!/usr/bin/env python3
#Refactored code

import cv2
import numpy as np
import pyttsx3
import sys
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
import queue
import threading
# Constants
MODEL_PATH = "./object_detection_models/yolov4.weights"
CONFIG_PATH = "./object_detection_models/yolov4.cfg"
CLASS_PATH = "./object_detection_models/coco.names"
CONFIDENCE_THRESHOLD = 0.5

class YOLOv4ObjectDetector:
    def __init__(self, model_path=MODEL_PATH, config_path=CONFIG_PATH, class_path=CLASS_PATH):
        """
        Initializes the YOLOv4 object detector.

        Args:
            model_path (str): Path to the YOLOv4 model weights.
            config_path (str): Path to the YOLOv4 model configuration.
            class_path (str): Path to the COCO class names.
        """
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = self._load_classes(class_path)
        self.confidence_threshold = 0.5
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Set speech rate
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self.speak_objects)
        self.speech_lock = threading.Lock()
        self.speech_thread.daemon = True  # So
        self.speech_thread.start()
    def _load_classes(self, class_path):
        """
        Loads the COCO class names.

        Args:
            class_path (str): Path to the COCO class names.

        Returns:
            list: List of class names.
        """
        with open(class_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def detect_objects(self, frame):
        """
        Performs object detection on the given frame.

        Args:
            frame (np.ndarray): Input frame.

        Returns:
            list: List of detected objects with their class IDs, confidence scores, and bounding boxes.
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        detected_objects = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold and class_id in range(80):
                    x, y, w, h = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    detected_objects.append({
                        "class_id": class_id,
                        "confidence": confidence,
                        "bounding_box": (int(x), int(y), int(x+w), int(y+h))
                    })

        return detected_objects

    def draw_bounding_boxes(self, frame, detected_objects):
        """
        Draws bounding boxes and class labels on the given frame.

        Args:
            frame (np.ndarray): Input frame.
            detected_objects (list): List of detected objects.

        Returns:
            np.ndarray: Frame with bounding boxes and class labels.
        """
        announced_objects = set()
        for obj in detected_objects:
            cv2.rectangle(frame, (obj["bounding_box"][0], obj["bounding_box"][1]), (obj["bounding_box"][2], obj["bounding_box"][3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{self.classes[obj['class_id']]} {obj['confidence']:.2f}", (obj["bounding_box"][0], obj["bounding_box"][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            object_name = self.classes[obj['class_id']]
            if object_name not in announced_objects:
                self.speech_queue.put(object_name)
                announced_objects.add(object_name)
            print(f"Object class ID: {object_name}")
            print(f"Confidence score: {obj['confidence']}")
        return frame

           
    
    def speak_objects(self):
        pass

            
    def stop_speech(self):
        pass





class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            detected_objects = self.detector.detect_objects(frame)
            frame = self.detector.draw_bounding_boxes(frame, detected_objects)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.change_pixmap_signal.emit(frame)
        cap.release()

    def stop(self):
        self._run_flag = False


class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = YOLOv4ObjectDetector()
        self.video_thread = VideoThread(self.detector)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.initUI()
        self.video_thread.start()

    def initUI(self):
        self.setWindowTitle("Object Detection App")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        self.start_button = QPushButton("Start")
        #self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_detection)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        
    def stop_detection(self):
        self.video_thread.stop()
        self.video_thread.wait()
        
        





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
