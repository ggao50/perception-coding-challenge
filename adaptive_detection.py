"""
Adaptive YOLOv8 Detection with Class-Specific Confidence Thresholds
Different confidence thresholds for different object types
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import os
from collections import deque

class AdaptiveDetector:
    def __init__(self):
        # Load custom trained model
        custom_model_path = 'runs/driving_detection/yolov8m_driving/weights/best.pt'
        if Path(custom_model_path).exists():
            print(f"Loading custom trained YOLOv8 model from {custom_model_path}...")
            self.yolo = YOLO(custom_model_path)
        else:
            print("Custom model not found, using default YOLOv8n")
            self.yolo = YOLO('yolov8n.pt')

        # Class names from your custom model
        self.class_names = ['golf_cart', 'green_traffic_light', 'pedestrian', 'red_traffic_light', 'traffic_barrier']

        # Class-specific confidence thresholds based on analysis
        self.class_thresholds = {
            'golf_cart': 0.50,           # High confidence, can use higher threshold
            'green_traffic_light': 0.10,  # Very low to catch weak detections
            'pedestrian': 0.25,           # Medium threshold
            'red_traffic_light': 0.10,    # Very low to catch weak detections
            'traffic_barrier': 0.30       # Medium-high threshold
        }

        # Colors for each class (BGR format)
        self.class_colors = {
            'golf_cart': (0, 255, 255),  # Yellow
            'green_traffic_light': (0, 255, 0),  # Green
            'pedestrian': (255, 0, 0),  # Blue
            'red_traffic_light': (0, 0, 255),  # Red
            'traffic_barrier': (255, 0, 255)  # Magenta
        }

        # Detection history for temporal smoothing
        self.detection_history = deque(maxlen=5)
        self.prev_detections = []

    def iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def interpolate_missing_detections(self, current_detections):
        """Fill in missing detections using previous frames"""
        interpolated = list(current_detections)

        for prev_det in self.prev_detections:
            found_match = False
            for curr_det in current_detections:
                if (prev_det['class'] == curr_det['class'] and
                    self.iou(prev_det['box'], curr_det['box']) > 0.3):
                    found_match = True
                    break

            if not found_match:
                prev_det['confidence'] *= 0.8
                # Use class-specific minimum threshold for interpolation
                min_threshold = self.class_thresholds.get(prev_det['class'], 0.1) * 0.5
                if prev_det['confidence'] > min_threshold:
                    prev_det['interpolated'] = True
                    interpolated.append(prev_det.copy())

        return interpolated

    def detect_and_draw(self, img, use_stabilization=True):
        """
        Run detection with class-specific confidence thresholds
        """
        # Run YOLO with very low threshold to get all possible detections
        results = self.yolo(img, conf=0.05, verbose=False)

        current_detections = []

        # Process detections with class-specific filtering
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    cls_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])

                    if cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id]
                    else:
                        class_name = f"class_{cls_id}"

                    # Apply class-specific threshold
                    threshold = self.class_thresholds.get(class_name, 0.25)
                    if confidence >= threshold:
                        current_detections.append({
                            'box': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': confidence,
                            'threshold': threshold,
                            'interpolated': False
                        })

        # Apply temporal smoothing if enabled
        if use_stabilization and self.prev_detections:
            current_detections = self.interpolate_missing_detections(current_detections)

        # Update history
        self.prev_detections = current_detections.copy()

        # Draw all detections
        for det in current_detections:
            x1, y1, x2, y2 = det['box']
            class_name = det['class']
            confidence = det['confidence']
            threshold = det.get('threshold', 0.25)
            is_interpolated = det.get('interpolated', False)

            color = self.class_colors.get(class_name, (255, 255, 255))
            thickness = 1 if is_interpolated else 2

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label with confidence and threshold info
            label = f"{class_name}: {confidence:.2f}"
            if is_interpolated:
                label += " (tracked)"
            else:
                label += f" [>{threshold:.2f}]"

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw label background
            cv2.rectangle(img, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(img, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

    def process_images(self, input_folder="dataset/rgb", output_video="adaptive_detections.mp4"):
        """
        Process all images with adaptive class-specific thresholds
        """
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

        if not image_files:
            print(f"No PNG files found in {input_folder}")
            return

        print(f"Found {len(image_files)} images to process")
        print("\nUsing class-specific confidence thresholds:")
        for class_name, threshold in self.class_thresholds.items():
            print(f"  {class_name}: {threshold:.2f}")
        print()

        # Read first image to get dimensions
        first_img = cv2.imread(os.path.join(input_folder, image_files[0]))
        height, width = first_img.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

        # Reset detection history
        self.detection_history.clear()
        self.prev_detections = []

        # Process each image
        for idx, img_file in enumerate(image_files):
            if idx % 30 == 0:
                print(f"Processing frame {idx}/{len(image_files)}")

            img_path = os.path.join(input_folder, img_file)
            img = cv2.imread(img_path)

            # Detect and draw with adaptive thresholds
            img_with_detections = self.detect_and_draw(img.copy(), use_stabilization=True)

            out.write(img_with_detections)

        out.release()
        print(f"\nSaved adaptive detection video to {output_video}")

    def analyze_with_adaptive_thresholds(self, input_folder="dataset/rgb"):
        """
        Analyze detection improvement with adaptive thresholds
        """
        print("\nAnalyzing detection improvement with adaptive thresholds...")
        print("-" * 60)

        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])[:30]

        # Count detections with default vs adaptive thresholds
        default_counts = {class_name: 0 for class_name in self.class_names}
        adaptive_counts = {class_name: 0 for class_name in self.class_names}

        for img_file in image_files:
            img_path = os.path.join(input_folder, img_file)
            img = cv2.imread(img_path)

            # Test with single threshold (0.25)
            results_default = self.yolo(img, conf=0.25, verbose=False)
            for result in results_default:
                if result.boxes is not None:
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i])
                        if cls_id < len(self.class_names):
                            default_counts[self.class_names[cls_id]] += 1

            # Test with adaptive thresholds
            results_adaptive = self.yolo(img, conf=0.05, verbose=False)
            for result in results_adaptive:
                if result.boxes is not None:
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i])
                        confidence = float(result.boxes.conf[i])
                        if cls_id < len(self.class_names):
                            class_name = self.class_names[cls_id]
                            if confidence >= self.class_thresholds[class_name]:
                                adaptive_counts[class_name] += 1

        print("\nDetection counts (30 frames):")
        print(f"{'Class':<20} {'Default (0.25)':<15} {'Adaptive':<15} {'Improvement':<15}")
        print("-" * 65)

        for class_name in self.class_names:
            default = default_counts[class_name]
            adaptive = adaptive_counts[class_name]
            improvement = ((adaptive - default) / max(default, 1)) * 100 if default > 0 else 100 if adaptive > 0 else 0
            print(f"{class_name:<20} {default:<15} {adaptive:<15} {improvement:+.1f}%")


if __name__ == "__main__":
    detector = AdaptiveDetector()

    # Analyze improvement with adaptive thresholds
    detector.analyze_with_adaptive_thresholds()

    # Process with adaptive thresholds
    print("\n" + "="*60)
    print("Processing with adaptive class-specific thresholds...")
    print("="*60)
    detector.process_images()