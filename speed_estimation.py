"""
Vehicle Speed Detection Core Module
Contains the core SpeedDetectionProcessor class for speed estimation.
"""
import logging
import cv2
import torch
import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import os

logging.getLogger().setLevel(logging.ERROR)


class SpeedDetectionProcessor:
    def __init__(self, source_points, frame_width, frame_height, max_speed, conf=0.5, 
             class_id=None, blur_id=None, model_path="yolov10s.pt"):
        """
        Initialize the speed detection processor

        Args:
            source_points: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            frame_width: Width for bird's eye view (meters)
            frame_height: Height for bird's eye view (meters)
            conf: Confidence threshold (0-1)
            class_id: Specific class to detect (None for all classes)
            blur_id: Class ID to blur (None for no blur)
            model_path: Path to YOLO model
        """
        self.source_points = np.array(source_points, dtype=np.float32)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.conf = conf
        self.class_id = class_id
        self.blur_id = blur_id
        self.max_speed = max_speed

        # Bird's eye view transformation matrix
        bird_eye_view = np.array([
            [0, 0],
            [frame_width, 0],
            [frame_width, frame_height],
            [0, frame_height]
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(self.source_points, bird_eye_view)

        # Initialize tracker and model
        print("Initializing DeepSort tracker...")
        self.tracker = DeepSort(max_age=50)

        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # Load class names
        self.class_names = self._get_default_class_names()

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3))

        # Tracking data
        self.prev_positions = {}
        self.speed_accumulator = {}
        self.violation_images = []
        self.captured_violations = set()  # Track vehicles already captured
        self.violations_dir = "processed/violations"
        os.makedirs(self.violations_dir, exist_ok=True)

        print("Initialization complete!")

    def _get_default_class_names(self):
        """Get default COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    @staticmethod
    def calculate_speed(distance, fps):
        """Calculate speed in km/h"""
        return (distance * fps) * 3.6

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    @staticmethod
    def draw_corner_rect(img, bbox, line_length=15, line_thickness=4,
                        rect_thickness=3, rect_color=(255, 255, 255),
                        line_color=(0, 255, 0)):
        """Draw stylized bounding box with corner lines"""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        if rect_thickness != 0:
            cv2.rectangle(img, bbox, rect_color, rect_thickness)

        # Top Left
        cv2.line(img, (x, y), (x + line_length, y), line_color, line_thickness)
        cv2.line(img, (x, y), (x, y + line_length), line_color, line_thickness)

        # Top Right
        cv2.line(img, (x1, y), (x1 - line_length, y), line_color, line_thickness)
        cv2.line(img, (x1, y), (x1, y + line_length), line_color, line_thickness)

        # Bottom Left
        cv2.line(img, (x, y1), (x + line_length, y1), line_color, line_thickness)
        cv2.line(img, (x, y1), (x, y1 - line_length), line_color, line_thickness)

        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - line_length, y1), line_color, line_thickness)
        cv2.line(img, (x1, y1), (x1, y1 - line_length), line_color, line_thickness)

        return img

    def process_video(self, input_path, output_path):
        """Process video and detect vehicle speeds"""
        print(f"\nProcessing video: {input_path}")
        print(f"Output will be saved to: {output_path}")

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")

        # Get video properties
        video_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {video_frame_width}x{video_frame_height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        print(f"Detection confidence threshold: {self.conf}")
        print(f"Bird's eye view dimensions: {self.frame_width}m x {self.frame_height}m")
        print(f"Source polygon points:")
        for i, point in enumerate(self.source_points):
            labels = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
            print(f"  {labels[i]}: ({point[0]:.0f}, {point[1]:.0f})")

        if self.class_id is not None:
            print(f"Filtering for class ID: {self.class_id} ({self.class_names[self.class_id]})")
        if self.blur_id is not None:
            print(f"Blurring class ID: {self.blur_id} ({self.class_names[self.blur_id]})")

        # Create polygon mask for region of interest
        pts = self.source_points.astype(np.int32).reshape((-1, 1, 2))
        polygon_mask = np.zeros((video_frame_height, video_frame_width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [pts], 255)

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps,
                                (video_frame_width, video_frame_height))

        frame_count = 0
        start_time = time.time()
        max_speeds = {}

        print("\nProcessing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = self.model(frame, verbose=False)
            detect = []

            for pred in results:
                for box in pred.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label = int(box.cls[0])

                    # Filter by confidence and class
                    if self.class_id is None:
                        # Only allow vehicle classes: car, motorcycle, bus, truck
                        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
                        if confidence < self.conf or label not in vehicle_classes:
                            continue
                    else:
                        if label != self.class_id or confidence < self.conf:
                            continue

                    # Check if detection is within polygon region
                    center_y = (y1 + y2) // 2
                    center_x = (x1 + x2) // 2

                    if polygon_mask[center_y, center_x] == 255:
                        detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, label])

            # Update tracker with detections
            tracks = self.tracker.update_tracks(detect, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)

                # Clamp coordinates to frame boundaries
                x1 = max(0, min(x1, video_frame_width - 1))
                y1 = max(0, min(y1, video_frame_height - 1))
                x2 = max(0, min(x2, video_frame_width - 1))
                y2 = max(0, min(y2, video_frame_height - 1))

                # Skip if bounding box is invalid
                if x2 <= x1 or y2 <= y1:
                    continue

                # Calculate center point
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2

                # Skip if center is outside polygon
                if center_y >= video_frame_height or center_x >= video_frame_width:
                    continue
                if polygon_mask[center_y, center_x] == 0:
                    continue

                # Get color for this class
                color = self.colors[class_id]
                B, G, R = map(int, color)
                text = f"{track_id} - {self.class_names[class_id]}"

                # Transform center point to bird's eye view
                center_pt = np.array([[center_x, center_y]], dtype=np.float32)
                transformed_pt = cv2.perspectiveTransform(center_pt[None, :, :], self.M)

                # Calculate speed based on previous position
                if track_id in self.prev_positions:
                    prev_position = self.prev_positions[track_id]
                    distance = self.calculate_distance(prev_position, transformed_pt[0][0])
                    speed = self.calculate_speed(distance, fps)

                    # Accumulate speed for averaging
                    if track_id in self.speed_accumulator:
                        self.speed_accumulator[track_id].append(speed)
                        # Keep only last 100 speed measurements
                        if len(self.speed_accumulator[track_id]) > 100:
                            self.speed_accumulator[track_id].pop(0)
                    else:
                        self.speed_accumulator[track_id] = [speed]

                # Update position
                self.prev_positions[track_id] = transformed_pt[0][0]

                # Draw speed if available
                if track_id in self.speed_accumulator:
                    avg_speed = sum(self.speed_accumulator[track_id]) / len(self.speed_accumulator[track_id])
                    max_speeds[track_id] = max(max_speeds.get(track_id, 0), avg_speed)

                    # Check for speed violation and capture CLEAN image BEFORE drawing bounding boxes
                    if avg_speed > self.max_speed and track_id not in self.captured_violations:
                        violation_filename = f"violation_{track_id}_{frame_count}.jpg"
                        violation_path = os.path.join(self.violations_dir, violation_filename)
                        
                        # Crop vehicle region with more padding for better quality
                        padding = 40
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(video_frame_width, x2 + padding)
                        crop_y2 = min(video_frame_height, y2 + padding)
                        
                        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        # Resize to at least 150x150 if smaller
                        if vehicle_crop.shape[0] < 150 or vehicle_crop.shape[1] < 150:
                            # Calculate new size maintaining aspect ratio
                            height, width = vehicle_crop.shape[:2]
                            scale = max(150 / width, 150 / height)
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            vehicle_crop = cv2.resize(vehicle_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        
                        # Calculate vehicle position relative to cropped image
                        rel_x1 = max(0, x1 - crop_x1)
                        rel_y1 = max(0, y1 - crop_y1)
                        rel_x2 = min(vehicle_crop.shape[1], x2 - crop_x1)
                        rel_y2 = min(vehicle_crop.shape[0], y2 - crop_y1)
                        
                        # Adjust coordinates if image was resized
                        if vehicle_crop.shape[0] >= 150 or vehicle_crop.shape[1] >= 150:
                            if 'scale' in locals():
                                rel_x1 = int(rel_x1 * scale)
                                rel_y1 = int(rel_y1 * scale)
                                rel_x2 = int(rel_x2 * scale)
                                rel_y2 = int(rel_y2 * scale)
                        
                        # Draw red bounding box around vehicle
                        cv2.rectangle(vehicle_crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 0, 255), 2)
                        
                        # Add yellow text with red background above the bounding box
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
                        
                        # Prepare text strings
                        text_line1 = f"{class_name.capitalize()}"
                        text_line2 = f"{avg_speed:.0f} km/h"
                        
                        # Calculate text dimensions
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 2
                        (text_width1, text_height1), _ = cv2.getTextSize(text_line1, font, font_scale, thickness)
                        (text_width2, text_height2), _ = cv2.getTextSize(text_line2, font, font_scale, thickness)
                        
                        # Use the wider text for background width
                        max_text_width = max(text_width1, text_width2)
                        total_text_height = text_height1 + text_height2 + 10  # 10px spacing between lines
                        
                        # Position text above the bounding box
                        text_x = rel_x1
                        text_y = max(total_text_height + 5, rel_y1 - 5)  # Above the bounding box
                        
                        # Draw red background for text
                        bg_x1 = text_x - 2
                        bg_y1 = text_y - total_text_height - 2
                        bg_x2 = text_x + max_text_width + 4
                        bg_y2 = text_y + 2
                        cv2.rectangle(vehicle_crop, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
                        
                        # Add yellow text on red background
                        cv2.putText(vehicle_crop, text_line1, (text_x, text_y - text_height2 - 5),
                                  font, font_scale, (0, 255, 255), thickness)
                        cv2.putText(vehicle_crop, text_line2, (text_x, text_y),
                                  font, font_scale, (0, 255, 255), thickness)
                        
                        cv2.imwrite(violation_path, vehicle_crop)
                        
                        if violation_path not in self.violation_images:
                            self.violation_images.append(violation_path)
                            self.captured_violations.add(track_id)

                # Draw bounding box with corner lines (AFTER capturing clean image)
                frame = self.draw_corner_rect(
                    frame, (x1, y1, x2 - x1, y2 - y1),
                    line_length=15, line_thickness=3, rect_thickness=1,
                    rect_color=(B, G, R), line_color=(R, G, B)
                )

                # Draw ID and class label
                cv2.rectangle(frame, (x1 - 1, y1 - 20),
                            (x1 + len(text) * 10, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 7),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw speed if available
                if track_id in self.speed_accumulator:
                    avg_speed = sum(self.speed_accumulator[track_id]) / len(self.speed_accumulator[track_id])
                    speed_text = f"Speed: {avg_speed:.0f} km/h"
                    speed_color = (0, 0, 255) if avg_speed > self.max_speed else (0, 255, 0)
                    cv2.rectangle(frame, (x1 - 1, y1 - 40),
                                (x1 + len(speed_text) * 10, y1 - 20), speed_color, -1)
                    cv2.putText(frame, speed_text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Apply Gaussian blur if requested
                if self.blur_id is not None and class_id == self.blur_id:
                    # Ensure blur region is within frame bounds
                    blur_x1 = max(0, x1)
                    blur_y1 = max(0, y1)
                    blur_x2 = min(frame.shape[1], x2)
                    blur_y2 = min(frame.shape[0], y2)

                    if blur_x2 > blur_x1 and blur_y2 > blur_y1:
                        frame[blur_y1:blur_y2, blur_x1:blur_x2] = cv2.GaussianBlur(
                            frame[blur_y1:blur_y2, blur_x1:blur_x2], (99, 99), 3
                        )

            # Draw polygon boundary
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Write frame to output video
            writer.write(frame)
            frame_count += 1

            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps_calc = frame_count / elapsed_time
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                      f"Processing FPS: {fps_calc:.2f}")

        # Release resources
        cap.release()
        writer.release()

        elapsed_time = time.time() - start_time

        # Print summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average processing FPS: {frame_count / elapsed_time:.2f}")
        print(f"Total vehicles tracked: {len(max_speeds)}")

        if max_speeds:
            print("\nMax speeds detected:")
            for track_id, speed in sorted(max_speeds.items(), key=lambda x: x[1], reverse=True):
                print(f"  Vehicle {track_id}: {speed:.1f} km/h")

        print(f"\nOutput saved to: {output_path}")
        print("="*60)

        # Re-encode video with ffmpeg for Chrome compatibility (H.264 + AAC)
        import subprocess
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Re-encoding speed estimation video with ffmpeg for web compatibility: {output_path}")
        
        # Create temporary file for web-compatible output
        temp_web_output = str(output_path).replace(".mp4", "_web.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            temp_web_output
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg encoding failed for speed estimation video: {result.stderr.decode()}")
            # Keep original file if ffmpeg fails
        elif os.path.exists(temp_web_output) and os.path.getsize(temp_web_output) > 1024:
            # Replace original with web-compatible version
            os.replace(temp_web_output, output_path)
            logger.info(f"Successfully re-encoded speed estimation video for web compatibility")
        else:
            logger.warning(f"FFmpeg produced empty or invalid file for speed estimation video, keeping original")

        # Read the video file and return as bytes
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        return {
            'total_frames': frame_count,
            'processing_time': elapsed_time,
            'fps': frame_count / elapsed_time if elapsed_time > 0 else 0,
            'max_speeds': max_speeds,
            'violation_images': self.violation_images,
            'video_bytes': video_bytes
        }
