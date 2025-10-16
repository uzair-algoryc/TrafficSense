"""
FastAPI Vehicle Counting Service
Accepts video uploads and line coordinates to count vehicles.
"""
# actually this is my model and i am loading the model already 
# """"
# FastAPI Vehicle Counting Service
# Accepts video uploads and line coordinates to count vehicles.
# """
# import subprocess
# import numpy as np
# from fastapi import FastAPI, UploadFile, File, Form, Response
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import os
# import cv2
# import supervision as sv
# import torch
# import logging
# from pathlib import Path
# from typing import Tuple
# from utilities import (
#     load_model, init_tracker, VehicleCounter, CLASS_ID, assign_tracker_ids
# )
# import tempfile
# import numpy as np
# from fast_alpr import ALPR
# from ultralytics import YOLO
# from fastapi import FastAPI, UploadFile, File, Form, Response
# from fastapi.responses import JSONResponse
# import shutil
# import cv2
# import supervision as sv
# from typing import Tuple
# from utilities import (
#     load_model, init_tracker, assign_tracker_ids, CLASS_ID, WrongWayZone, WrongWayDetector, CLASS_NAMES_DICT
# )
# # from fastapi import FastAPI, UploadFile, File, Response
# from speed_estimation import SpeedDetectionProcessor
# import uuid
# import random
# import pinggy
# import string
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import supervision as sv
import torch
import logging
from pathlib import Path
from typing import Tuple
import utilities  # Import the module, not just the objects
from utilities import (
    load_model, init_tracker, VehicleCounter, CLASS_ID, assign_tracker_ids,
    WrongWayZone, WrongWayDetector,HybridVehicleCounter
)
import tempfile
import numpy as np
from fast_alpr import ALPR
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse
import shutil
import cv2
import supervision as sv
from typing import Tuple
from speed_estimation import SpeedDetectionProcessor
import uuid
import random
import pinggy
import string

app = FastAPI(title="Traffic Monitoring APIs", version="1.0")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def generate_unique_filename(base_path: str, prefix: str, original_filename: str) -> str:
    """
    Generate unique filename to prevent overwriting existing files.
    Adds random suffix if file already exists.
    """
    # Extract file extension
    name, ext = os.path.splitext(original_filename)
    
    # Try original filename first
    output_path = os.path.join(base_path, f"{prefix}_{original_filename}")
    
    # If file doesn't exist, use original name
    if not os.path.exists(output_path):
        return output_path
    
    # Generate unique suffix if file exists
    counter = 1
    while True:
        # Try with counter first
        unique_filename = f"{prefix}_{name}_{counter}{ext}"
        output_path = os.path.join(base_path, unique_filename)
        
        if not os.path.exists(output_path):
            return output_path
            
        counter += 1
        
        # If counter gets too high, add random string
        if counter > 100:
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            unique_filename = f"{prefix}_{name}_{random_suffix}{ext}"
            output_path = os.path.join(base_path, unique_filename)
            
            if not os.path.exists(output_path):
                return output_path

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = load_model("rtdetr-x.pt")
print("loading model")
print(model.model.names)


def create_error_response(error_type: str, message: str, details: str = None):
    """Create standardized error response"""
    response = {
        "success": False,
        "error_type": error_type,
        "message": message
    }
    if details:
        response["details"] = details
    return response


@app.post("/count_vehicles")
def count_vehicles(
    file: UploadFile = File(...),
    coordinates: str = Form(...)
):
    """
    Count vehicles crossing a line in a video using hybrid trajectory analysis.
    """
    try:
        # Validate file type - must be video
        if not file.content_type or not file.content_type.startswith('video/'):
            error_msg = f"Invalid file type. Expected video file, got: {file.content_type}"
            logger.error(f"Validation error: {error_msg}")
            return JSONResponse(
                status_code=400,
                content=create_error_response("validation_error", error_msg)
            )
        
        # Validate coordinates - must be string
        if not isinstance(coordinates, str):
            error_msg = f"Invalid coordinates type. Expected string, got: {type(coordinates).__name__}"
            logger.error(f"Validation error: {error_msg}")
            return JSONResponse(
                status_code=400,
                content=create_error_response("validation_error", error_msg)
            )
        
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Generate unique filename to prevent overwriting
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "counted", file.filename)
        output_path = Path(unique_output_path)
        
        coords = [int(c.strip()) for c in coordinates.split(',')]
        if len(coords) != 4:
            raise ValueError("Coordinates must be in format: x1,y1,x2,y2")
        
        start_point = (coords[0], coords[1])
        end_point = (coords[2], coords[3])
        
        logger.info(f"Processing video: {file.filename}")
        logger.info(f"Line: {start_point} -> {end_point}")
        
        results = process_hybrid_count_video(
            str(input_path), 
            str(output_path), 
            start_point, 
            end_point
        )
        
        return JSONResponse({
            "in_count": results['in_count'],
            "out_count": results['out_count'],
            "output_video": f"/media/{output_path.name}"
        })
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


def process_hybrid_count_video(
    input_path: str, 
    output_path: str, 
    start_point: Tuple[int, int], 
    end_point: Tuple[int, int]
):
    """
    Process video with hybrid trajectory-based vehicle counting.
    """
    try:
        tracker = init_tracker()
        counter = HybridVehicleCounter(start_point, end_point)
        
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_gen = sv.get_video_frames_generator(input_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        video_info_mp4v = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            total_frames=video_info.total_frames
        )
        
        with sv.VideoSink(output_path, video_info_mp4v, codec='mp4v') as sink:
            for frame in frame_gen:
                try:
                    model_results = model(frame, verbose=False, conf=0.3, device=device)
                    
                    if model_results is None or len(model_results) == 0:
                        logger.warning("Model returned no results for frame, skipping...")
                        cv2.line(frame, start_point, end_point, (0, 255, 255), 5)
                        cv2.putText(frame, f"IN: {counter.in_count}", (60, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(frame, f"OUT: {counter.out_count}", (60, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        sink.write_frame(frame)
                        continue
                    
                    results = model_results[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    if detections is None or len(detections) == 0:
                        logger.debug("No detections found in frame")
                        cv2.line(frame, start_point, end_point, (0, 255, 255), 5)
                        cv2.putText(frame, f"IN: {counter.in_count}", (60, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(frame, f"OUT: {counter.out_count}", (60, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        sink.write_frame(frame)
                        continue
                    
                    detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                    detections = assign_tracker_ids(tracker, detections)
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 5)
                    cv2.putText(frame, f"IN: {counter.in_count}", (60, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"OUT: {counter.out_count}", (60, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    sink.write_frame(frame)
                    continue
                
                # Update trajectory counter
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None):
                        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                        counter.update(detections.tracker_id[i], x_center, y_center)
                
                # Draw trajectory trails for active vehicles (magenta trails)
                # for track_id, trajectory in counter.vehicle_trajectories.items():
                #     if len(trajectory) > 1:
                #         # Draw trajectory trail with gradient (newer points brighter)
                #         for j in range(1, len(trajectory)):
                #             pt1 = tuple(map(int, trajectory[j-1]))
                #             pt2 = tuple(map(int, trajectory[j]))
                #             # Color intensity based on recency (newer = brighter)
                #             intensity = int(255 * (j / len(trajectory)))
                #             cv2.line(frame, pt1, pt2, (intensity, 0, 255), 2)  # Magenta trail
                
                # Custom bounding box drawing with vehicle classes (color-coded)
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None):
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        
                        # Use utilities.CLASS_NAMES_DICT to get the updated value
                        if (detections.class_id is not None and 
                            i < len(detections.class_id) and 
                            utilities.CLASS_NAMES_DICT is not None):
                            class_id = detections.class_id[i]
                            if class_id is not None and class_id in utilities.CLASS_NAMES_DICT:
                                vehicle_class = utilities.CLASS_NAMES_DICT[class_id].capitalize()
                            else:
                                vehicle_class = "Vehicle"
                                class_id = None
                        else:
                            vehicle_class = "Vehicle"
                            class_id = None
                        
                        # Set colors based on vehicle class (same as trajectory API)
                        if class_id == 2:  # Car (COCO class 2)
                            box_color = (255, 0, 0)  # Blue
                        elif class_id == 3:  # Motorcycle (COCO class 3)
                            box_color = (0, 255, 0)  # Green
                        elif class_id == 5:  # Bus (COCO class 5)
                            box_color = (0, 165, 255)  # Orange
                        elif class_id == 7:  # Truck (COCO class 7)
                            box_color = (255, 0, 255)  # Magenta
                        else:
                            box_color = (128, 128, 128)  # Gray
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(frame, vehicle_class, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                
                # Draw counting line (yellow, thicker)
                cv2.line(frame, start_point, end_point, (0, 255, 255), 5)
                
                # Display counts
                cv2.putText(frame, f"IN: {counter.in_count}", (60, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"OUT: {counter.out_count}", (60, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                sink.write_frame(frame)
        
        # Re-encode video with ffmpeg for Chrome compatibility (H.264 + AAC)
        logger.info(f"Re-encoding video with ffmpeg for web compatibility: {output_path}")
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
            logger.error(f"FFmpeg encoding failed: {result.stderr.decode()}")
        elif os.path.exists(temp_web_output) and os.path.getsize(temp_web_output) > 1024:
            os.replace(temp_web_output, output_path)
            logger.info(f"Successfully re-encoded video for web compatibility")
        else:
            logger.warning(f"FFmpeg produced empty or invalid file, keeping original")
        
        return counter.get_counts()
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise Exception(str(e))

@app.post("/wrong_way_detection")
def detect_wrong_way(
    file: UploadFile = File(..., description="Video file to process"),
    coordinates: str = Form(..., description="Zone coordinates (8 values): x1,y1,x2,y2,x3,y3,x4,y4"),
    direction: str = Form(..., description="Allowed direction: up or down")
):
    """
    Detect wrong-way vehicles in a video.
    """
    try:
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            f.write(file.file.read())
        
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "wrong_way", file.filename)
        output_path = Path(unique_output_path)
        
        if not coordinates:
            raise ValueError("Coordinates are required")
        
        coords = [int(c.strip()) for c in coordinates.split(',')]
        if len(coords) != 8:
            raise ValueError("Coordinates must be 8 values: x1,y1,x2,y2,x3,y3,x4,y4 (top-left, top-right, bottom-right, bottom-left)")
        
        zone_coords = {
            'top_left': (coords[0], coords[1]),
            'top_right': (coords[2], coords[3]),
            'bottom_right': (coords[4], coords[5]),
            'bottom_left': (coords[6], coords[7])
        }
        
        if direction.lower() not in ["up", "down"]:
            raise ValueError("Direction must be: up or down")
        
        logger.info(f"Processing video: {file.filename}")
        logger.info(f"Zone coordinates: {zone_coords}")
        logger.info(f"Direction: {direction}")
        
        results = process_wrong_way_video(
            str(input_path), 
            str(output_path), 
            zone_coords, 
            direction.lower()
        )
        
        wrong_way_image_urls = []
        for img_path in results['wrong_way_images']:
            img_filename = os.path.basename(img_path)
            wrong_way_image_urls.append(f"/media/wrong_side/{img_filename}")
        
        return JSONResponse({
            "wrong_way_count": results['wrong_way_count'],
            "wrong_way_images": wrong_way_image_urls,
            "output_video": f"/media/{output_path.name}"
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


def process_wrong_way_video(
    input_path: str,
    output_path: str,
    zone_coords: dict,
    direction: str
) -> dict:
    """
    Process video with wrong-way detection.
    """
    try:
        zone = WrongWayZone(
            top_left=zone_coords['top_left'],
            top_right=zone_coords['top_right'],
            bottom_left=zone_coords['bottom_left'],
            bottom_right=zone_coords['bottom_right'],
            allowed_direction=direction
        )
        
        detector = WrongWayDetector(zone)
        tracker = init_tracker()
        
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_gen = sv.get_video_frames_generator(input_path)
        box_annotator = sv.BoxAnnotator(thickness=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        wrong_way_count = 0
        counted_ids = set()
        wrong_way_images = []
        wrong_way_dir = "processed/wrong_side"
        os.makedirs(wrong_way_dir, exist_ok=True)
        captured_wrong_way = set()
        frame_count = 0
        
        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                frame_count += 1
                results = model(frame, verbose=False, conf=0.3, device=device)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                detections = assign_tracker_ids(tracker, detections)
                
                for i in range(len(detections.xyxy)):
                    if hasattr(detections, 'tracker_id') and detections.tracker_id[i] is not None:
                        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                        
                        is_wrong = detector.update(detections.tracker_id[i], x_center, y_center)
                        if is_wrong and detections.tracker_id[i] not in counted_ids:
                            wrong_way_count += 1
                            counted_ids.add(detections.tracker_id[i])
                            
                            if detections.tracker_id[i] not in captured_wrong_way:
                                violation_filename = f"wrong_way_{detections.tracker_id[i]}_{frame_count}.jpg"
                                violation_path = os.path.join(wrong_way_dir, violation_filename)
                                
                                padding = 20
                                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                                crop_x1 = max(0, x1 - padding)
                                crop_y1 = max(0, y1 - padding)
                                crop_x2 = min(frame.shape[1], x2 + padding)
                                crop_y2 = min(frame.shape[0], y2 + padding)
                                
                                vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                cv2.imwrite(violation_path, vehicle_crop)
                                wrong_way_images.append(violation_path)
                                captured_wrong_way.add(detections.tracker_id[i])
                
                # Custom bounding box drawing with vehicle classes
                for i in range(len(detections.xyxy)):
                    if hasattr(detections, 'tracker_id') and detections.tracker_id[i] is not None:
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        class_id = detections.class_id[i]
                        
                        # Use utilities.CLASS_NAMES_DICT instead
                        vehicle_class = utilities.CLASS_NAMES_DICT[class_id].capitalize() if utilities.CLASS_NAMES_DICT else "Vehicle"
                        
                        is_wrong_way = detector.is_wrong_way(detections.tracker_id[i])
                        
                        box_color = (0, 0, 255) if is_wrong_way else (255, 0, 0)
                        text_color = (0, 0, 255) if is_wrong_way else (255, 0, 0)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame, vehicle_class, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                        
                        if is_wrong_way:
                            cv2.putText(frame, "WRONG WAY", (x1, y2 + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                detector.draw_zone(frame)
                cv2.putText(frame, f"Wrong Way Count: {wrong_way_count}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                sink.write_frame(frame)
        
        # Re-encode video with ffmpeg
        logger.info(f"Re-encoding wrong-way video with ffmpeg for web compatibility: {output_path}")
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
            logger.error(f"FFmpeg encoding failed for wrong-way video: {result.stderr.decode()}")
        elif os.path.exists(temp_web_output) and os.path.getsize(temp_web_output) > 1024:
            os.replace(temp_web_output, output_path)
            logger.info(f"Successfully re-encoded wrong-way video for web compatibility")
        else:
            logger.warning(f"FFmpeg produced empty or invalid file for wrong-way video, keeping original")
        
        return {'wrong_way_count': wrong_way_count, 'wrong_way_images': wrong_way_images}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise Exception(str(e))


@app.post("/estimate_speed")
def estimate_speed(
    file: UploadFile = File(...),
    coordinates: str = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    max_speed: float = Form(...),
    conf: float = Form(0.5)
):
    """
    Estimate vehicle speeds in a video using perspective transformation.
    """
    try:
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Generate unique filename to prevent overwriting
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "speed", file.filename)
        output_path = Path(unique_output_path)
        
        coords = [float(c.strip()) for c in coordinates.split(',')]
        if len(coords) != 8:
            raise ValueError("Coordinates must be 8 values: x1,y1,x2,y2,x3,y3,x4,y4 (top-left, top-right, bottom-right, bottom-left)")
        
        # Parse coordinates into points array
        source_points = [
            [coords[0], coords[1]],  # top-left
            [coords[2], coords[3]],  # top-right
            [coords[4], coords[5]],  # bottom-right
            [coords[6], coords[7]]   # bottom-left
        ]
        
        logger.info(f"Processing video: {file.filename}")
        logger.info(f"Source points: {source_points}")
        logger.info(f"Bird's eye view: {width}m x {height}m")
        
        results = process_speed_estimation_video(
            str(input_path), 
            str(output_path), 
            source_points, 
            width, 
            height,
            max_speed,
            conf
        )
        
        # Save the processed video to file instead of returning bytes
        with open(output_path, "wb") as f:
            f.write(results['video_bytes'])
        
        # Convert violation image paths to media URLs
        violation_image_urls = []
        for img_path in results['violation_images']:
            # Extract filename from full path
            img_filename = os.path.basename(img_path)
            violation_image_urls.append(f"/media/violations/{img_filename}")
        
        return JSONResponse({
            "violation_images": violation_image_urls,
            "output_video": f"/media/{output_path.name}"
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


def process_speed_estimation_video(
    input_path: str,
    output_path: str,
    source_points: list,
    width: float,
    height: float,
    max_speed: float,
    conf: float = 0.5
) -> dict:
    """
    Process video with speed estimation using SpeedDetectionProcessor.
    """
    try:
        processor = SpeedDetectionProcessor(
            source_points=source_points,
            frame_width=width,
            frame_height=height,
            max_speed=max_speed,
            conf=conf,
            model_path="yolov10s.pt"
        )
        
        return processor.process_video(input_path, output_path)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise Exception(str(e))


vehicle_detector = YOLO("yolo11l.pt")
vehicle_class_names = ['car', 'motorcycle', 'bus', 'truck']
allowed_class_ids = [i for i, name in vehicle_detector.names.items() if name in vehicle_class_names]

alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-s-v1-global-model"
)

def process_alpr_image(image_bytes: bytes, draw_vehicle_boxes: bool = True) -> bytes:
    """Process uploaded image bytes and return annotated image bytes."""
    try:
        if not image_bytes:
            raise ValueError("No image data received. Please check your form-data key name — it must be 'file'.")
        
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        original_annotated = image.copy()

        # === Step 1: Detect Vehicles ===
        results = vehicle_detector(image)[0]
        vehicle_boxes = [box for box in results.boxes if int(box.cls.item()) in allowed_class_ids]

        for i, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            class_name = vehicle_detector.names[class_id]

            # Draw vehicle box
            if draw_vehicle_boxes:
                cv2.rectangle(original_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(original_annotated, f"{class_name.capitalize()} {i+1}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Crop vehicle
            vehicle_crop = image[y1:y2, x1:x2]
            if vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                continue

            # === Step 2: Run ALPR ===
            alpr_results = alpr.predict(vehicle_crop)

            if not alpr_results:
                cv2.putText(original_annotated, "Licence Plate Is Not Clearly Visible",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

            for plate in alpr_results:
                box = plate.detection.bounding_box
                text = plate.ocr.text

                px1, py1, px2, py2 = box.x1, box.y1, box.x2, box.y2
                abs_x1 = x1 + int(px1)
                abs_y1 = y1 + int(py1)
                abs_x2 = x1 + int(px2)
                abs_y2 = y1 + int(py2)

                # Draw bounding boxes and OCR text
                cv2.rectangle(original_annotated, (abs_x1, abs_y1),
                              (abs_x2, abs_y2), (0, 0, 255), 2)
                cv2.putText(original_annotated, "Licence Plate", (abs_x1, abs_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = abs_x1
                text_y = abs_y1 - 40 if abs_y1 - 40 > 30 else abs_y1 + 30

                cv2.rectangle(original_annotated,
                              (text_x, text_y - text_size[1] - 10),
                              (text_x + text_size[0], text_y),
                              (0, 255, 0), -1)
                cv2.putText(original_annotated, text,
                            (text_x, text_y - 5), font, font_scale,
                            (0, 0, 0), thickness)

        # Encode back to bytes (no saving)
        success, buffer = cv2.imencode(".jpg", original_annotated)
        if not success:
            raise ValueError("Failed to encode processed image")

        return buffer.tobytes()
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


# app = FastAPI(title="Automatic License Plate Recognition")

@app.post("/alpr_image", summary="Upload one image or provide a path")
async def alpr_image(
    file: UploadFile = File(None),
    image_path: str = Form(None)
):
    """
    Upload an image (jpg/png) OR provide an image path.
    The API will process it and return the annotated image directly.
    """
    try:
        # === Step 1: Read image bytes ===
        if file:
            image_bytes = await file.read()
            input_name = file.filename
            print("✅ File received:", file.filename, file.content_type)

        elif image_path:
            BASE_MEDIA_PATH = "/home/algoryc/traffic_monitoring_demo/processed"
            if image_path.startswith("/media/"):
                full_path = os.path.join(BASE_MEDIA_PATH, image_path.lstrip("/media/"))
            else:
                full_path = image_path

            if not os.path.exists(full_path):
                raise ValueError(f"Provided image path does not exist: {full_path}")

            with open(full_path, "rb") as f:
                image_bytes = f.read()
            input_name = os.path.basename(full_path)
            print("✅ Image loaded from path:", full_path)

        else:
            raise ValueError("No image file or path provided")

        # === Step 2: Save input image ===
        upload_path = UPLOAD_DIR / input_name
        with open(upload_path, "wb") as f:
            f.write(image_bytes)

        # === Step 3: Process image ===
        processed_bytes = process_alpr_image(image_bytes)

        # === Step 4: Save processed output ===
        processed_path = PROCESSED_DIR / f"processed_{input_name}"
        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        # === Step 5: Return response ===
        return {
            "message": "Processing successful",
            "output_path": f"/media/{processed_path.name}"
        }

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


# def process_alpr_video(video_bytes: bytes) -> bytes:
#     """Process uploaded video bytes and return annotated video bytes."""
#     try:
#         if not video_bytes:
#             raise ValueError("No video data received. Please check your form-data key name — it must be 'file'.")

#         # === Create a temporary input video file ===
#         with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_input:
#             temp_input.write(video_bytes)
#             temp_input_path = temp_input.name

#         # === Capture video ===
#         cap = cv2.VideoCapture(temp_input_path)
#         if not cap.isOpened():
#             raise ValueError("Failed to open video file")

#         # === Prepare output writer ===
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
#         out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

#         frame_count = 0
#         trigger_line_y = int(height * 0.6)  # from top

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_count += 1
#             # === Draw detection line ===
#             cv2.line(frame, (0, trigger_line_y), (width, trigger_line_y), (0, 255, 255), 2)
#             cv2.putText(frame, "Detection Line", (10, trigger_line_y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 4)

#             # === Vehicle detection ===
#             results = vehicle_detector(frame)[0]
#             vehicle_boxes = [box for box in results.boxes if int(box.cls.item()) in allowed_class_ids]

#             for i, box in enumerate(vehicle_boxes):
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 class_id = int(box.cls.item())
#                 class_name = vehicle_detector.names[class_id]

#                 # Calculate center of vehicle bounding box
#                 cx = int((x1 + x2) / 2)
#                 cy = int((y1 + y2) / 2)

#                 # Draw vehicle box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
#                 # cv2.putText(frame, f"{class_name.capitalize()} {i+1}", (x1, y1 - 10),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#                 cv2.putText(frame, class_name.capitalize(), (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 4)
                

#                 # === Trigger ALPR only when vehicle crosses the 30% height line ===
#                 if cy >= trigger_line_y:
#                     vehicle_crop = frame[y1:y2, x1:x2]
#                     if vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
#                         continue

#                     alpr_results = alpr.predict(vehicle_crop)

#                     if not alpr_results:
#                         cv2.putText(frame, "Licence Plate Not Clearly Visible",
#                                     (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#                                     (0, 0, 255), 3)
#                         continue

#                     for plate in alpr_results:
#                         box = plate.detection.bounding_box
#                         text = plate.ocr.text

#                         px1, py1, px2, py2 = box.x1, box.y1, box.x2, box.y2
#                         abs_x1 = x1 + int(px1)
#                         abs_y1 = y1 + int(py1)
#                         abs_x2 = x1 + int(px2)
#                         abs_y2 = y1 + int(py2)

#                         cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
#                         cv2.putText(frame, text, (abs_x1, abs_y1 - 10),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

#             out.write(frame)

#         cap.release()
#         out.release()

#         # === Read output video bytes ===
#         with open(temp_output.name, "rb") as f:
#             video_data = f.read()

#         return video_data
#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         return JSONResponse(
#             status_code=400,
#             content=create_error_response("validation_error", str(e))
#         )
#     except Exception as e:
#         logger.error(f"Error processing video: {str(e)}")
#         return JSONResponse(
#             status_code=500,
#             content=create_error_response("processing_error", str(e))
#         )


def process_alpr_video(video_bytes: bytes) -> bytes:
    """Process uploaded video bytes and return annotated video bytes."""
    try:
        if not video_bytes:
            raise ValueError("No video data received. Please check your form-data key name — it must be 'file'.")

        # === Create a temporary input video file ===
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_input:
            temp_input.write(video_bytes)
            temp_input_path = temp_input.name

        # === Capture video ===
        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # === Prepare output writer ===
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 25.0  # fallback if FPS not readable

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_output_path = temp_output.name
        temp_output.close()

        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        frame_count = 0
        trigger_line_y = int(height * 0.55)  # from top

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # === Draw detection line ===
            cv2.line(frame, (0, trigger_line_y), (width, trigger_line_y), (0, 255, 255), 2)
            cv2.putText(frame, "Detection Line", (10, trigger_line_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # === Vehicle detection ===
            results = vehicle_detector(frame)[0]
            vehicle_boxes = [box for box in results.boxes if int(box.cls.item()) in allowed_class_ids]

            for i, box in enumerate(vehicle_boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls.item())
                class_name = vehicle_detector.names[class_id]

                # Calculate center of bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, class_name.capitalize(), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # === Trigger ALPR when vehicle crosses line ===
                if cy >= trigger_line_y:
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                        continue

                    alpr_results = alpr.predict(vehicle_crop)
                    if not alpr_results:
                        cv2.putText(frame, "Licence Plate Not Clearly Visible",
                                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        continue

                    for plate in alpr_results:
                        box = plate.detection.bounding_box
                        text = plate.ocr.text

                        px1, py1, px2, py2 = box.x1, box.y1, box.x2, box.y2
                        abs_x1 = x1 + int(px1)
                        abs_y1 = y1 + int(py1)
                        abs_x2 = x1 + int(px2)
                        abs_y2 = y1 + int(py2)

                        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                        cv2.putText(frame, text, (abs_x1, abs_y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # === Step 2: Re-encode for Chrome compatibility (H.264 + AAC) ===
        chrome_safe_output = temp_output_path.replace(".mp4", "_web.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-i", temp_output_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            chrome_safe_output
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not os.path.exists(chrome_safe_output) or os.path.getsize(chrome_safe_output) < 1024:
            raise ValueError("FFmpeg re-encoding failed or produced empty file")

        # === Read output video bytes ===
        with open(chrome_safe_output, "rb") as f:
            video_data = f.read()

        return video_data

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


@app.post("/alpr_video", summary="Upload video result")
async def alpr_video(file: UploadFile = File(...)):
    try:
        # === Save uploaded video ===
        video_bytes = await file.read()
        upload_path = UPLOAD_DIR / file.filename
        with open(upload_path, "wb") as f:
            f.write(video_bytes)

        # === Process video ===
        processed_bytes = process_alpr_video(video_bytes)

        # === Save processed video ===
        processed_path = PROCESSED_DIR / f"processed_{file.filename}"
        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        # === Return only output path ===
        return {
            "message": "Processing successful",
            "output_path": f"/media/{processed_path.name}"
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )

from fastapi.responses import FileResponse
@app.get("/play_video/{video_name}")
def play_video(video_name: str):
    """
    Serve a video file by name for browser playback.
    
    Args:
        video_name: Name of the video file (with or without extension)
    
    Returns:
        Video file that can be played directly in browser/Swagger UI
    """
    try:
        # Add .mp4 extension if not provided
        if not video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_name += '.mp4'
        
        # Check in processed directory first, then uploads
        video_path = None
        
        # Look in processed directory
        processed_path = PROCESSED_DIR / video_name
        if processed_path.exists():
            video_path = processed_path
        else:
            # Look in uploads directory
            upload_path = UPLOAD_DIR / video_name
            if upload_path.exists():
                video_path = upload_path
        
        if video_path is None:
            logger.error(f"Video not found: {video_name}")
            return JSONResponse(
                status_code=404,
                content=create_error_response("validation_error", f"Video '{video_name}' not found in processed or uploads directory")
            )
        
        # Return video file with proper headers for browser playback
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f"inline; filename={video_name}"
            }
        )
    
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


@app.get("/list_videos")
def list_videos():
    """
    List all available video files in processed and uploads directories.
    
    Returns:
        JSON response with lists of available videos
    """
    try:
        processed_videos = []
        upload_videos = []
        
        # Get videos from processed directory
        if PROCESSED_DIR.exists():
            for file_path in PROCESSED_DIR.glob("*.mp4"):
                processed_videos.append(file_path.name)
            for file_path in PROCESSED_DIR.glob("*.avi"):
                upload_videos.append(file_path.name)
        
        # Get videos from uploads directory
        if UPLOAD_DIR.exists():
            for file_path in UPLOAD_DIR.glob("*.mp4"):
                upload_videos.append(file_path.name)
            for file_path in UPLOAD_DIR.glob("*.avi"):
                upload_videos.append(file_path.name)
        
        return JSONResponse({
            "processed_videos": sorted(processed_videos),
            "upload_videos": sorted(upload_videos),
            "total_videos": len(processed_videos) + len(upload_videos),
            "usage": "Use /play_video/{video_name} to play any video from the lists above"
        })
    
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


# ============================================================================
# TRAJECTORY-BASED VEHICLE COUNTING SYSTEM
# ============================================================================

class TrajectoryVehicleCounter:
    """Trajectory-based vehicle counter using movement analysis"""
    
    def __init__(self, start_point: tuple, end_point: tuple):
        """
        Initialize trajectory-based counter
        
        Args:
            start_point: Line start coordinates (x, y)
            end_point: Line end coordinates (x, y)
        """
        self.start = np.array(start_point, dtype=np.float64)
        self.end = np.array(end_point, dtype=np.float64)
        self.road_type = "vertical"  # Default to vertical road analysis
        
        # Line equation: Ax + By + C = 0
        self.line_vec = self.end - self.start
        self.A = self.line_vec[1]  # dy
        self.B = -self.line_vec[0]  # -dx
        self.C = self.line_vec[0] * self.start[1] - self.line_vec[1] * self.start[0]
        
        # Trajectory parameters (optimized for faster response and better accuracy)
        self.trajectory_buffer_size = 15  # Reduced for faster response (0.5 seconds at 30 FPS)
        self.min_points_for_counting = 7  # Reduced for quicker decisions (0.23 seconds at 30 FPS)
        self.direction_threshold = 4.0  # Increased to filter noise better
        
        # Vehicle tracking data
        self.vehicle_trajectories = {}  # track_id -> list of (x, y) points
        self.vehicle_crossed = set()  # track_id of vehicles already counted
        self.vehicle_last_side = {}  # track_id -> side of line (1 or -1)
        
        # Counters
        self.incoming_count = 0
        self.outgoing_count = 0
    
    def _get_line_side(self, point):
        """Get which side of line the point is on using line equation"""
        x, y = point
        value = self.A * x + self.B * y + self.C
        return 1 if value > 0 else -1
    
    def _line_crossed(self, prev_point, curr_point):
        """Check if vehicle crossed the line between two points"""
        prev_side = self._get_line_side(prev_point)
        curr_side = self._get_line_side(curr_point)
        return prev_side != curr_side
    
    def _calculate_movement_direction(self, trajectory):
        """Calculate average movement direction from trajectory"""
        if len(trajectory) < 2:
            return None
        
        # Calculate movement vectors between consecutive points
        movements = []
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            movements.append((dx, dy))
        
        # Calculate average movement
        avg_dx = sum(mov[0] for mov in movements) / len(movements)
        avg_dy = sum(mov[1] for mov in movements) / len(movements)
        
        return avg_dx, avg_dy
    
    def _determine_direction(self, avg_dx, avg_dy):
        """Determine if movement is incoming or outgoing based on road type"""
        # Always use vertical analysis (Y-axis movement)
        print(f"Movement analysis: avg_dx={avg_dx:.2f}, avg_dy={avg_dy:.2f}, threshold={self.direction_threshold}")
        
        if abs(avg_dy) > self.direction_threshold:
            if avg_dy > 0:
                print("Direction: INCOMING (moving down)")
                return "incoming"  # Moving down (positive Y)
            else:
                print("Direction: OUTGOING (moving up)")
                return "outgoing"  # Moving up (negative Y)
        
        print("Direction: UNCLEAR (movement too small)")
        return None  # Movement too small or unclear
    
    def update(self, tracker_id: int, x_center: float, y_center: float):
        """Update vehicle trajectory and check for crossing"""
        if tracker_id is None:
            return
        
        # Skip if already counted
        if tracker_id in self.vehicle_crossed:
            return
        
        current_point = np.array([x_center, y_center], dtype=np.float64)
        
        # Initialize trajectory for new vehicle
        if tracker_id not in self.vehicle_trajectories:
            self.vehicle_trajectories[tracker_id] = [current_point]
            self.vehicle_last_side[tracker_id] = self._get_line_side(current_point)
            return
        
        # Add current point to trajectory
        trajectory = self.vehicle_trajectories[tracker_id]
        trajectory.append(current_point)
        
        # Maintain buffer size
        if len(trajectory) > self.trajectory_buffer_size:
            trajectory.pop(0)
        
        # Check for line crossing
        if len(trajectory) >= 2:
            prev_point = trajectory[-2]
            curr_point = trajectory[-1]
            
            if self._line_crossed(prev_point, curr_point):
                # Line crossed! Check if we have enough trajectory data
                if len(trajectory) >= self.min_points_for_counting:
                    # Calculate movement direction
                    movement = self._calculate_movement_direction(trajectory)
                    if movement:
                        avg_dx, avg_dy = movement
                        direction = self._determine_direction(avg_dx, avg_dy)
                        
                        if direction == "incoming":
                            self.incoming_count += 1
                            self.vehicle_crossed.add(tracker_id)
                            print(f"✓ COUNTED: Vehicle {tracker_id} as INCOMING. Total incoming: {self.incoming_count}")
                        elif direction == "outgoing":
                            self.outgoing_count += 1
                            self.vehicle_crossed.add(tracker_id)
                            print(f"✓ COUNTED: Vehicle {tracker_id} as OUTGOING. Total outgoing: {self.outgoing_count}")
                        else:
                            print(f"✗ NOT COUNTED: Vehicle {tracker_id} - unclear direction")
    
    def get_counts(self):
        """Get current counts"""
        return {
            'in_count': self.incoming_count,
            'out_count': self.outgoing_count
        }
    
    def reset(self):
        """Reset all counters and tracking data"""
        self.vehicle_trajectories = {}
        self.vehicle_crossed = set()
        self.vehicle_last_side = {}
        self.incoming_count = 0
        self.outgoing_count = 0


def process_trajectory_counting_video(
    input_path: str,
    output_path: str, 
    start_point: Tuple[int, int],
    end_point: Tuple[int, int]
):
    """
    Process video with trajectory-based vehicle counting.
    """
    try:
        tracker = init_tracker()
        counter = TrajectoryVehicleCounter(start_point, end_point)
        
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_gen = sv.get_video_frames_generator(input_path)
        box_annotator = sv.BoxAnnotator(thickness=2)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                results = model(frame, verbose=False, conf=0.5, device=device)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                detections = assign_tracker_ids(tracker, detections)
                
                # Update trajectory counter
                for i in range(len(detections.xyxy)):
                    if hasattr(detections, 'tracker_id') and detections.tracker_id[i] is not None:
                        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                        counter.update(detections.tracker_id[i], x_center, y_center)
                
                # Draw annotations
                frame = box_annotator.annotate(scene=frame, detections=detections)
                
                # Draw counting line (thicker for trajectory system)
                cv2.line(frame, start_point, end_point, (0, 255, 255), 5)  # Yellow line
                
                # Draw trajectory trails for active vehicles
                for track_id, trajectory in counter.vehicle_trajectories.items():
                    if len(trajectory) > 1:
                        # Draw trajectory trail with gradient (newer points brighter)
                        for j in range(1, len(trajectory)):
                            pt1 = tuple(map(int, trajectory[j-1]))
                            pt2 = tuple(map(int, trajectory[j]))
                            # Color intensity based on recency (newer = brighter)
                            intensity = int(255 * (j / len(trajectory)))
                            cv2.line(frame, pt1, pt2, (intensity, 0, 255), 3)  # Magenta trail
                        
                        # Draw trajectory length info
                        if len(trajectory) >= counter.min_points_for_counting:
                            start_pt = tuple(map(int, trajectory[0]))
                            cv2.putText(frame, f"T{track_id}:{len(trajectory)}", start_pt, 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display counts with trajectory labels
                cv2.putText(frame, f"INCOMING: {counter.incoming_count}", (60, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"OUTGOING: {counter.outgoing_count}", (60, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "TRAJECTORY MODE", (60, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                sink.write_frame(frame)
        
        return counter.get_counts()
    except Exception as e:
        logger.error(f"Error processing trajectory video: {str(e)}")
        raise Exception(str(e))


@app.post("/count_vehicles_trajectory")
def count_vehicles_trajectory(
    file: UploadFile = File(...),
    coordinates: str = Form(...)
):
    """
    Count vehicles using trajectory-based analysis.
    
    Args:
        file: Video file to process
        coordinates: Line coordinates as "x1,y1,x2,y2"
    
    Returns:
        JSON with in_count, out_count, and output_video path
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            error_msg = f"Invalid file type. Expected video file, got: {file.content_type}"
            logger.error(f"Validation error: {error_msg}")
            return JSONResponse(
                status_code=400,
                content=create_error_response("validation_error", error_msg)
            )
        
        # Validate and parse coordinates
        try:
            coords = [int(float(c.strip())) for c in coordinates.split(',')]
            if len(coords) != 4:
                raise ValueError("Coordinates must be exactly 4 values")
            start_point = (coords[0], coords[1])
            end_point = (coords[2], coords[3])
        except (ValueError, IndexError) as e:
            error_msg = f"Invalid coordinates format. Expected 'x1,y1,x2,y2', got: {coordinates}"
            logger.error(f"Validation error: {error_msg}")
            return JSONResponse(
                status_code=400,
                content=create_error_response("validation_error", error_msg)
            )
        
        
        # Save uploaded file
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Process video with trajectory counting
        # Generate unique filename to prevent overwriting
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "trajectory", file.filename)
        output_path = Path(unique_output_path)
        results = process_trajectory_counting_video(
            str(input_path),
            str(output_path), 
            start_point,
            end_point
        )
        
        return JSONResponse({
            "in_count": results['in_count'],
            "out_count": results['out_count'],
            "output_video": f"/media/{output_path.name}"
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing trajectory video: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", str(e))
        )


# =============================================================================
# DEVELOPMENT MEDIA SERVER (NGINX-STYLE)
# =============================================================================

from fastapi.staticfiles import StaticFiles

# Mount static files for development (serves files directly like nginx)
if os.path.exists("processed"):
    app.mount("/media", StaticFiles(directory="processed"), name="media")
    logger.info("📁 Media server enabled at /media/ (Development only)")
    logger.info("💡 Usage: GET /media/filename.mp4 to serve files directly")
else:
    logger.warning("⚠️  'processed' directory not found - media server disabled")


# PINGGY
tunnel = pinggy.start_tunnel(forwardto="localhost:8001")
print(f"Tunnel started. Urls: {tunnel.urls}")