"""
FastAPI Vehicle Counting Service
Accepts video uploads and line coordinates to count vehicles.
"""
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
from utilities import (
    load_model, init_tracker, VehicleCounter, CLASS_ID, assign_tracker_ids,
    WrongWayZone, WrongWayDetector, CLASS_NAMES_DICT
)
import tempfile
import numpy as np
from fast_alpr import ALPR
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, Response, HTTPException
from fastapi.responses import JSONResponse
import shutil
import cv2
import supervision as sv
from typing import Tuple, Dict, List, Optional
import random
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
    name, ext = os.path.splitext(original_filename)
    output_path = os.path.join(base_path, f"{prefix}_{original_filename}")
    
    if not os.path.exists(output_path):
        return output_path
    
    counter = 1
    while True:
        unique_filename = f"{prefix}_{name}_{counter}{ext}"
        output_path = os.path.join(base_path, unique_filename)
        
        if not os.path.exists(output_path):
            return output_path
            
        counter += 1
        
        if counter > 100:
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            unique_filename = f"{prefix}_{name}_{random_suffix}{ext}"
            output_path = os.path.join(base_path, unique_filename)
            
            if not os.path.exists(output_path):
                return output_path

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = load_model("rtdetr-x.pt")

def create_error_response(error_type: str, message: str, details: str = None) -> dict:
    """Create standardized error response."""
    response = {
        "success": False,
        "error_type": error_type,
        "message": message
    }
    if details:
        response["details"] = details
    return response

@app.post("/count_vehicles")
async def count_vehicles(
    file: UploadFile = File(...),
    coordinates: str = Form(...)
):
    """
    Count vehicles crossing a line in a video.
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
        
        # Process video
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "counted", file.filename)
        output_path = Path(unique_output_path)
        
        results = process_vehicle_count_video(
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

def process_vehicle_count_video(
    input_path: str,
    output_path: str,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int]
) -> dict:
    """
    Process video with vehicle detection, tracking, and counting.
    """
    try:
        # Initialize tracker and counter
        tracker = init_tracker()
        counter = VehicleCounter(start_point, end_point)
        
        # Get video info
        video_info = sv.VideoInfo.from_video_path(input_path)
        
        # Initialize video writer
        box_annotator = sv.BoxAnnotator(thickness=2)
        
        # Process video
        frame_gen = sv.get_video_frames_generator(input_path)
        
        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                # Run model inference
                results = model(frame, verbose=False, conf=0.3, device='cuda' if torch.cuda.is_available() else 'cpu')[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for vehicle classes only
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                
                # Update tracker
                detections = assign_tracker_ids(tracker, detections)
                
                # Update counter
                counter.update(detections)
                
                # Draw annotations
                frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections,
                    labels=[f"#{tracker_id}" for tracker_id in detections.tracker_id]
                )
                
                # Draw counting line
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                
                # Draw counts
                cv2.putText(frame, f"In: {counter.in_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Out: {counter.out_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                sink.write_frame(frame)
        
        return {
            'in_count': counter.in_count,
            'out_count': counter.out_count
        }
        
    except Exception as e:
        logger.error(f"Error in process_vehicle_count_video: {str(e)}")
        raise

@app.post("/detect_wrong_way")
async def detect_wrong_way(
    file: UploadFile = File(..., description="Video file to process"),
    coordinates: str = Form(..., description="Zone coordinates (8 values): x1,y1,x2,y2,x3,y3,x4,y4"),
    direction: str = Form(..., description="Allowed direction: up, down, left, right")
):
    """
    Detect wrong-way vehicles in a video using single-zone detection.
    
    Args:
        file: Video file to process
        coordinates: Zone coordinates as "x1,y1,x2,y2,x3,y3,x4,y4" (top-left, top-right, bottom-right, bottom-left)
        direction: Allowed traffic direction ("up", "down", "left", "right")
    
    Returns:
        JSON with wrong_way_count, wrong_way_images, and output_video
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
            if len(coords) != 8:
                raise ValueError("Coordinates must be exactly 8 values (x1,y1,x2,y2,x3,y3,x4,y4)")
            
            zone_coords = {
                'top_left': (coords[0], coords[1]),
                'top_right': (coords[2], coords[3]),
                'bottom_right': (coords[4], coords[5]),
                'bottom_left': (coords[6], coords[7])
            }
            
            # Validate direction
            if direction.lower() not in ["up", "down", "left", "right"]:
                raise ValueError("Direction must be one of: up, down, left, right")
                
        except (ValueError, IndexError) as e:
            error_msg = f"Invalid coordinates format. Expected 'x1,y1,x2,y2,x3,y3,x4,y4', got: {coordinates}"
            logger.error(f"Validation error: {error_msg}")
            return JSONResponse(
                status_code=400,
                content=create_error_response("validation_error", str(e))
            )
        
        # Save uploaded file
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Process video with wrong-way detection
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "wrong_way", file.filename)
        output_path = Path(unique_output_path)
        
        results = process_wrong_way_video(
            str(input_path),
            str(output_path),
            zone_coords,
            direction.lower()
        )
        
        return JSONResponse({
            "wrong_way_count": results['wrong_way_count'],
            "wrong_way_images": results['wrong_way_images'],
            "output_video": f"/media/{output_path.name}"
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing wrong-way detection: {str(e)}")
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
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        zone_coords: Dictionary with zone coordinates
        direction: Allowed direction of travel
        
    Returns:
        Dictionary with wrong_way_count and wrong_way_images
    """
    try:
        # Initialize zone and detector
        zone = WrongWayZone(
            top_left=zone_coords['top_left'],
            top_right=zone_coords['top_right'],
            bottom_left=zone_coords['bottom_left'],
            bottom_right=zone_coords['bottom_right'],
            allowed_direction=direction
        )
        
        detector = WrongWayDetector(zone)
        tracker = init_tracker()
        
        # Get video info
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_gen = sv.get_video_frames_generator(input_path)
        
        # Initialize annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        
        # For storing wrong-way images
        wrong_way_images = []
        wrong_way_dir = "processed/wrong_way"
        os.makedirs(wrong_way_dir, exist_ok=True)
        
        frame_count = 0
        wrong_way_count = 0
        counted_ids = set()
        
        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                frame_count += 1
                
                # Run model inference
                results = model(frame, verbose=False, conf=0.3, device='cuda' if torch.cuda.is_available() else 'cpu')[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for vehicle classes only
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                
                # Update tracker
                detections = assign_tracker_ids(tracker, detections)
                
                # Check for wrong-way vehicles
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    for i, tracker_id in enumerate(detections.tracker_id):
                        if tracker_id is not None:
                            # Get center of bounding box
                            x1, y1, x2, y2 = map(int, detections.xyxy[i])
                            x_center = (x1 + x2) // 2
                            y_center = (y1 + y2) // 2
                            
                            # Check if vehicle is going the wrong way
                            is_wrong_way = detector.update(tracker_id, x_center, y_center)
                            
                            # If wrong way and not already counted
                            if is_wrong_way and tracker_id not in counted_ids:
                                wrong_way_count += 1
                                counted_ids.add(tracker_id)
                                
                                # Save image of wrong-way vehicle
                                img_path = os.path.join(wrong_way_dir, f"wrong_way_{tracker_id}_{frame_count}.jpg")
                                cv2.imwrite(img_path, frame[y1:y2, x1:x2])
                                wrong_way_images.append(img_path)
                
                # Draw annotations
                for i, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get class name
                    class_name = CLASS_NAMES_DICT.get(class_id, "Vehicle")
                    
                    # Check if this is a wrong-way vehicle
                    is_wrong_way = False
                    if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id):
                        tracker_id = detections.tracker_id[i]
                        if tracker_id is not None:
                            is_wrong_way = detector.is_wrong_way(tracker_id)
                    
                    # Draw bounding box
                    color = (0, 0, 255) if is_wrong_way else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw class label
                    label = f"{class_name}"
                    if hasattr(detections, 'tracker_id') and i < len(detections.tracker_id):
                        tracker_id = detections.tracker_id[i]
                        if tracker_id is not None:
                            label = f"{tracker_id}: {class_name}"
                    
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add wrong way indicator
                    if is_wrong_way:
                        cv2.putText(frame, "WRONG WAY", (x1, y1 - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw zone
                zone_points = np.array([
                    zone.top_left, zone.top_right, 
                    zone.bottom_right, zone.bottom_left
                ], dtype=np.int32)
                cv2.polylines(frame, [zone_points], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(frame, f"ZONE: {direction.upper()} ONLY", 
                          (zone.top_left[0], zone.top_left[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw counter
                cv2.putText(frame, f"Wrong Way Count: {wrong_way_count}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                sink.write_frame(frame)
        
        return {
            'wrong_way_count': wrong_way_count,
            'wrong_way_images': wrong_way_images
        }
        
    except Exception as e:
        logger.error(f"Error in process_wrong_way_video: {str(e)}")
        raise

# =============================================================================
# DEVELOPMENT MEDIA SERVER (NGINX-STYLE)
# =============================================================================
from fastapi.staticfiles import StaticFiles
import os

# Mount static files for development (serves files directly like nginx)
if os.path.exists("processed"):
    app.mount("/media", StaticFiles(directory="processed"), name="media")
    logger.info("📁 Media server enabled at /media/ (Development only)")
    logger.info("💡 Usage: GET /media/filename.mp4 to serve files directly")
else:
    logger.warning("⚠️  'processed' directory not found - media server disabled")
