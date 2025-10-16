"""
Multi-Zone Wrong-Way Detection API
Handles both single and dual-zone wrong-way vehicle detection in videos.
"""
import os
import cv2
import torch
import logging
import shutil
import random
import string
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import supervision as sv
from utilities import (
    load_model, 
    init_tracker, 
    assign_tracker_ids, 
    CLASS_ID, 
    WrongWayZone, 
    WrongWayDetector, 
    CLASS_NAMES_DICT
)

# Initialize FastAPI app
app = FastAPI(title="Multi-Zone Wrong-Way Detection API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Load the model
model = load_model("rtdetr-x.pt")

def generate_unique_filename(base_path: str, prefix: str, original_filename: str) -> str:
    """
    Generate a unique filename to prevent overwriting existing files.
    Adds a random suffix if the file already exists.
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

def create_error_response(error_type: str, message: str, details: str = None) -> dict:
    """Create a standardized error response."""
    response = {
        "success": False,
        "error_type": error_type,
        "message": message
    }
    if details:
        response["details"] = details
    return response

def process_wrong_way_video(
    input_path: str,
    output_path: str,
    zone_coords: Dict[str, Tuple[int, int]],
    direction: str
) -> Dict[str, any]:
    """
    Process video with single-zone wrong-way detection.
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
        frame_count = 0

        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                frame_count += 1
                
                # Run model inference
                model_results = model(frame, verbose=False, conf=0.3, device=device)
                
                # Handle case where model returns no results
                if model_results is None or len(model_results) == 0:
                    logger.warning("Model returned no results for frame, skipping...")
                    detector.draw_zone(frame)
                    cv2.putText(frame, f"Wrong Way Count: {wrong_way_count}", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    sink.write_frame(frame)
                    continue
                
                results = model_results[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for vehicle classes only
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                detections = assign_tracker_ids(tracker, detections)
                
                # Process detections
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None):
                        
                        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                        
                        is_wrong_way = detector.update(detections.tracker_id[i], x_center, y_center)
                        
                        if is_wrong_way and detections.tracker_id[i] not in counted_ids:
                            wrong_way_count += 1
                            counted_ids.add(detections.tracker_id[i])
                            
                            # Capture wrong-way violation image
                            violation_filename = f"wrong_way_{detections.tracker_id[i]}_{frame_count}.jpg"
                            violation_path = os.path.join(wrong_way_dir, violation_filename)
                            
                            # Crop vehicle region with padding
                            padding = 20
                            x1, y1, x2, y2 = map(int, detections.xyxy[i])
                            crop_x1 = max(0, x1 - padding)
                            crop_y1 = max(0, y1 - padding)
                            crop_x2 = min(frame.shape[1], x2 + padding)
                            crop_y2 = min(frame.shape[0], y2 + padding)
                            
                            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            cv2.imwrite(violation_path, vehicle_crop)
                            wrong_way_images.append(violation_path)
                
                # Draw annotations
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None and
                        i < len(detections.class_id) and
                        detections.class_id[i] is not None):
                        
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        class_id = detections.class_id[i]
                        
                        # Get vehicle class name
                        vehicle_class = CLASS_NAMES_DICT.get(class_id, "Vehicle").capitalize()
                        
                        # Check if vehicle is going wrong way
                        is_wrong_way = detector.is_wrong_way(detections.tracker_id[i])
                        
                        # Set colors based on vehicle class
                        if class_id == 2:  # Car
                            box_color = (0, 0, 255)  # Red
                        elif class_id == 3:  # Motorcycle  
                            box_color = (0, 255, 0)  # Green
                        elif class_id == 5:  # Bus
                            box_color = (0, 165, 255)  # Orange
                        elif class_id == 7:  # Truck
                            box_color = (255, 0, 255)  # Magenta
                        else:
                            box_color = (128, 128, 128)  # Gray
                        
                        # Override with red for wrong-way vehicles
                        if is_wrong_way:
                            box_color = (0, 0, 255)  # Red
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Draw vehicle class name
                        cv2.putText(frame, vehicle_class, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                        
                        # Add "WRONG WAY" text for violations
                        if is_wrong_way:
                            cv2.putText(frame, "WRONG WAY", (x1, y2 + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw zone and counter
                detector.draw_zone(frame)
                cv2.putText(frame, f"Wrong Way Count: {wrong_way_count}", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                sink.write_frame(frame)
        
        return {
            'wrong_way_count': wrong_way_count,
            'wrong_way_images': wrong_way_images
        }
        
    except Exception as e:
        logger.error(f"Error in process_wrong_way_video: {str(e)}")
        raise

def process_dual_zone_wrong_way_video(
    input_path: str,
    output_path: str,
    zone_coords_1: Dict[str, Tuple[int, int]],
    zone_coords_2: Dict[str, Tuple[int, int]],
    direction_1: str,
    direction_2: str
) -> Dict[str, any]:
    """
    Process video with dual-zone wrong-way detection for two-way roads.
    """
    try:
        # Create two zones
        zone_1 = WrongWayZone(
            top_left=zone_coords_1['top_left'],
            top_right=zone_coords_1['top_right'],
            bottom_left=zone_coords_1['bottom_left'],
            bottom_right=zone_coords_1['bottom_right'],
            allowed_direction=direction_1
        )
        
        zone_2 = WrongWayZone(
            top_left=zone_coords_2['top_left'],
            top_right=zone_coords_2['top_right'],
            bottom_left=zone_coords_2['bottom_left'],
            bottom_right=zone_coords_2['bottom_right'],
            allowed_direction=direction_2
        )
        
        # Create two detectors
        detector_1 = WrongWayDetector(zone_1)
        detector_2 = WrongWayDetector(zone_2)
        tracker = init_tracker()
        
        video_info = sv.VideoInfo.from_video_path(input_path)
        frame_gen = sv.get_video_frames_generator(input_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        wrong_way_count = 0
        counted_ids = set()
        wrong_way_images = []
        wrong_way_dir = "processed/wrong_side"
        os.makedirs(wrong_way_dir, exist_ok=True)
        frame_count = 0

        with sv.VideoSink(output_path, video_info) as sink:
            for frame in frame_gen:
                frame_count += 1
                
                # Run model inference
                model_results = model(frame, verbose=False, conf=0.3, device=device)
                
                # Handle case where model returns no results
                if model_results is None or len(model_results) == 0:
                    logger.warning("Model returned no results for frame, skipping...")
                    detector_1.draw_zone(frame)
                    detector_2.draw_zone(frame)
                    sink.write_frame(frame)
                    continue
                
                results = model_results[0]
                detections = sv.Detections.from_ultralytics(results)
                
                # Filter for vehicle classes only
                detections = detections[[cls in CLASS_ID for cls in detections.class_id]]
                detections = assign_tracker_ids(tracker, detections)
                
                # Process detections
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None):
                        
                        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
                        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
                        
                        # Check both zones
                        is_wrong_1 = detector_1.update(detections.tracker_id[i], x_center, y_center)
                        is_wrong_2 = detector_2.update(detections.tracker_id[i], x_center, y_center)
                        
                        # Count violation if detected in either zone
                        if (is_wrong_1 or is_wrong_2) and detections.tracker_id[i] not in counted_ids:
                            wrong_way_count += 1
                            counted_ids.add(detections.tracker_id[i])
                            
                            # Capture wrong-way violation image
                            violation_filename = f"wrong_way_{detections.tracker_id[i]}_{frame_count}.jpg"
                            violation_path = os.path.join(wrong_way_dir, violation_filename)
                            
                            # Crop vehicle region with padding
                            padding = 20
                            x1, y1, x2, y2 = map(int, detections.xyxy[i])
                            crop_x1 = max(0, x1 - padding)
                            crop_y1 = max(0, y1 - padding)
                            crop_x2 = min(frame.shape[1], x2 + padding)
                            crop_y2 = min(frame.shape[0], y2 + padding)
                            
                            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            cv2.imwrite(violation_path, vehicle_crop)
                            wrong_way_images.append(violation_path)
                
                # Draw annotations
                for i in range(len(detections.xyxy)):
                    if (hasattr(detections, 'tracker_id') and 
                        detections.tracker_id is not None and 
                        i < len(detections.tracker_id) and 
                        detections.tracker_id[i] is not None and
                        i < len(detections.class_id) and
                        detections.class_id[i] is not None):
                        
                        x1, y1, x2, y2 = map(int, detections.xyxy[i])
                        class_id = detections.class_id[i]
                        
                        # Get vehicle class name
                        vehicle_class = CLASS_NAMES_DICT.get(class_id, "Vehicle").capitalize()
                        
                        # Check if vehicle is going wrong way in either zone
                        is_wrong_way = (detector_1.is_wrong_way(detections.tracker_id[i]) or 
                                      detector_2.is_wrong_way(detections.tracker_id[i]))
                        
                        # Set colors based on vehicle class
                        if class_id == 2:  # Car
                            box_color = (0, 0, 255)  # Red
                        elif class_id == 3:  # Motorcycle  
                            box_color = (0, 255, 0)  # Green
                        elif class_id == 5:  # Bus
                            box_color = (0, 165, 255)  # Orange
                        elif class_id == 7:  # Truck
                            box_color = (255, 0, 255)  # Magenta
                        else:
                            box_color = (128, 128, 128)  # Gray
                        
                        # Override with red for wrong-way vehicles
                        if is_wrong_way:
                            box_color = (0, 0, 255)  # Red
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Draw vehicle class name
                        cv2.putText(frame, vehicle_class, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
                        
                        # Add "WRONG WAY" text for violations
                        if is_wrong_way:
                            cv2.putText(frame, "WRONG WAY", (x1, y2 + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw both zones with labels
                detector_1.draw_zone(frame)
                detector_2.draw_zone(frame)
                
                # Add zone labels
                zone_1_center = (
                    (zone_coords_1['top_left'][0] + zone_coords_1['bottom_right'][0]) // 2,
                    (zone_coords_1['top_left'][1] + zone_coords_1['bottom_right'][1]) // 2
                )
                zone_2_center = (
                    (zone_coords_2['top_left'][0] + zone_coords_2['bottom_right'][0]) // 2,
                    (zone_coords_2['top_left'][1] + zone_coords_2['bottom_right'][1]) // 2
                )
                
                cv2.putText(frame, "ZONE 1", zone_1_center, 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)  # Cyan
                cv2.putText(frame, "ZONE 2", zone_2_center, 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)  # Magenta
                
                # Add counter
                cv2.putText(frame, f"Wrong Way Count: {wrong_way_count}", (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                sink.write_frame(frame)
        
        return {
            'wrong_way_count': wrong_way_count,
            'wrong_way_images': wrong_way_images
        }
        
    except Exception as e:
        logger.error(f"Error in process_dual_zone_wrong_way_video: {str(e)}")
        raise

@app.post("/wrong_way_detection")
async def detect_wrong_way(
    file: UploadFile = File(..., description="Video file to process"),
    coordinates: str = Form(..., description="Zone coordinates (8 values): x1,y1,x2,y2,x3,y3,x4,y4"),
    direction: str = Form(..., description="Allowed direction: up, down, left, right"),
    two_way_road: bool = Form(False, description="Enable dual-zone detection for two-way roads"),
    coordinates_2: str = Form(None, description="Second zone coordinates (8 values) for two-way roads"),
    direction_2: str = Form(None, description="Allowed direction for second zone (up, down, left, right)")
):
    """
    Detect wrong-way vehicles in a video using single or dual-zone detection.
    
    For single-zone detection:
    - Provide coordinates for one zone and its direction
    - Set two_way_road=False (default)
    
    For dual-zone detection (two-way roads):
    - Set two_way_road=True
    - Provide coordinates and direction for both zones
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
        
        # Save uploaded file
        input_path = UPLOAD_DIR / file.filename
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Generate unique filename to prevent overwriting
        unique_output_path = generate_unique_filename(str(PROCESSED_DIR), "wrong_way", file.filename)
        output_path = Path(unique_output_path)
        
        # Parse and validate coordinates
        try:
            coords = [int(c.strip()) for c in coordinates.split(',')]
            if len(coords) != 8:
                raise ValueError("Coordinates must be 8 values: x1,y1,x2,y2,x3,y3,x4,y4 (top-left, top-right, bottom-right, bottom-left)")
            
            zone_coords = {
                'top_left': (coords[0], coords[1]),     # top-left
                'top_right': (coords[2], coords[3]),    # top-right
                'bottom_right': (coords[4], coords[5]), # bottom-right
                'bottom_left': (coords[6], coords[7])   # bottom-left
            }
        except Exception as e:
            raise ValueError(f"Invalid coordinates format: {str(e)}")
        
        # Validate direction
        if direction.lower() not in ["up", "down", "left", "right"]:
            raise ValueError("Direction must be: up, down, left, or right")
        
        # Process based on single or dual-zone
        if two_way_road:
            # Dual-zone mode
            if not coordinates_2 or not direction_2:
                raise ValueError("For two-way detection, both coordinates_2 and direction_2 must be provided")
                
            if direction_2.lower() not in ["up", "down", "left", "right"]:
                raise ValueError("Direction 2 must be: up, down, left, or right")
                
            # Parse second zone coordinates
            try:
                coords_2 = [int(c.strip()) for c in coordinates_2.split(',')]
                if len(coords_2) != 8:
                    raise ValueError("Second zone coordinates must be 8 values: x1,y1,x2,y2,x3,y3,x4,y4")
                    
                zone_coords_2 = {
                    'top_left': (coords_2[0], coords_2[1]),
                    'top_right': (coords_2[2], coords_2[3]),
                    'bottom_right': (coords_2[4], coords_2[5]),
                    'bottom_left': (coords_2[6], coords_2[7])
                }
            except Exception as e:
                raise ValueError(f"Invalid second zone coordinates format: {str(e)}")
            
            logger.info(f"Processing dual-zone wrong-way detection: {file.filename}")
            logger.info(f"Zone 1 direction: {direction}, Zone 2 direction: {direction_2}")
            
            # Process with dual zones
            results = process_dual_zone_wrong_way_video(
                str(input_path),
                str(output_path),
                zone_coords,
                zone_coords_2,
                direction.lower(),
                direction_2.lower()
            )
        else:
            # Single zone mode
            logger.info(f"Processing single zone wrong-way detection: {file.filename}")
            logger.info(f"Direction: {direction}")
            
            results = process_wrong_way_video(
                str(input_path), 
                str(output_path), 
                zone_coords, 
                direction.lower()
            )
        
        # Convert image paths to media URLs
        wrong_way_image_urls = []
        for img_path in results['wrong_way_images']:
            img_filename = os.path.basename(img_path)
            wrong_way_image_urls.append(f"/media/wrong_side/{img_filename}")
        
        return JSONResponse({
            "success": True,
            "wrong_way_count": results['wrong_way_count'],
            "wrong_way_images": wrong_way_image_urls,
            "output_video": f"/media/{output_path.name}",
            "two_way_detection": two_way_road
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=create_error_response("validation_error", str(e))
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response("processing_error", f"Failed to process video: {str(e)}")
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
