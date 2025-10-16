import os
import cv2
import tempfile
import numpy as np
from pathlib import Path
from fast_alpr import ALPR
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Response

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# === Load models once (kept in memory) ===
vehicle_detector = YOLO("yolo11l.pt")
vehicle_class_names = ['car', 'motorcycle', 'bus', 'truck']
allowed_class_ids = [i for i, name in vehicle_detector.names.items() if name in vehicle_class_names]

alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-s-v1-global-model"
)

def process_image(image_bytes: bytes) -> bytes:
    """Process uploaded image bytes and return annotated image bytes."""
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


app = FastAPI(title="Automatic License Plate Recognition")

@app.post("/alpr_image", summary="Upload one image and get result")
async def alpr_image(file: UploadFile = File(...)):
    """
    Upload an image (jpg/png).
    The API will process it and return the annotated image directly.
    """
    try:
        # Read uploaded image
        image_bytes = await file.read()
        print("✅ File received:", file.filename, file.content_type)

        # === Save uploaded image ===
        upload_path = UPLOAD_DIR / file.filename
        with open(upload_path, "wb") as f:
            f.write(image_bytes)

        # === Process image ===
        processed_bytes = process_image(image_bytes)

        # === Save processed output ===
        processed_path = PROCESSED_DIR / f"processed_{file.filename}"
        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        # === Return only output path ===
        return {
            "message": "Processing successful",
            "output_path": str(processed_path)
        }

    except Exception as e:
        return {"error": str(e)}
    

def process_video(video_bytes: bytes) -> bytes:
    """Process uploaded video bytes and return annotated video bytes."""
    if not video_bytes:
        raise ValueError("No video data received. Please check your form-data key name — it must be 'file'.")

    # Create a temporary input video file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_input:
        temp_input.write(video_bytes)
        temp_input_path = temp_input.name

    # Capture video
    cap = cv2.VideoCapture(temp_input_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = vehicle_detector(frame)[0]
        vehicle_boxes = [box for box in results.boxes if int(box.cls.item()) in allowed_class_ids]

        for i, box in enumerate(vehicle_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            class_name = vehicle_detector.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name.capitalize()} {i+1}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.shape[0] < 30 or vehicle_crop.shape[1] < 30:
                continue

            alpr_results = alpr.predict(vehicle_crop)

            if not alpr_results:
                cv2.putText(frame, "Licence Plate Not Clearly Visible",
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

                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 0, 255), 2)
                cv2.putText(frame, text, (abs_x1, abs_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Read output video bytes
    with open(temp_output.name, "rb") as f:
        video_data = f.read()

    return video_data

@app.post("/alpr_video", summary="Upload video result")
async def alpr_video(file: UploadFile = File(...)):
    try:
        # === Save uploaded video ===
        video_bytes = await file.read()
        upload_path = UPLOAD_DIR / file.filename
        with open(upload_path, "wb") as f:
            f.write(video_bytes)

        # === Process video ===
        processed_bytes = process_video(video_bytes)

        # === Save processed video ===
        processed_path = PROCESSED_DIR / f"processed_{file.filename}"
        with open(processed_path, "wb") as f:
            f.write(processed_bytes)

        # === Return only output path ===
        return {
            "message": "Processing successful",
            "output_path": str(processed_path)
        }

    except Exception as e:
        return {"error": str(e)}