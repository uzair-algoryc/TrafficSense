import numpy as np
from ultralytics import RTDETR
import supervision as sv
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch
import cv2
import torch

CLASS_ID = [2, 3, 5, 7]
CLASS_NAMES_DICT = None


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def load_model(model_path: str = "rtdetr-x.pt"):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR(model_path)
    model.fuse()
    if device == 'cuda':
        model.to(device)
    global CLASS_NAMES_DICT
    CLASS_NAMES_DICT = model.model.names
    return model


def init_tracker():
    return sv.ByteTrack()


def detections2boxes(detections):
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks):
    return np.array([track.tlbr for track in tracks], dtype=float)


def match_detections_with_tracks(detections, tracks):
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))
    tracks_boxes = tracks2boxes(tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    return tracker_ids


def assign_tracker_ids(tracker, detections):
    if tracker is None:
        detections.tracker_id = [None] * len(detections.xyxy)
        return detections
    
    if hasattr(tracker, "update_with_detections"):
        try:
            updated = tracker.update_with_detections(detections)
            if hasattr(updated, "tracker_id"):
                return updated
        except Exception:
            pass
    
    if hasattr(tracker, "update"):
        try:
            tracks = tracker.update(detections)
            tracker_ids = match_detections_with_tracks(detections, tracks)
            try:
                detections.tracker_id = tracker_ids
            except Exception:
                setattr(detections, "tracker_id", tracker_ids)
            return detections
        except Exception:
            pass
    
    detections.tracker_id = [None] * len(detections.xyxy)
    return detections


class VehicleCounter:
    def __init__(self, start_point: tuple, end_point: tuple, mode: str = "both"):
        """
        mode: "both" (2-way traffic), "in" (only IN direction), "out" (only OUT direction)
        """
        self.start = np.array(start_point, dtype=np.float64)
        self.end = np.array(end_point, dtype=np.float64)
        
        self.line_vec = self.end - self.start
        self.line_length = np.linalg.norm(self.line_vec)
        self.line_unit = self.line_vec / self.line_length
        
        # Create perpendicular vector for thick line detection
        self.perpendicular = np.array([-self.line_vec[1], self.line_vec[0]], dtype=np.float64)
        self.perpendicular = self.perpendicular / np.linalg.norm(self.perpendicular)
        self.line_thickness = 10  # 10-pixel thick line
        
        # Store vehicle tracking data
        self.position_history = {}
        self.vehicle_last_side = {}  # Track which side of line vehicle was last on
        self.counted_vehicles = set()  # PERMANENT set - once counted, never count again
        self.in_count = 0
        self.out_count = 0
        
        self.mode = mode.lower()

    def _distance_to_line(self, point):
        """Calculate perpendicular distance from point to line"""
        point_vec = point - self.start
        # Project point onto line direction
        projection = np.dot(point_vec, self.line_unit) * self.line_unit
        # Get perpendicular distance
        perpendicular_vec = point_vec - projection
        return np.linalg.norm(perpendicular_vec)

    def _get_side_of_line(self, point):
        """Get which side of line the point is on: 1 or -1"""
        cross = np.cross(self.line_vec, point - self.start)
        return 1 if cross > 0 else -1

    def _point_on_thick_line(self, point):
        """Check if point is ON the 10-pixel thick line"""
        return self._distance_to_line(point) <= (self.line_thickness / 2)

    def _get_movement_direction(self, tracker_id, current_point):
        """Calculate movement direction for curved road handling"""
        if tracker_id not in self.position_history or len(self.position_history[tracker_id]) < 2:
            return None
        
        # Get last few positions to determine movement direction
        positions = self.position_history[tracker_id][-3:]  # Last 3 positions
        if len(positions) < 2:
            return None
        
        # Calculate average movement vector
        movement_vectors = []
        for i in range(1, len(positions)):
            movement_vectors.append(positions[i] - positions[i-1])
        
        if not movement_vectors:
            return None
        
        # Average movement direction
        avg_movement = np.mean(movement_vectors, axis=0)
        movement_magnitude = np.linalg.norm(avg_movement)
        
        if movement_magnitude < 1.0:  # Too small movement
            return None
        
        return avg_movement / movement_magnitude

    def update(self, tracker_id: int, x_center: float, y_center: float):
        if tracker_id is None:
            return
        
        current_point = np.array([x_center, y_center], dtype=np.float64)
        
        # PERMANENT COUNTING PREVENTION - once counted, never count again
        if tracker_id in self.counted_vehicles:
            return
        
        # Initialize tracking for new vehicle
        if tracker_id not in self.position_history:
            self.position_history[tracker_id] = [current_point]
            # Determine initial side (only if NOT on the thick line)
            if not self._point_on_thick_line(current_point):
                self.vehicle_last_side[tracker_id] = self._get_side_of_line(current_point)
            else:
                self.vehicle_last_side[tracker_id] = None
            return
        
        # Update position history
        self.position_history[tracker_id].append(current_point)
        if len(self.position_history[tracker_id]) > 3:
            self.position_history[tracker_id].pop(0)
        
        # Check if vehicle is currently ON the thick line
        on_thick_line = self._point_on_thick_line(current_point)
        
        if not on_thick_line:
            # Vehicle is clearly on one side of the line
            current_side = self._get_side_of_line(current_point)
            last_side = self.vehicle_last_side.get(tracker_id)
            
            # Check for crossing: was on one side, now on opposite side
            if last_side is not None and last_side != current_side:
                # Vehicle crossed the thick line!
                if last_side == 1 and current_side == -1:
                    direction = "out"
                elif last_side == -1 and current_side == 1:
                    direction = "in"
                else:
                    direction = None
                
                # Count based on direction and mode
                if direction == "in" and self.mode in ["both", "in"]:
                    self.in_count += 1
                    self.counted_vehicles.add(tracker_id)  # PERMANENT - never count again
                elif direction == "out" and self.mode in ["both", "out"]:
                    self.out_count += 1
                    self.counted_vehicles.add(tracker_id)  # PERMANENT - never count again
            
            # Update last known side
            self.vehicle_last_side[tracker_id] = current_side

    def get_counts(self):
        return {'in_count': self.in_count, 'out_count': self.out_count}

    def reset(self):
        self.position_history = {}
        self.vehicle_last_side = {}
        self.counted_vehicles = set()
        self.in_count = 0
        self.out_count = 0


@dataclass
class WrongWayZone:
    """Represents a traffic zone with allowed direction."""
    top_left: tuple
    top_right: tuple
    bottom_left: tuple
    bottom_right: tuple
    allowed_direction: str  # "up", "down"


class WrongWayDetector:
    """Detects vehicles moving in wrong direction within a zone."""
    
    def __init__(self, zone: WrongWayZone):
        """
        Initialize wrong way detector.
        
        Args:
            zone: WrongWayZone object with coordinates and allowed direction
        """
        self.zone = zone
        self.polygon = np.array([
            zone.top_left,
            zone.top_right,
            zone.bottom_right,
            zone.bottom_left
        ], dtype=np.int32)
        
        self.vehicle_history = {}
        self.wrong_way_vehicles = set()
    
    def _point_in_zone(self, point: tuple) -> bool:
        """Check if point is inside zone polygon."""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0
    
    def _get_direction(self, prev_point: tuple, curr_point: tuple) -> str:
        """
        Determine movement direction - simplified to up/down only.
        
        Returns: "up", "down", or "stationary"
        """
        dx = curr_point[0] - prev_point[0]
        dy = curr_point[1] - prev_point[1]
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Increase threshold for stationary to handle curve tolerance
        if abs_dx < 5 and abs_dy < 5:
            return "stationary"
        
        # Only consider up/down movement, ignore horizontal (curve tolerance)
        if abs_dy > abs_dx * 0.5:  # Allow some horizontal movement for curves
            return "up" if dy < 0 else "down"
        else:
            return "stationary"  # Too much horizontal movement, likely curve entry/exit
    
    def update(self, tracker_id: int, x_center: float, y_center: float) -> bool:
        """
        Update vehicle position and check if moving wrong way.
        
        Args:
            tracker_id: Vehicle tracking ID
            x_center: X coordinate of vehicle center
            y_center: Y coordinate of vehicle center
        
        Returns:
            True if vehicle is moving wrong way, False otherwise
        """
        current_point = (x_center, y_center)
        in_zone = self._point_in_zone(current_point)
        
        if not in_zone:
            if tracker_id in self.vehicle_history:
                del self.vehicle_history[tracker_id]
            return False
        
        if tracker_id not in self.vehicle_history:
            self.vehicle_history[tracker_id] = [current_point]
            return False
        
        prev_point = self.vehicle_history[tracker_id][-1]
        direction = self._get_direction(prev_point, current_point)
        
        self.vehicle_history[tracker_id].append(current_point)
        
        if direction == "stationary":
            return tracker_id in self.wrong_way_vehicles
        
        is_wrong_way = direction != self.zone.allowed_direction
        
        if is_wrong_way:
            self.wrong_way_vehicles.add(tracker_id)
        else:
            self.wrong_way_vehicles.discard(tracker_id)
        
        return is_wrong_way
    
    def is_wrong_way(self, tracker_id: int) -> bool:
        """Check if vehicle is marked as wrong way."""
        return tracker_id in self.wrong_way_vehicles
    
    def draw_zone(self, frame) -> None:
        """Draw zone polygon on frame."""
        cv2.polylines(frame, [self.polygon], True, (0, 255, 255), 2)
        cv2.putText(
            frame,
            f"Direction: {self.zone.allowed_direction.upper()}",
            self.zone.top_left,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
    
    def reset(self) -> None:
        """Reset detector state."""
        self.vehicle_history.clear()
        self.wrong_way_vehicles.clear()