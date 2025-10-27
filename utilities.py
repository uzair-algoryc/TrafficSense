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


"""
UPDATED WrongWayDetector Class in utilities.py
Replace the entire WrongWayDetector class with this version.
Keeps trajectory buffer for smooth direction tracking, removes angle tolerance.
"""

import math

@dataclass
class WrongWayZone:
    """Represents a traffic zone with allowed direction."""
    top_left: tuple
    top_right: tuple
    bottom_left: tuple
    bottom_right: tuple
    allowed_direction: str  # "up", "down"

class WrongWayDetector:
    """
    Enhanced detector with trajectory-based direction tracking.
    Uses buffer to smooth detections and reduce false positives.
    """
    
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
        
        # Enhanced vehicle tracking with trajectory buffer
        self.vehicle_history = {}  # track_id -> list of (x, y) positions
        self.wrong_way_vehicles = set()  # Persistent set of violating vehicles
        self.trajectory_buffer_size = 10  # Store last 10 positions (0.33s @ 30fps)
    
    def _point_in_zone(self, point: tuple) -> bool:
        """Check if point is inside zone polygon."""
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0
    
    def _get_movement_direction(self, trajectory: list) -> str:
        """
        Determine movement direction from trajectory.
        Uses buffer of positions for smooth, reliable direction detection.
        
        Returns:
            "up", "down", or "stationary"
        """
        if len(trajectory) < 3:
            return "stationary"
        
        # Calculate net movement from first to last position in buffer
        first_point = trajectory[0]
        last_point = trajectory[-1]
        
        dx = last_point[0] - first_point[0]
        dy = last_point[1] - first_point[1]
        
        # Calculate total movement distance
        total_distance = math.sqrt(dx**2 + dy**2)
        
        # If vehicle barely moved, consider it stationary
        if total_distance < 3.0:
            return "stationary"
        
        # Determine direction: if vertical movement is dominant
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Require at least 60% of movement to be vertical to classify as up/down
        if abs_dy > abs_dx * 0.6:
            # dy < 0 means moving UP (in image coords, lower y = up)
            # dy > 0 means moving DOWN (in image coords, higher y = down)
            return "up" if dy < 0 else "down"
        else:
            # Too much horizontal movement, likely curving into/out of zone
            return "stationary"
    
    def update(self, tracker_id: int, x_center: float, y_center: float) -> bool:
        """
        Update vehicle position and check if moving wrong way.
        
        Args:
            tracker_id: Vehicle tracking ID
            x_center: X coordinate of vehicle centroid
            y_center: Y coordinate of vehicle centroid
        
        Returns:
            True if vehicle is moving wrong way, False otherwise
        """
        current_point = (x_center, y_center)
        
        # Check if centroid is inside/touching zone
        in_zone = self._point_in_zone(current_point)
        
        if not in_zone:
            # Vehicle outside zone - clear history and remove from violations
            if tracker_id in self.vehicle_history:
                del self.vehicle_history[tracker_id]
            self.wrong_way_vehicles.discard(tracker_id)
            return False
        
        # Initialize trajectory for new vehicle
        if tracker_id not in self.vehicle_history:
            self.vehicle_history[tracker_id] = [current_point]
            return False
        
        # Add current position to trajectory buffer
        trajectory = self.vehicle_history[tracker_id]
        trajectory.append(current_point)
        
        # Maintain buffer size
        if len(trajectory) > self.trajectory_buffer_size:
            trajectory.pop(0)
        
        # Get movement direction from trajectory (minimum 3 points for reliability)
        if len(trajectory) >= 3:
            movement_direction = self._get_movement_direction(trajectory)
            
            # Check if moving in wrong direction
            if movement_direction != "stationary":
                is_allowed = (movement_direction == self.zone.allowed_direction)
                
                # Mark as wrong-way if moving opposite to allowed direction
                if not is_allowed:
                    self.wrong_way_vehicles.add(tracker_id)
                else:
                    self.wrong_way_vehicles.discard(tracker_id)
            else:
                # Stationary or mostly horizontal movement - not considered wrong-way
                self.wrong_way_vehicles.discard(tracker_id)
        
        return tracker_id in self.wrong_way_vehicles
    
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


class HybridVehicleCounter:
    """
    Hybrid vehicle counter combining trajectory analysis with line crossing detection.
    Uses trajectory direction validation for accurate counting on curved roads.
    """
    
    def __init__(self, start_point: tuple, end_point: tuple, mode: str = "both"):
        """
        Initialize hybrid counter.
        
        Args:
            start_point: Line start coordinates (x, y)
            end_point: Line end coordinates (x, y)
            mode: "both" (2-way), "in" (only IN), "out" (only OUT)
        """
        self.start = np.array(start_point, dtype=np.float64)
        self.end = np.array(end_point, dtype=np.float64)
        
        # Line equation: Ax + By + C = 0
        self.line_vec = self.end - self.start
        self.A = self.line_vec[1]  # dy
        self.B = -self.line_vec[0]  # -dx
        self.C = self.line_vec[0] * self.start[1] - self.line_vec[1] * self.start[0]
        
        # Vehicle class tracking with majority voting
        self.vehicle_classes = {}  # track_id -> final_class_id
        self.vehicle_class_history = {}  # track_id -> list of class_ids for majority voting
        
        # Trajectory parameters
        self.trajectory_buffer_size = 15  # 0.5 seconds at 30 FPS
        self.min_points_for_counting = 3  # Minimum trajectory length
        self.movement_threshold = 2.0  # Minimum movement magnitude
        
        # Vehicle tracking data
        self.vehicle_trajectories = {}  # track_id -> list of (x, y) points
        self.vehicle_last_side = {}  # track_id -> side of line (1 or -1)
        self.counted_vehicles = set()  # PERMANENT - never count twice
        # Note: vehicle_classes already defined above with majority voting support
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        self.mode = mode.lower()
        
        # Class-specific directional counters
        self.class_counts_in = {
            2: 0,  # Car IN
            3: 0,  # Motorcycle IN
            5: 0,  # Bus IN
            7: 0   # Truck IN
        }
        self.class_counts_out = {
            2: 0,  # Car OUT
            3: 0,  # Motorcycle OUT
            5: 0,  # Bus OUT
            7: 0   # Truck OUT
        }
    
    def _get_line_side(self, point):
        """Get which side of line the point is on using line equation"""
        x, y = point
        value = self.A * x + self.B * y + self.C
        return 1 if value > 0 else -1
    
    def _line_crossed(self, prev_point, curr_point):
        """Check if vehicle crossed the line between two points"""
        prev_side = self._get_line_side(prev_point)
        curr_side = self._get_line_side(curr_point)
        return prev_side != curr_side, prev_side, curr_side
    
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
    
    def _project_movement_on_line(self, avg_dx, avg_dy):
        """
        Project movement vector onto line perpendicular direction.
        Returns positive value for one direction, negative for opposite.
        """
        # Normalize line vector
        line_length = np.linalg.norm(self.line_vec)
        line_unit = self.line_vec / line_length
        
        # Perpendicular to line (normal vector)
        perpendicular = np.array([-line_unit[1], line_unit[0]], dtype=np.float64)
        
        # Movement vector
        movement = np.array([avg_dx, avg_dy], dtype=np.float64)
        
        # Project movement onto perpendicular direction
        projection = np.dot(movement, perpendicular)
        
        return projection
    
    def _determine_crossing_direction(self, prev_side, curr_side, trajectory):
        """
        Determine crossing direction using trajectory analysis.
        
        Returns: "in", "out", or None
        """
        if len(trajectory) < self.min_points_for_counting:
            return None
        
        # Calculate average movement direction
        movement = self._calculate_movement_direction(trajectory)
        if movement is None:
            return None
        
        avg_dx, avg_dy = movement
        
        # Check if movement is significant enough
        movement_magnitude = np.sqrt(avg_dx**2 + avg_dy**2)
        if movement_magnitude < self.movement_threshold:
            return None
        
        # Project movement onto line perpendicular
        projection = self._project_movement_on_line(avg_dx, avg_dy)
        
        # Determine direction based on side change and movement projection
        # Side -1 to +1 with positive projection = "in"
        # Side +1 to -1 with negative projection = "out"
        
        if prev_side == -1 and curr_side == 1:
            # Crossing from -1 to +1
            if projection > 0:
                return "in"
            else:
                return "out"
        elif prev_side == 1 and curr_side == -1:
            # Crossing from +1 to -1
            if projection < 0:
                return "out"
            else:
                return "in"
        
        return None
    
    def _update_vehicle_class(self, tracker_id: int, class_id: int):
        """
        Update vehicle class using majority voting system for robust classification.
        
        Args:
            tracker_id: Vehicle tracking ID
            class_id: Current frame's detected class ID
        """
        if class_id is None:
            return
        
        # Initialize class history if not exists
        if tracker_id not in self.vehicle_class_history:
            self.vehicle_class_history[tracker_id] = []
        
        # Add current class detection to history
        self.vehicle_class_history[tracker_id].append(class_id)
        
        # Keep only last 10 detections for efficiency and recent accuracy
        if len(self.vehicle_class_history[tracker_id]) > 10:
            self.vehicle_class_history[tracker_id].pop(0)
        
        # Determine most frequent class (majority vote)
        class_history = self.vehicle_class_history[tracker_id]
        if len(class_history) >= 3:  # Need at least 3 detections for reliable voting
            # Count occurrences of each class
            class_counts = {}
            for cls in class_history:
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            # Get the most frequent class
            most_frequent_class = max(class_counts, key=class_counts.get)
            
            # Update final class assignment
            old_class = self.vehicle_classes.get(tracker_id, None)
            if old_class != most_frequent_class:
                # Class changed - log for debugging
                class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                old_name = class_names.get(old_class, f"Class{old_class}") if old_class else "None"
                new_name = class_names.get(most_frequent_class, f"Class{most_frequent_class}")
                print(f"🔄 Vehicle {tracker_id}: Class updated from {old_name} to {new_name} (votes: {class_counts})")
            
            self.vehicle_classes[tracker_id] = most_frequent_class
        elif len(class_history) == 1:
            # For first detection, use it as initial assignment
            self.vehicle_classes[tracker_id] = class_history[0]
            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
            class_name = class_names.get(class_history[0], f"Class{class_history[0]}")
            print(f"🆕 Vehicle {tracker_id}: Initial class assignment = {class_name}")
    
    def update(self, tracker_id: int, x_center: float, y_center: float, class_id: int = None):
        """Update vehicle trajectory and check for line crossing"""
        if tracker_id is None:
            return
        
        # Skip if already counted
        if tracker_id in self.counted_vehicles:
            return
        
        current_point = np.array([x_center, y_center], dtype=np.float64)
        
        # Initialize trajectory for new vehicle
        if tracker_id not in self.vehicle_trajectories:
            self.vehicle_trajectories[tracker_id] = [current_point]
            self.vehicle_last_side[tracker_id] = self._get_line_side(current_point)
            # Initialize class history for new vehicle
            if tracker_id not in self.vehicle_class_history:
                self.vehicle_class_history[tracker_id] = []
            return
        
        # Update vehicle class using majority voting system
        self._update_vehicle_class(tracker_id, class_id)
        
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
            
            crossed, prev_side, curr_side = self._line_crossed(prev_point, curr_point)
            
            if crossed:
                # Line crossed! Determine direction using trajectory
                direction = self._determine_crossing_direction(prev_side, curr_side, trajectory)
                
                if direction == "in" and self.mode in ["both", "in"]:
                    self.in_count += 1
                    self.counted_vehicles.add(tracker_id)
                    # Increment class IN counter with logging
                    if tracker_id in self.vehicle_classes:
                        vehicle_class = self.vehicle_classes[tracker_id]
                        if vehicle_class in self.class_counts_in:
                            self.class_counts_in[vehicle_class] += 1
                            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                            class_name = class_names.get(vehicle_class, f"Class{vehicle_class}")
                            print(f"✅ COUNTED IN: Vehicle {tracker_id} as {class_name} | {class_name}_In: {self.class_counts_in[vehicle_class]}")
                            
                elif direction == "out" and self.mode in ["both", "out"]:
                    self.out_count += 1
                    self.counted_vehicles.add(tracker_id)
                    # Increment class OUT counter with logging
                    if tracker_id in self.vehicle_classes:
                        vehicle_class = self.vehicle_classes[tracker_id]
                        if vehicle_class in self.class_counts_out:
                            self.class_counts_out[vehicle_class] += 1
                            class_names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                            class_name = class_names.get(vehicle_class, f"Class{vehicle_class}")
                            print(f"✅ COUNTED OUT: Vehicle {tracker_id} as {class_name} | {class_name}_Out: {self.class_counts_out[vehicle_class]}")
                
                # Update last side
                self.vehicle_last_side[tracker_id] = curr_side
    
    def get_counts(self):
        """Get current counts"""
        return {
            'in_count': self.in_count,
            'out_count': self.out_count
        }
    
    def reset(self):
        """Reset all counters and tracking data"""
        self.vehicle_trajectories = {}
        self.vehicle_last_side = {}
        self.counted_vehicles = set()
        self.vehicle_classes = {}
        self.vehicle_class_history = {}  # Reset class history
        self.in_count = 0
        self.out_count = 0
        # Reset directional class counters
        self.class_counts_in = {2: 0, 3: 0, 5: 0, 7: 0}
        self.class_counts_out = {2: 0, 3: 0, 5: 0, 7: 0}