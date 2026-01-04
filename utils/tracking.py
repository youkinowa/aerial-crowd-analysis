import numpy as np
import cv2
from collections import defaultdict, deque
import torch
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    """
    Multi-object tracker for aerial footage using IoU matching
    and Kalman filtering for trajectory smoothing
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the tracker
        
        Args:
            max_age: Maximum frames to keep a track alive without matching detections
            min_hits: Minimum number of matches needed to establish a reliable track
            iou_threshold: Minimum IoU to associate detections with tracks
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Track counter and active tracks
        self.next_track_id = 0
        self.tracks = []
        
        # Historical tracks for analysis
        self.track_history = defaultdict(list)
        
        # For calculating flow
        self.prev_frame_detections = None
    
    def update(self, detections, frame=None):
        """
        Update tracks with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, score, class_id]
            frame: Current video frame (optional, for appearance matching)
            
        Returns:
            List of active tracks as [x1, y1, x2, y2, track_id, class_id]
        """
        # Initialize lists of matched, unmatched tracks and detections
        matched_tracks = []
        unmatched_tracks = []
        unmatched_detections = []
        
        # Extract bounding boxes
        if len(detections) > 0:
            detection_boxes = np.array([d[:4] for d in detections])
            detection_scores = np.array([d[4] for d in detections])
            detection_classes = np.array([d[5] for d in detections]) if len(detections[0]) > 5 else np.zeros(len(detections))
        else:
            detection_boxes = np.empty((0, 4))
            detection_scores = np.array([])
            detection_classes = np.array([])
        
        # Handle edge case: no tracks or no detections
        if len(self.tracks) == 0:
            # Initialize tracks for all detections
            for i, det in enumerate(detection_boxes):
                self._initiate_track(det, detection_scores[i], detection_classes[i])
            return self._get_track_results()
            
        if len(detection_boxes) == 0:
            # No detections - update all tracks with no matches
            for track in self.tracks:
                track['time_since_update'] += 1
            
            # Remove old tracks
            self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
            return self._get_track_results()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detection_boxes)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detection_boxes):
                # Only match with same class if class info available
                if track['class_id'] == detection_classes[d] or detection_classes[d] == -1:
                    iou_matrix[t, d] = self._calculate_iou(track['bbox'], det)
        
        # Solve the linear assignment problem using Hungarian algorithm
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
        
        # Process assignments
        for t, d in zip(track_indices, detection_indices):
            if iou_matrix[t, d] >= self.iou_threshold:
                self._update_track(self.tracks[t], detection_boxes[d], detection_scores[d])
                matched_tracks.append(t)
                unmatched_detections.append(d)
            else:
                unmatched_tracks.append(t)
        
        # Add unmatched tracks
        unmatched_tracks.extend([i for i in range(len(self.tracks)) if i not in matched_tracks])
        
        # Add unmatched detections
        unmatched_detections = [i for i in range(len(detection_boxes)) if i not in unmatched_detections]
        
        # Update unmatched tracks
        for i in unmatched_tracks:
            self.tracks[i]['time_since_update'] += 1
        
        # Create new tracks for unmatched detections
        for i in unmatched_detections:
            self._initiate_track(detection_boxes[i], detection_scores[i], detection_classes[i])
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        # Update flow information
        if frame is not None:
            self.prev_frame_detections = detection_boxes.copy()
        
        # Return results
        return self._get_track_results()
    
    def _get_track_results(self):
        """
        Get active track results in the format [x1, y1, x2, y2, track_id, class_id]
        
        Returns:
            List of active tracks
        """
        results = []
        
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] <= 1:
                box = track['bbox']
                results.append([box[0], box[1], box[2], box[3], track['track_id'], track['class_id']])
                
                # Save to history for trajectory analysis
                self.track_history[track['track_id']].append({
                    'bbox': box,
                    'frame': track['last_frame'],
                    'class_id': track['class_id']
                })
        
        return np.array(results)
    
    def _initiate_track(self, bbox, score, class_id):
        """
        Create a new track from a detection
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            score: Detection confidence score
            class_id: Object class ID
        """
        self.tracks.append({
            'track_id': self.next_track_id,
            'bbox': bbox,
            'score': score,
            'class_id': class_id,
            'hits': 1,
            'time_since_update': 0,
            'last_frame': 0,
            'velocity': [0, 0, 0, 0],
            'kalman': self._init_kalman_filter(bbox)
        })
        
        self.next_track_id += 1
    
    def _update_track(self, track, bbox, score, frame_id=None):
        """
        Update an existing track with a new detection
        
        Args:
            track: Existing track dictionary
            bbox: New bounding box [x1, y1, x2, y2]
            score: New detection confidence score
            frame_id: Current frame ID
        """
        # Apply Kalman prediction and update
        if track['kalman'] is not None:
            kalman_bbox = self._update_kalman(track['kalman'], bbox)
            # Blend Kalman and detection (smoothing)
            bbox = self._blend_boxes(kalman_bbox, bbox, 0.7)
        
        # Calculate velocity: current minus previous position
        prev_bbox = track['bbox']
        track['velocity'] = [
            bbox[0] - prev_bbox[0],
            bbox[1] - prev_bbox[1],
            bbox[2] - prev_bbox[2],
            bbox[3] - prev_bbox[3]
        ]
        
        # Update track info
        track['bbox'] = bbox
        track['score'] = max(track['score'], score) if track['score'] is not None else score
        track['hits'] += 1
        track['time_since_update'] = 0
        
        if frame_id is not None:
            track['last_frame'] = frame_id
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def _init_kalman_filter(self, bbox):
        """
        Initialize a Kalman filter for a track
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            
        Returns:
            OpenCV Kalman filter object
        """
        kalman = cv2.KalmanFilter(8, 4)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Initialize state
        kalman.statePre = np.array([
            [bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]
        ], np.float32)
        
        kalman.statePost = np.array([
            [bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]], [0], [0], [0], [0]
        ], np.float32)
        
        return kalman
    
    def _update_kalman(self, kalman, bbox):
        """
        Update the Kalman filter with a new measurement
        
        Args:
            kalman: OpenCV Kalman filter object
            bbox: New bounding box [x1, y1, x2, y2]
            
        Returns:
            Predicted and updated bounding box
        """
        # Predict
        prediction = kalman.predict()
        
        # Create measurement
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], np.float32)
        
        # Update with measurement
        kalman.correct(measurement)
        
        # Get state
        state = kalman.statePost
        
        # Return as bbox
        return [state[0, 0], state[1, 0], state[2, 0], state[3, 0]]
    
    def _blend_boxes(self, box1, box2, weight=0.5):
        """
        Blend two bounding boxes with a weight
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            weight: Weight of the first box (0-1)
            
        Returns:
            Blended box
        """
        return [
            box1[0] * weight + box2[0] * (1 - weight),
            box1[1] * weight + box2[1] * (1 - weight),
            box1[2] * weight + box2[2] * (1 - weight),
            box1[3] * weight + box2[3] * (1 - weight)
        ]
    
    def get_crowd_flow(self, frame_shape, grid_size=(10, 10)):
        """
        Calculate crowd flow from tracked objects
        
        Args:
            frame_shape: Shape of the frame (height, width)
            grid_size: Size of the grid (rows, cols)
            
        Returns:
            Grid of motion vectors with magnitude (velocity)
        """
        height, width = frame_shape[:2]
        cell_height = height // grid_size[0]
        cell_width = width // grid_size[1]
        
        # Initialize grid
        flow_grid = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.float32)
        count_grid = np.zeros((grid_size[0], grid_size[1]), dtype=np.int32)
        
        # Calculate average velocity in each grid cell
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] <= 1:
                # Get track center
                bbox = track['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Get grid cell
                grid_x = min(int(center_x / cell_width), grid_size[1] - 1)
                grid_y = min(int(center_y / cell_height), grid_size[0] - 1)
                
                # Get velocity
                vx = (track['velocity'][0] + track['velocity'][2]) / 2
                vy = (track['velocity'][1] + track['velocity'][3]) / 2
                velocity = np.sqrt(vx**2 + vy**2)
                
                # Update grid
                flow_grid[grid_y, grid_x, 0] += vx
                flow_grid[grid_y, grid_x, 1] += vy
                flow_grid[grid_y, grid_x, 2] += velocity
                count_grid[grid_y, grid_x] += 1
        
        # Normalize by count
        for y in range(grid_size[0]):
            for x in range(grid_size[1]):
                if count_grid[y, x] > 0:
                    flow_grid[y, x] /= count_grid[y, x]
        
        return flow_grid, count_grid
    
    def get_density_map(self, frame_shape, sigma=20):
        """
        Calculate crowd density from tracked objects
        
        Args:
            frame_shape: Shape of the frame (height, width)
            sigma: Sigma for Gaussian kernel
            
        Returns:
            Density map as a heatmap
        """
        height, width = frame_shape[:2]
        density_map = np.zeros((height, width), dtype=np.float32)
        
        # Add a Gaussian for each track
        for track in self.tracks:
            if track['hits'] >= self.min_hits and track['time_since_update'] <= 1:
                # Get track center
                bbox = track['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                if 0 <= center_x < width and 0 <= center_y < height:
                    density_map[center_y, center_x] += 1
        
        # Apply Gaussian filter
        if np.sum(density_map) > 0:
            density_map = cv2.GaussianBlur(density_map, (0, 0), sigma)
            # Normalize
            density_map = density_map / np.max(density_map) if np.max(density_map) > 0 else density_map
        
        return density_map
    
    def get_trajectories(self, max_age=None):
        """
        Get all tracked trajectories
        
        Args:
            max_age: Maximum age of trajectory in frames to include
            
        Returns:
            Dictionary of track_id -> list of positions
        """
        trajectories = {}
        
        for track_id, history in self.track_history.items():
            if max_age is None or len(history) <= max_age:
                trajectories[track_id] = []
                for point in history:
                    bbox = point['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    trajectories[track_id].append((center_x, center_y, point['frame']))
        
        return trajectories
    
    def reset(self):
        """Reset the tracker"""
        self.tracks = []
        self.next_track_id = 0
        self.track_history = defaultdict(list)
        self.prev_frame_detections = None

class TrackingFeatureExtractor:
    """
    Extract features from tracked object trajectories for behavior analysis
    """
    
    def __init__(self, feature_types=None, window_size=10):
        """
        Initialize feature extractor
        
        Args:
            feature_types: List of feature types to extract
            window_size: Window size for feature extraction
        """
        self.feature_types = feature_types or [
            'velocity', 'acceleration', 'direction', 'curvature',
            'stop_ratio', 'trajectory_length'
        ]
        self.window_size = window_size
    
    def extract_features(self, trajectories):
        """
        Extract features from trajectories
        
        Args:
            trajectories: Dictionary of track_id -> list of (x, y, frame) points
            
        Returns:
            Dictionary of track_id -> features
        """
        features = {}
        
        for track_id, points in trajectories.items():
            if len(points) < 3:  # Need at least 3 points for meaningful features
                continue
                
            track_features = {}
            
            # Extract positions and calculate basic properties
            positions = np.array([(p[0], p[1]) for p in points])
            frames = np.array([p[2] for p in points])
            
            # Calculate velocity
            if 'velocity' in self.feature_types:
                velocities = np.zeros((len(positions) - 1, 2))
                for i in range(len(positions) - 1):
                    velocities[i] = positions[i+1] - positions[i]
                
                track_features['velocity_mean'] = np.mean(np.linalg.norm(velocities, axis=1))
                track_features['velocity_std'] = np.std(np.linalg.norm(velocities, axis=1))
                track_features['velocity_max'] = np.max(np.linalg.norm(velocities, axis=1))
            
            # Calculate acceleration
            if 'acceleration' in self.feature_types and len(velocities) > 1:
                accelerations = np.zeros((len(velocities) - 1, 2))
                for i in range(len(velocities) - 1):
                    accelerations[i] = velocities[i+1] - velocities[i]
                
                track_features['acceleration_mean'] = np.mean(np.linalg.norm(accelerations, axis=1))
                track_features['acceleration_std'] = np.std(np.linalg.norm(accelerations, axis=1))
                track_features['acceleration_max'] = np.max(np.linalg.norm(accelerations, axis=1))
            
            # Calculate direction changes
            if 'direction' in self.feature_types and len(velocities) > 1:
                directions = np.zeros(len(velocities))
                for i in range(len(velocities)):
                    directions[i] = np.arctan2(velocities[i, 1], velocities[i, 0])
                
                direction_changes = np.zeros(len(directions) - 1)
                for i in range(len(directions) - 1):
                    diff = directions[i+1] - directions[i]
                    # Normalize to [-pi, pi]
                    if diff > np.pi:
                        diff -= 2 * np.pi
                    elif diff < -np.pi:
                        diff += 2 * np.pi
                    direction_changes[i] = np.abs(diff)
                
                track_features['direction_change_mean'] = np.mean(direction_changes)
                track_features['direction_change_std'] = np.std(direction_changes)
                track_features['direction_change_max'] = np.max(direction_changes)
            
            # Calculate curvature
            if 'curvature' in self.feature_types and len(positions) > 2:
                curvatures = []
                for i in range(1, len(positions) - 1):
                    # Use 3 points to calculate curvature
                    p1, p2, p3 = positions[i-1], positions[i], positions[i+1]
                    
                    # Calculate vectors
                    v1 = p2 - p1
                    v2 = p3 - p2
                    
                    # Normalize
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        # Calculate angle between vectors
                        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                        cos_angle = min(1.0, max(-1.0, cos_angle))  # Clip to valid range
                        angle = np.arccos(cos_angle)
                        
                        # Calculate curvature (approximation)
                        curvature = angle / ((v1_norm + v2_norm) / 2)
                        curvatures.append(curvature)
                
                if curvatures:
                    track_features['curvature_mean'] = np.mean(curvatures)
                    track_features['curvature_std'] = np.std(curvatures)
                    track_features['curvature_max'] = np.max(curvatures)
            
            # Calculate stop ratio (frames with very low velocity)
            if 'stop_ratio' in self.feature_types and len(velocities) > 0:
                velocity_magnitudes = np.linalg.norm(velocities, axis=1)
                stop_threshold = 0.1 * np.mean(velocity_magnitudes)
                stops = (velocity_magnitudes < stop_threshold).sum()
                track_features['stop_ratio'] = stops / len(velocity_magnitudes)
            
            # Calculate trajectory length
            if 'trajectory_length' in self.feature_types:
                length = 0
                for i in range(len(positions) - 1):
                    length += np.linalg.norm(positions[i+1] - positions[i])
                track_features['trajectory_length'] = length
                
                # Direct distance (displacement)
                displacement = np.linalg.norm(positions[-1] - positions[0])
                track_features['displacement'] = displacement
                
                # Straightness ratio
                track_features['straightness_ratio'] = displacement / length if length > 0 else 1.0
            
            # Add window-based features
            if len(positions) >= self.window_size:
                window_features = self._extract_window_features(positions, frames)
                track_features.update(window_features)
            
            features[track_id] = track_features
        
        return features
    
    def _extract_window_features(self, positions, frames):
        """
        Extract features using a sliding window approach
        
        Args:
            positions: Array of (x, y) positions
            frames: Array of frame numbers
            
        Returns:
            Dictionary of window-based features
        """
        window_features = {}
        
        # Extract windows
        windows = []
        for i in range(0, len(positions) - self.window_size + 1, self.window_size // 2):
            windows.append(positions[i:i+self.window_size])
        
        if not windows:
            return window_features
        
        # Calculate features for each window
        window_velocities = []
        window_directions = []
        window_curvatures = []
        
        for window in windows:
            # Velocity in window
            window_velocity = np.linalg.norm(window[-1] - window[0]) / len(window)
            window_velocities.append(window_velocity)
            
            # Direction in window
            window_direction = np.arctan2(window[-1][1] - window[0][1], 
                                        window[-1][0] - window[0][0])
            window_directions.append(window_direction)
            
            # Curvature in window
            curves = []
            for i in range(1, len(window) - 1):
                v1 = window[i] - window[i-1]
                v2 = window[i+1] - window[i]
                
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = min(1.0, max(-1.0, cos_angle))
                    angle = np.arccos(cos_angle)
                    curves.append(angle)
            
            if curves:
                window_curvatures.append(np.mean(curves))
        
        # Calculate window statistics
        if window_velocities:
            window_features['window_velocity_mean'] = np.mean(window_velocities)
            window_features['window_velocity_std'] = np.std(window_velocities)
            window_features['window_velocity_max'] = np.max(window_velocities)
        
        if window_directions:
            # Calculate direction changes between windows
            direction_changes = []
            for i in range(1, len(window_directions)):
                diff = window_directions[i] - window_directions[i-1]
                # Normalize to [-pi, pi]
                if diff > np.pi:
                    diff -= 2 * np.pi
                elif diff < -np.pi:
                    diff += 2 * np.pi
                direction_changes.append(np.abs(diff))
            
            if direction_changes:
                window_features['window_direction_change_mean'] = np.mean(direction_changes)
                window_features['window_direction_change_max'] = np.max(direction_changes)
        
        if window_curvatures:
            window_features['window_curvature_mean'] = np.mean(window_curvatures)
            window_features['window_curvature_max'] = np.max(window_curvatures)
        
        return window_features
    
    def get_feature_vector(self, features):
        """
        Convert feature dictionary to a feature vector
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector as numpy array
        """
        # Define the order of features
        feature_keys = [
            'velocity_mean', 'velocity_std', 'velocity_max',
            'acceleration_mean', 'acceleration_std', 'acceleration_max',
            'direction_change_mean', 'direction_change_std', 'direction_change_max',
            'curvature_mean', 'curvature_std', 'curvature_max',
            'stop_ratio', 'trajectory_length', 'displacement', 'straightness_ratio',
            'window_velocity_mean', 'window_velocity_std', 'window_velocity_max',
            'window_direction_change_mean', 'window_direction_change_max',
            'window_curvature_mean', 'window_curvature_max'
        ]
        
        # Extract features in order, replacing missing with 0
        vector = []
        for key in feature_keys:
            if key in features:
                vector.append(features[key])
            else:
                vector.append(0.0)
        
        return np.array(vector)
    
    def get_feature_vectors(self, feature_dict):
        """
        Convert dictionary of features to matrix of feature vectors
        
        Args:
            feature_dict: Dictionary of track_id -> features
            
        Returns:
            Feature matrix (n_samples, n_features) and track IDs
        """
        if not feature_dict:
            return np.array([]), []
            
        track_ids = list(feature_dict.keys())
        feature_vectors = [self.get_feature_vector(feature_dict[tid]) for tid in track_ids]
        
        return np.array(feature_vectors), track_ids 