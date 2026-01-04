import numpy as np
from hmmlearn import hmm
import pickle
import os
import torch
import math
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import scipy.stats as stats

class CrowdBehaviorHMM:
    """Hidden Markov Model for crowd behavior analysis"""
    
    def __init__(self, n_states=5, n_features=4, covariance_type='diag', n_iter=100):
        """
        Args:
            n_states: Number of hidden states in the model
            n_features: Number of features (e.g., velocity, acceleration, density, etc.)
            covariance_type: Covariance type for the emission probabilities
            n_iter: Number of iterations for training
        """
        self.n_states = n_states
        self.n_features = n_features
        
        # Initialize HMM model
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            verbose=True
        )
        
        # State labels for interpretation
        self.state_labels = {
            0: "normal_movement",
            1: "slow_movement",
            2: "stationary",
            3: "rapid_movement",
            4: "dispersing"
        }
        
        # Store thresholds for anomaly detection
        self.log_likelihood_threshold = None
        self.state_sequence_history = deque(maxlen=100)  # Store recent state sequences
        self.transition_counts = np.zeros((n_states, n_states))
        
        # History of previous events
        self.event_history = []
    
    def preprocess_trajectories(self, trajectories):
        """
        Preprocess raw trajectories into feature vectors
        
        Args:
            trajectories: List of trajectories, each being a list of (x, y) coordinates
        
        Returns:
            Features array with shape (n_samples, n_features)
        """
        features = []
        
        for trajectory in trajectories:
            # Convert to numpy array if it's not already
            if isinstance(trajectory, list):
                trajectory = np.array(trajectory)
            
            # Skip trajectories that are too short
            if len(trajectory) < 3:
                continue
            
            # Calculate velocities (displacement between consecutive points)
            velocities = np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1))
            
            # Calculate accelerations
            accelerations = np.diff(velocities)
            
            # Calculate direction changes (angles between consecutive velocity vectors)
            directions = []
            for i in range(len(trajectory) - 2):
                v1 = trajectory[i+1] - trajectory[i]
                v2 = trajectory[i+2] - trajectory[i+1]
                
                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
                angle = np.arccos(cos_angle)
                directions.append(angle)
            
            # Ensure all feature vectors have the same length
            min_length = min(len(velocities)-1, len(accelerations), len(directions))
            
            # Create feature vectors for each time step in the trajectory
            for i in range(min_length):
                # Features: [velocity, acceleration, direction_change, optional: density]
                if self.n_features == 3:
                    feature_vector = [velocities[i], accelerations[i], directions[i]]
                else:  # n_features == 4, add a placeholder for density
                    feature_vector = [velocities[i], accelerations[i], directions[i], 0.0]
                
                features.append(feature_vector)
        
        return np.array(features)
    
    def add_crowd_density_feature(self, features, density_map):
        """
        Add crowd density as a feature
        
        Args:
            features: Feature array [n_samples, n_features]
            density_map: 2D array representing crowd density at different locations
            
        Returns:
            Updated features with density information
        """
        if self.n_features < 4 or density_map is None:
            return features
        
        # Assuming the first two columns of original trajectories are preserved
        for i, feature in enumerate(features):
            x, y = int(feature[0]), int(feature[1])  # Use velocity as proxy for position
            
            # Ensure coordinates are within density map bounds
            x = max(0, min(x, density_map.shape[1] - 1))
            y = max(0, min(y, density_map.shape[0] - 1))
            
            # Add density value as a feature
            features[i, 3] = density_map[y, x]
            
        return features
    
    def fit(self, features):
        """
        Train the HMM model on feature data
        
        Args:
            features: Feature array with shape [n_samples, n_features]
        """
        # Ensure features is a numpy array with the right shape
        features = np.array(features)
        
        if features.shape[1] != self.n_features:
            raise ValueError(f"Features dimension mismatch. Expected {self.n_features}, got {features.shape[1]}")
        
        # Normalize features
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        normalized_features = (features - feature_means) / (feature_stds + 1e-10)
        
        # Reshape for hmmlearn if needed
        X = normalized_features.reshape(-1, self.n_features)
        
        # Train the model
        self.model.fit(X)
        
        # Calculate log likelihood threshold for anomaly detection (mean - 2*std)
        log_likelihoods = []
        for i in range(0, len(X), 100):  # Process in chunks to avoid memory issues
            chunk = X[i:i+100]
            log_likelihoods.append(self.model.score(chunk))
        
        self.log_likelihood_threshold = np.mean(log_likelihoods) - 2 * np.std(log_likelihoods)
        
        return self
    
    def predict_states(self, features):
        """
        Predict the hidden states for the given features
        
        Args:
            features: Feature array with shape [n_samples, n_features]
            
        Returns:
            Predicted state sequence
        """
        # Normalize features using the same parameters as during training
        features = np.array(features)
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        normalized_features = (features - feature_means) / (feature_stds + 1e-10)
        
        # Reshape for hmmlearn if needed
        X = normalized_features.reshape(-1, self.n_features)
        
        # Predict hidden states
        state_sequence = self.model.predict(X)
        
        # Update state sequence history
        self.state_sequence_history.append(state_sequence)
        
        # Update transition counts
        for i in range(len(state_sequence) - 1):
            from_state = state_sequence[i]
            to_state = state_sequence[i + 1]
            self.transition_counts[from_state, to_state] += 1
        
        return state_sequence
    
    def detect_anomalies(self, features, anomaly_threshold=0.05):
        """
        Detect anomalies in the crowd behavior
        
        Args:
            features: Feature array with shape [n_samples, n_features]
            anomaly_threshold: Threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly scores and detected anomalies
        """
        # Normalize features
        features = np.array(features)
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        normalized_features = (features - feature_means) / (feature_stds + 1e-10)
        
        X = normalized_features.reshape(-1, self.n_features)
        
        # Get log likelihood of the sequence
        log_likelihood = self.model.score(X)
        
        # Predict states
        state_sequence = self.model.predict(X)
        
        # Find unusual state transitions
        unusual_transitions = []
        for i in range(len(state_sequence) - 1):
            from_state = state_sequence[i]
            to_state = state_sequence[i + 1]
            
            # Calculate transition probability
            total_transitions_from_state = np.sum(self.transition_counts[from_state])
            
            if total_transitions_from_state > 0:
                transition_prob = self.transition_counts[from_state, to_state] / total_transitions_from_state
                
                if transition_prob < anomaly_threshold:
                    unusual_transitions.append((i, from_state, to_state, transition_prob))
        
        # Detect state distribution anomalies
        state_counts = np.bincount(state_sequence, minlength=self.n_states)
        state_distribution = state_counts / len(state_sequence)
        
        # Compare with historical state distributions
        historical_distributions = []
        for past_sequence in self.state_sequence_history:
            if len(past_sequence) > 0:
                past_counts = np.bincount(past_sequence, minlength=self.n_states)
                historical_distributions.append(past_counts / len(past_sequence))
        
        distribution_anomaly = False
        dist_anomaly_score = 0
        
        if historical_distributions:
            # Calculate mean historical distribution
            mean_hist_dist = np.mean(historical_distributions, axis=0)
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (state_distribution + mean_hist_dist)
            js_div = 0.5 * (stats.entropy(state_distribution, m) + stats.entropy(mean_hist_dist, m))
            
            dist_anomaly_score = js_div
            distribution_anomaly = js_div > anomaly_threshold
        
        # Determine if sequence is anomalous
        is_anomalous = (log_likelihood < self.log_likelihood_threshold) or len(unusual_transitions) > 0 or distribution_anomaly
        
        return {
            'is_anomalous': is_anomalous,
            'log_likelihood': log_likelihood,
            'log_likelihood_threshold': self.log_likelihood_threshold,
            'unusual_transitions': unusual_transitions,
            'distribution_anomaly': distribution_anomaly,
            'distribution_anomaly_score': dist_anomaly_score,
            'state_sequence': state_sequence,
            'state_labels': [self.state_labels[s] for s in state_sequence]
        }
    
    def analyze_behavior_pattern(self, state_sequence):
        """
        Analyze a state sequence to identify behavior patterns
        
        Args:
            state_sequence: Array of state indices
            
        Returns:
            Dictionary with behavior pattern analysis
        """
        # Count occurrences of each state
        state_counts = np.bincount(state_sequence, minlength=self.n_states)
        state_percentages = state_counts / len(state_sequence) * 100
        
        # Find the most common state
        most_common_state = np.argmax(state_counts)
        
        # Check for rapid state changes (potential indicator of chaotic behavior)
        state_changes = np.sum(np.diff(state_sequence) != 0)
        change_rate = state_changes / (len(state_sequence) - 1) if len(state_sequence) > 1 else 0
        
        # Identify runs of the same state (stable behavior periods)
        runs = []
        current_run = [0, state_sequence[0]]  # [length, state]
        
        for i in range(1, len(state_sequence)):
            if state_sequence[i] == current_run[1]:
                current_run[0] += 1
            else:
                runs.append(current_run)
                current_run = [1, state_sequence[i]]
        
        runs.append(current_run)
        
        # Find the longest stable period
        longest_run = max(runs, key=lambda x: x[0]) if runs else [0, 0]
        
        # Determine overall behavior pattern
        if change_rate > 0.5:
            pattern = "erratic"
        elif state_percentages[3] > 40:  # If "rapid_movement" is dominant
            pattern = "rapid"
        elif state_percentages[2] > 40:  # If "stationary" is dominant
            pattern = "stationary"
        elif state_percentages[4] > 30:  # If "dispersing" is significant
            pattern = "dispersing"
        else:
            pattern = "normal"
        
        return {
            'pattern': pattern,
            'state_percentages': state_percentages,
            'most_common_state': most_common_state,
            'most_common_state_label': self.state_labels[most_common_state],
            'change_rate': change_rate,
            'longest_stable_period': longest_run,
            'longest_stable_state': self.state_labels[longest_run[1]]
        }
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'state_labels': self.state_labels,
                'log_likelihood_threshold': self.log_likelihood_threshold,
                'transition_counts': self.transition_counts,
                'n_states': self.n_states,
                'n_features': self.n_features,
                'event_history': self.event_history
            }, f)
    
    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.state_labels = data['state_labels']
        self.log_likelihood_threshold = data['log_likelihood_threshold']
        self.transition_counts = data['transition_counts']
        self.n_states = data['n_states']
        self.n_features = data['n_features']
        self.event_history = data['event_history']
        
        return self
    
    def record_event(self, event_type, timestamp, location, description=None):
        """Record an event into the history for future reference"""
        event = {
            'type': event_type,
            'timestamp': timestamp,
            'location': location,
            'description': description
        }
        self.event_history.append(event)
    
    def get_similar_past_events(self, current_features, top_k=3):
        """Find similar past events based on feature similarity"""
        if not self.event_history:
            return []
        
        # Extract features from past events
        past_features = []
        for event in self.event_history:
            if 'features' in event:
                past_features.append((event, event['features']))
        
        if not past_features:
            return []
        
        # Calculate similarity with current features
        similarities = []
        for event, features in past_features:
            # Simple Euclidean distance
            distance = np.mean(np.sqrt(np.sum((features - current_features)**2, axis=1)))
            similarities.append((event, 1.0 / (1.0 + distance)))  # Convert distance to similarity
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k similar events
        return similarities[:top_k]

class MultiPersonTracker:
    """Helper class for tracking multiple people across frames"""
    
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.tracks = {}  # Dictionary to store tracks: {id: track}
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
    
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2, confidence, class_id]
            
        Returns:
            Dictionary of active tracks {id: track}
        """
        # Convert detections to the format: [x1, y1, x2, y2, confidence]
        dets = []
        for det in detections:
            if len(det) >= 5:  # Make sure detection has enough elements
                dets.append(det[:5])  # Take only the first 5 elements
        
        # If no detections or no existing tracks
        if not dets:
            # Increment age for all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                
                # Remove old tracks
                if self.tracks[track_id]['age'] > self.max_age:
                    self.tracks.pop(track_id)
            return self.tracks
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for det in dets:
                self.tracks[self.next_id] = {
                    'bbox': det[:4],
                    'score': det[4],
                    'age': 0,
                    'hits': 1,
                    'trajectory': [((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)]  # Center point
                }
                self.next_id += 1
            return self.tracks
        
        # Calculate IoU between existing tracks and new detections
        iou_matrix = np.zeros((len(self.tracks), len(dets)))
        for i, track_id in enumerate(self.tracks.keys()):
            for j, det in enumerate(dets):
                iou_matrix[i, j] = self._calculate_iou(self.tracks[track_id]['bbox'], det[:4])
        
        # Hungarian algorithm for assignment
        try:
            from scipy.optimize import linear_sum_assignment
            track_indices, det_indices = linear_sum_assignment(-iou_matrix)
        except ImportError:
            # Fallback to greedy assignment
            track_indices, det_indices = [], []
            for i in range(min(len(self.tracks), len(dets))):
                track_indices.append(i)
                det_indices.append(i)
        
        # Update matched tracks
        track_ids = list(self.tracks.keys())
        for track_idx, det_idx in zip(track_indices, det_indices):
            if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                track_id = track_ids[track_idx]
                self.tracks[track_id]['bbox'] = dets[det_idx][:4]
                self.tracks[track_id]['score'] = dets[det_idx][4]
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                
                # Update trajectory
                center = ((dets[det_idx][0] + dets[det_idx][2]) / 2, 
                          (dets[det_idx][1] + dets[det_idx][3]) / 2)
                self.tracks[track_id]['trajectory'].append(center)
            else:
                # Increment age for unmatched tracks
                self.tracks[track_ids[track_idx]]['age'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in range(len(dets)):
            if det_idx not in det_indices:
                self.tracks[self.next_id] = {
                    'bbox': dets[det_idx][:4],
                    'score': dets[det_idx][4],
                    'age': 0,
                    'hits': 1,
                    'trajectory': [((dets[det_idx][0] + dets[det_idx][2]) / 2, 
                                    (dets[det_idx][1] + dets[det_idx][3]) / 2)]
                }
                self.next_id += 1
        
        # Increment age for all tracks not updated
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            # Remove old tracks
            if self.tracks[track_id]['age'] > self.max_age:
                self.tracks.pop(track_id)
        
        return self.tracks
    
    def get_trajectories(self, min_length=5):
        """
        Get all trajectories with at least min_length points
        
        Returns:
            List of trajectories, each being a list of (x, y) coordinates
        """
        trajectories = []
        for track_id, track in self.tracks.items():
            if len(track['trajectory']) >= min_length:
                trajectories.append(track['trajectory'])
        return trajectories
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        # Each bounding box should be [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou 