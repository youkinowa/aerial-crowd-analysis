import numpy as np
import torch
import datetime
from collections import defaultdict, deque
from .hmm_model import CrowdBehaviorHMM
import math
import cv2

class ThreatAssessmentSystem:
    """System for assessing threats based on crowd behavior analysis"""
    
    def __init__(self, hmm_model_path=None, threat_thresholds=None):
        """
        Args:
            hmm_model_path: Path to pre-trained HMM model, if available
            threat_thresholds: Dictionary of thresholds for different threat types
        """
        # Load or create HMM model
        if hmm_model_path:
            self.hmm = CrowdBehaviorHMM().load_model(hmm_model_path)
        else:
            # Default model with 5 states and 4 features
            self.hmm = CrowdBehaviorHMM(n_states=5, n_features=4)
        
        # Set default threat thresholds if not provided
        self.threat_thresholds = threat_thresholds or {
            'unusual_movement': 0.7,       # Threshold for unusual movement patterns
            'abnormal_density': 0.8,       # Threshold for abnormal crowd density
            'erratic_behavior': 0.6,       # Threshold for erratic state changes
            'reverse_flow': 0.5,           # Threshold for counter-flow movement
            'rapid_dispersion': 0.7,       # Threshold for sudden crowd dispersion
            'static_congregation': 0.8     # Threshold for unusual static congregation
        }
        
        # Initialize threat history
        self.threat_history = deque(maxlen=100)
        
        # Track current scene state
        self.current_state = {
            'threat_level': 0.0,           # Overall threat level (0-1)
            'threat_factors': {},          # Specific threat factors and their scores
            'crowd_count': 0,              # Number of people detected
            'crowd_density': 0.0,          # Average crowd density
            'crowd_flow': None,            # Dominant movement vector
            'hotspots': []                 # Areas of high density or unusual behavior
        }
        
        # Movement analysis
        self.flow_history = deque(maxlen=10)  # Store recent flow vectors
        
        # Density estimation
        self.density_map = None
        self.density_history = deque(maxlen=30)  # Store density maps over time
    
    def update_density_map(self, detections, frame_shape):
        """
        Update crowd density map based on current detections
        
        Args:
            detections: List of bounding boxes [x1, y1, x2, y2, confidence, class_id]
            frame_shape: Tuple of (height, width) of the frame
        """
        height, width = frame_shape[:2]
        
        # Initialize or reset density map
        self.density_map = np.zeros((height, width), dtype=np.float32)
        
        # Convert detections to the format: [x1, y1, x2, y2, confidence]
        dets = []
        for det in detections:
            if len(det) >= 5 and det[5] == 0:  # Only consider 'person' class (class_id=0)
                dets.append(det[:5])
        
        # Update density map based on detections
        for det in dets:
            x1, y1, x2, y2 = map(int, det[:4])
            confidence = det[4]
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            # Add weighted gaussian blob at detection location
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            sigma = max(1, min(width, height) // 30)  # Scale sigma based on frame size
            
            # Generate Gaussian kernel
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            kernel = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            kernel = kernel * confidence  # Weight by detection confidence
            
            # Add to density map
            self.density_map += kernel
        
        # Normalize density map to [0, 1]
        if np.max(self.density_map) > 0:
            self.density_map /= np.max(self.density_map)
        
        # Store in history
        self.density_history.append(self.density_map.copy())
        
        # Calculate current average crowd density
        self.current_state['crowd_density'] = np.mean(self.density_map)
        self.current_state['crowd_count'] = len(dets)
    
    def analyze_crowd_flow(self, trajectories):
        """
        Analyze crowd flow patterns from trajectories
        
        Args:
            trajectories: List of trajectories, each being a list of (x, y) coordinates
        
        Returns:
            Flow analysis results
        """
        # Skip if not enough trajectories
        if not trajectories or len(trajectories) < 5:
            return None
        
        # Calculate flow vectors for each trajectory
        flow_vectors = []
        for trajectory in trajectories:
            if len(trajectory) < 2:
                continue
                
            # Calculate overall displacement vector
            start = np.array(trajectory[0])
            end = np.array(trajectory[-1])
            displacement = end - start
            
            # Calculate distance and angle
            distance = np.linalg.norm(displacement)
            angle = np.arctan2(displacement[1], displacement[0])
            
            # Skip negligible movement
            if distance < 5:
                continue
                
            flow_vectors.append((distance, angle, displacement))
        
        if not flow_vectors:
            return None
            
        # Calculate dominant flow
        distances = np.array([v[0] for v in flow_vectors])
        angles = np.array([v[1] for v in flow_vectors])
        displacements = np.array([v[2] for v in flow_vectors])
        
        # Weight by distance
        weights = distances / np.sum(distances)
        
        # Calculate weighted average displacement
        avg_displacement = np.sum(displacements * weights[:, np.newaxis], axis=0)
        
        # Calculate consistency of flow (0 = chaotic, 1 = uniform)
        angle_diff = np.abs(angles - np.mean(angles))
        angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # Handle angle wrapping
        flow_consistency = 1.0 - np.mean(angle_diff) / np.pi
        
        # Store flow data
        flow_data = {
            'dominant_vector': avg_displacement,
            'dominant_angle': np.arctan2(avg_displacement[1], avg_displacement[0]),
            'magnitude': np.linalg.norm(avg_displacement),
            'consistency': flow_consistency,
            'vector_count': len(flow_vectors)
        }
        
        self.flow_history.append(flow_data)
        self.current_state['crowd_flow'] = flow_data
        
        return flow_data
    
    def detect_reverse_flow(self, trajectories, reference_direction=None):
        """
        Detect people moving against the dominant flow
        
        Args:
            trajectories: List of trajectories
            reference_direction: Optional reference direction (radians)
            
        Returns:
            List of trajectories moving against the flow
        """
        if not trajectories or len(trajectories) < 5:
            return []
            
        # If no reference provided, use the dominant flow from history
        if reference_direction is None and self.flow_history:
            reference_direction = self.flow_history[-1]['dominant_angle']
        elif reference_direction is None:
            # Calculate reference from current trajectories
            displacements = []
            for trajectory in trajectories:
                if len(trajectory) < 2:
                    continue
                start = np.array(trajectory[0])
                end = np.array(trajectory[-1])
                displacements.append(end - start)
            
            if not displacements:
                return []
                
            avg_displacement = np.mean(displacements, axis=0)
            reference_direction = np.arctan2(avg_displacement[1], avg_displacement[0])
        
        # Detect reverse flow trajectories
        reverse_trajectories = []
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) < 2:
                continue
                
            # Calculate trajectory angle
            start = np.array(trajectory[0])
            end = np.array(trajectory[-1])
            displacement = end - start
            
            # Skip negligible movement
            if np.linalg.norm(displacement) < 5:
                continue
                
            angle = np.arctan2(displacement[1], displacement[0])
            
            # Calculate angular difference (accounting for wrapping)
            diff = abs(angle - reference_direction)
            if diff > np.pi:
                diff = 2*np.pi - diff
                
            # If moving in opposite direction (diff > 90 degrees)
            if diff > np.pi/2:
                reverse_trajectories.append((i, trajectory, diff))
        
        return reverse_trajectories
    
    def detect_abnormal_congregation(self, threshold=0.7):
        """
        Detect unusual static congregation in the crowd
        
        Args:
            threshold: Density threshold for congregation
            
        Returns:
            List of hotspot regions
        """
        if self.density_map is None or not self.density_history:
            return []
            
        # Calculate average density map over recent history
        if len(self.density_history) > 5:
            avg_density = np.mean(list(self.density_history)[-5:], axis=0)
        else:
            avg_density = self.density_map.copy()
        
        # Threshold the density map
        high_density = avg_density > threshold
        
        # Find connected components (congregations)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            high_density.astype(np.uint8), connectivity=8)
        
        # Filter components by size
        min_size = 100  # Minimum pixel area to be considered a congregation
        congregations = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            size = stats[i, cv2.CC_STAT_AREA]
            if size < min_size:
                continue
                
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            # Calculate density value for this congregation
            density_value = np.mean(avg_density[y:y+h, x:x+w])
            
            congregations.append({
                'bbox': (x, y, x+w, y+h),
                'center': (cx, cy),
                'size': size,
                'density': density_value
            })
        
        return congregations
    
    def assess_threat_level(self, trajectories, detections, frame_shape):
        """
        Main function to assess threat level based on all available data
        
        Args:
            trajectories: List of trajectories
            detections: List of detection bounding boxes
            frame_shape: Shape of the video frame
            
        Returns:
            Threat assessment result
        """
        # Update density map
        self.update_density_map(detections, frame_shape)
        
        # Analyze crowd flow
        flow_data = self.analyze_crowd_flow(trajectories)
        
        # Process trajectories with HMM
        features = self.hmm.preprocess_trajectories(trajectories)
        
        # Add density features if available
        if self.density_map is not None and features.size > 0:
            features = self.hmm.add_crowd_density_feature(features, self.density_map)
        
        # Skip further analysis if not enough data
        if features.size == 0:
            return {
                'threat_level': 0.0,
                'factors': {},
                'timestamp': datetime.datetime.now().isoformat()
            }
        
        # Detect anomalies using HMM
        anomaly_results = self.hmm.detect_anomalies(features)
        
        # Analyze behavior pattern
        behavior_analysis = self.hmm.analyze_behavior_pattern(anomaly_results['state_sequence'])
        
        # Detect reverse flow
        reverse_flow = self.detect_reverse_flow(trajectories)
        
        # Detect abnormal congregation
        congregations = self.detect_abnormal_congregation()
        
        # Initialize threat factors
        threat_factors = {}
        
        # Assess unusual movement
        if anomaly_results['is_anomalous']:
            threat_factors['unusual_movement'] = min(1.0, max(0.0, 
                abs(anomaly_results['log_likelihood_threshold'] - anomaly_results['log_likelihood']) / 
                abs(anomaly_results['log_likelihood_threshold'])))
        else:
            threat_factors['unusual_movement'] = 0.0
        
        # Assess erratic behavior
        if behavior_analysis['pattern'] == 'erratic':
            threat_factors['erratic_behavior'] = min(1.0, behavior_analysis['change_rate'] * 1.5)
        else:
            threat_factors['erratic_behavior'] = 0.0
        
        # Assess reverse flow
        if reverse_flow:
            reverse_percentage = len(reverse_flow) / len(trajectories)
            threat_factors['reverse_flow'] = min(1.0, reverse_percentage * 2.0)  # Scale up for sensitivity
        else:
            threat_factors['reverse_flow'] = 0.0
        
        # Assess abnormal density
        if congregations:
            max_density = max([c['density'] for c in congregations])
            threat_factors['abnormal_density'] = min(1.0, max_density * 1.2)  # Scale up for sensitivity
        else:
            threat_factors['abnormal_density'] = 0.0
        
        # Assess rapid dispersion
        if flow_data and 'consistency' in flow_data:
            if flow_data['consistency'] < 0.3 and flow_data['magnitude'] > 20:
                threat_factors['rapid_dispersion'] = min(1.0, (1.0 - flow_data['consistency']) * 
                                                      min(1.0, flow_data['magnitude'] / 50.0))
            else:
                threat_factors['rapid_dispersion'] = 0.0
        else:
            threat_factors['rapid_dispersion'] = 0.0
        
        # Assess static congregation
        if congregations and behavior_analysis['pattern'] == 'stationary':
            congregation_size = sum([c['size'] for c in congregations])
            frame_size = frame_shape[0] * frame_shape[1]
            threat_factors['static_congregation'] = min(1.0, (congregation_size / frame_size) * 10.0)
        else:
            threat_factors['static_congregation'] = 0.0
        
        # Calculate overall threat level as weighted average
        weights = {
            'unusual_movement': 0.2,
            'erratic_behavior': 0.2,
            'reverse_flow': 0.15,
            'abnormal_density': 0.15,
            'rapid_dispersion': 0.2,
            'static_congregation': 0.1
        }
        
        overall_threat = 0.0
        for factor, score in threat_factors.items():
            weight = weights.get(factor, 0.1)
            if score >= self.threat_thresholds.get(factor, 0.5):  # Only count if above threshold
                overall_threat += score * weight
        
        # Normalize to [0, 1]
        overall_threat = min(1.0, overall_threat)
        
        # Update current state
        self.current_state['threat_level'] = overall_threat
        self.current_state['threat_factors'] = threat_factors
        self.current_state['hotspots'] = congregations
        
        # Add to history
        self.threat_history.append({
            'threat_level': overall_threat,
            'factors': threat_factors,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Update HMM event history if significant threat
        if overall_threat > 0.7:
            self.hmm.record_event(
                event_type='high_threat',
                timestamp=datetime.datetime.now().isoformat(),
                location='frame',
                description=f"High threat level: {overall_threat:.2f}"
            )
        
        return {
            'threat_level': overall_threat,
            'factors': threat_factors,
            'hotspots': congregations,
            'reverse_flow': reverse_flow,
            'behavior_pattern': behavior_analysis['pattern'],
            'timestamp': datetime.datetime.now().isoformat()
        }

class RiskAssessmentEngine:
    """Engine for risk assessment and alert generation"""
    
    def __init__(self, threat_system, alert_thresholds=None):
        """
        Args:
            threat_system: ThreatAssessmentSystem instance
            alert_thresholds: Dictionary of thresholds for different alert levels
        """
        self.threat_system = threat_system
        
        # Set default alert thresholds if not provided
        self.alert_thresholds = alert_thresholds or {
            'low': 0.3,      # Low risk
            'medium': 0.5,   # Medium risk
            'high': 0.7,     # High risk
            'critical': 0.9  # Critical risk
        }
        
        # Initialize alert state
        self.current_alert_level = 'none'
        self.alert_history = deque(maxlen=100)
        self.alert_cooldown = 0  # Cooldown timer to prevent alert spam
    
    def generate_alert(self, threat_assessment):
        """
        Generate appropriate alerts based on threat assessment
        
        Args:
            threat_assessment: Result from threat_system.assess_threat_level()
            
        Returns:
            Alert information if generated, None otherwise
        """
        # Check cooldown
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1
            return None
        
        threat_level = threat_assessment['threat_level']
        
        # Determine alert level
        if threat_level >= self.alert_thresholds['critical']:
            alert_level = 'critical'
        elif threat_level >= self.alert_thresholds['high']:
            alert_level = 'high'
        elif threat_level >= self.alert_thresholds['medium']:
            alert_level = 'medium'
        elif threat_level >= self.alert_thresholds['low']:
            alert_level = 'low'
        else:
            alert_level = 'none'
        
        # Only generate alert if level increased
        if self._should_generate_alert(alert_level):
            alert = {
                'level': alert_level,
                'threat_level': threat_level,
                'factors': threat_assessment['factors'],
                'timestamp': datetime.datetime.now().isoformat(),
                'description': self._generate_alert_description(threat_assessment)
            }
            
            # Update alert state
            self.current_alert_level = alert_level
            self.alert_history.append(alert)
            
            # Set cooldown based on alert level
            if alert_level == 'critical':
                self.alert_cooldown = 0  # No cooldown for critical alerts
            elif alert_level == 'high':
                self.alert_cooldown = 5
            elif alert_level == 'medium':
                self.alert_cooldown = 10
            else:
                self.alert_cooldown = 20
            
            return alert
        
        return None
    
    def _should_generate_alert(self, new_level):
        """Determine if a new alert should be generated"""
        # Alert level hierarchy
        level_hierarchy = {
            'none': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        
        # Generate alert if level increased
        return level_hierarchy.get(new_level, 0) > level_hierarchy.get(self.current_alert_level, 0)
    
    def _generate_alert_description(self, threat_assessment):
        """Generate human-readable alert description"""
        factors = threat_assessment['factors']
        
        # Find main contributing factors (top 2)
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        main_factors = [f for f, v in sorted_factors[:2] if v > 0.3]
        
        if not main_factors:
            return "Potential security concern detected."
        
        # Generate description based on main factors
        if 'unusual_movement' in main_factors:
            description = "Unusual movement patterns detected in crowd."
        elif 'erratic_behavior' in main_factors:
            description = "Erratic behavior detected in crowd movement."
        elif 'reverse_flow' in main_factors:
            description = "People moving against the main crowd flow."
        elif 'abnormal_density' in main_factors:
            description = "Abnormal crowd density detected."
        elif 'rapid_dispersion' in main_factors:
            description = "Crowd rapidly dispersing from area."
        elif 'static_congregation' in main_factors:
            description = "Unusual static congregation of people."
        else:
            description = "Multiple minor anomalies detected in crowd behavior."
        
        return description 