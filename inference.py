import os
import cv2
import torch
import numpy as np
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

# Import model components
from models.detection.backbone import load_enhanced_yolov8
from models.detection.preprocessing import LightingRobustPreprocessor
from models.detection.domain_adapt import LightingAdaptationCycleGAN
from models.behavior.hmm_model import CrowdBehaviorHMM, MultiPersonTracker
from models.behavior.threat_detection import ThreatAssessmentSystem, RiskAssessmentEngine

class AerialCrowdAnalysisPipeline:
    """End-to-end pipeline for aerial crowd analysis"""
    
    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration
        
        Args:
            config: Dictionary or path to JSON configuration
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize detection model
        self.detector = self._init_detector()
        
        # Initialize preprocessing
        self.preprocessor = LightingRobustPreprocessor()
        
        # Initialize domain adaptation (if enabled)
        if self.config.get('use_domain_adaptation', False):
            self.domain_adapter = LightingAdaptationCycleGAN(device=self.config['device'])
            if os.path.exists(self.config.get('domain_adapt_weights', '')):
                self.domain_adapter.load_models(self.config['domain_adapt_weights'])
        else:
            self.domain_adapter = None
        
        # Initialize tracking system
        self.tracker = MultiPersonTracker(
            max_age=self.config.get('tracker_max_age', 30),
            min_hits=self.config.get('tracker_min_hits', 3),
            iou_threshold=self.config.get('tracker_iou_threshold', 0.3)
        )
        
        # Initialize behavior analysis
        self._init_behavior_analysis()
        
        # Set up visualization
        self.visualize = self.config.get('visualize', True)
        self.save_output = self.config.get('save_output', False)
        
        if self.save_output:
            os.makedirs(self.config.get('output_dir', 'output'), exist_ok=True)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detection_time': 0,
            'tracking_time': 0,
            'behavior_time': 0,
            'visualization_time': 0,
            'total_time': 0,
            'alerts': []
        }
    
    def _load_config(self, config):
        """Load configuration from file or dict"""
        default_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'detection_model': 'yolov8n.pt',
            'detection_weights': None,
            'detection_confidence': 0.3,
            'detection_iou_threshold': 0.45,
            'use_domain_adaptation': False,
            'domain_adapt_weights': None,
            'hmm_model_path': None,
            'visualize': True,
            'save_output': False,
            'output_dir': 'output',
            'log_file': 'drone_analysis.log'
        }
        
        if config is None:
            return default_config
            
        if isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        elif isinstance(config, dict):
            default_config.update(config)
        
        return default_config
    
    def _init_detector(self):
        """Initialize object detection model"""
        # Load enhanced YOLOv8 model
        print(f"Loading detection model on {self.config['device']}...")
        if self.config.get('detection_weights'):
            detector = load_enhanced_yolov8(self.config['detection_weights'], device=self.config['device'])
        else:
            detector = load_enhanced_yolov8(device=self.config['device'])
        
        detector.eval()
        return detector
    
    def _init_behavior_analysis(self):
        """Initialize behavior analysis models"""
        # Load or create HMM model
        if self.config.get('hmm_model_path') and os.path.exists(self.config['hmm_model_path']):
            print(f"Loading HMM model from {self.config['hmm_model_path']}...")
            self.hmm = CrowdBehaviorHMM().load_model(self.config['hmm_model_path'])
        else:
            print("Creating new HMM model...")
            self.hmm = CrowdBehaviorHMM()
        
        # Initialize threat assessment
        self.threat_system = ThreatAssessmentSystem(hmm_model_path=self.config.get('hmm_model_path'))
        
        # Initialize risk assessment
        self.risk_engine = RiskAssessmentEngine(self.threat_system)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for detection"""
        # Apply lighting enhancement
        enhanced_frame = self.preprocessor.preprocess(frame, apply_augmentation=False)
        
        # Apply domain adaptation if enabled
        if self.domain_adapter is not None:
            enhanced_frame = self.domain_adapter.transfer_lighting(enhanced_frame, target_domain='normal')
        
        return enhanced_frame
    
    def detect_objects(self, frame):
        """Detect objects in the frame"""
        # Preprocess frame
        preprocessed = self.preprocess_frame(frame)
        
        # Convert to tensor if needed
        if not isinstance(preprocessed, torch.Tensor):
            # Normalize to [0, 1]
            if preprocessed.max() > 1.0:
                preprocessed = preprocessed / 255.0
                
            preprocessed = torch.from_numpy(preprocessed).permute(2, 0, 1).float()
            
        # Add batch dimension if needed
        if preprocessed.dim() == 3:
            preprocessed = preprocessed.unsqueeze(0)
            
        # Move to device
        preprocessed = preprocessed.to(self.config['device'])
        
        # Run detection
        with torch.no_grad():
            detections = self.detector(preprocessed)
            
        # Process results
        results = []
        for i, det in enumerate(detections[0]):  # Assuming batch size 1
            *xyxy, conf, cls = det.cpu().numpy()
            if conf >= self.config['detection_confidence']:
                results.append([*xyxy, conf, cls])
        
        return results
    
    def update_tracks(self, detections):
        """Update tracking with new detections"""
        tracks = self.tracker.update(detections)
        return tracks
    
    def analyze_behavior(self, tracks, detections, frame_shape):
        """Analyze behavior using HMM and threat assessment"""
        # Extract trajectories from tracks
        trajectories = self.tracker.get_trajectories(min_length=5)
        
        # Assess threat level
        threat_assessment = self.threat_system.assess_threat_level(
            trajectories, detections, frame_shape
        )
        
        # Generate alerts if needed
        alert = self.risk_engine.generate_alert(threat_assessment)
        if alert:
            self.stats['alerts'].append(alert)
            print(f"ALERT: {alert['level'].upper()} - {alert['description']}")
        
        return threat_assessment, alert
    
    def visualize_results(self, frame, detections, tracks, threat_assessment=None, alert=None):
        """Visualize detection, tracking and threat assessment results"""
        if not self.visualize:
            return frame
        
        # Create copy of frame for visualization
        vis_frame = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Convert to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_frame, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 5), (0, 255, 0), -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw tracking trajectories
        for track_id, track in tracks.items():
            # Draw trajectory
            if len(track['trajectory']) > 1:
                points = np.array(track['trajectory'], dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, (255, 0, 0), 2)
                
                # Draw ID
                last_point = track['trajectory'][-1]
                cv2.putText(vis_frame, str(track_id), (int(last_point[0]), int(last_point[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display threat assessment
        if threat_assessment:
            threat_level = threat_assessment.get('threat_level', 0)
            threat_pattern = threat_assessment.get('behavior_pattern', 'normal')
            
            # Color based on threat level (green to red)
            color = (0, 255 - int(255 * threat_level), int(255 * (1 - threat_level)))
            
            # Draw threat gauge at top of frame
            h, w = vis_frame.shape[:2]
            gauge_width = int(w * threat_level)
            cv2.rectangle(vis_frame, (0, 0), (gauge_width, 20), color, -1)
            cv2.rectangle(vis_frame, (0, 0), (w, 20), (255, 255, 255), 1)
            
            # Add threat info
            threat_text = f"Threat: {threat_level:.2f} | Pattern: {threat_pattern}"
            cv2.putText(vis_frame, threat_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Visualize hotspots
            for hotspot in threat_assessment.get('hotspots', []):
                bbox = hotspot['bbox']
                cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(vis_frame, f"Hotspot", (bbox[0], bbox[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display alert if any
        if alert:
            level = alert['level'].upper()
            desc = alert['description']
            
            # Choose color based on alert level
            if level == 'CRITICAL':
                color = (0, 0, 255)  # Red
            elif level == 'HIGH':
                color = (0, 69, 255)  # Orange
            elif level == 'MEDIUM':
                color = (0, 215, 255)  # Yellow
            else:
                color = (0, 255, 128)  # Light green
            
            # Draw alert box
            cv2.rectangle(vis_frame, (0, 25), (vis_frame.shape[1], 65), color, -1)
            cv2.putText(vis_frame, f"ALERT: {level}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, desc, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_frame
    
    def process_frame(self, frame):
        """Process a single frame through the entire pipeline"""
        start_time = time.time()
        
        # Detect objects
        detect_start = time.time()
        detections = self.detect_objects(frame)
        detect_time = time.time() - detect_start
        
        # Update tracking
        track_start = time.time()
        tracks = self.update_tracks(detections)
        track_time = time.time() - track_start
        
        # Analyze behavior
        behavior_start = time.time()
        threat_assessment, alert = self.analyze_behavior(tracks, detections, frame.shape)
        behavior_time = time.time() - behavior_start
        
        # Visualize results
        vis_start = time.time()
        vis_frame = self.visualize_results(frame, detections, tracks, threat_assessment, alert)
        vis_time = time.time() - vis_start
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['detection_time'] += detect_time
        self.stats['tracking_time'] += track_time
        self.stats['behavior_time'] += behavior_time
        self.stats['visualization_time'] += vis_time
        
        total_time = time.time() - start_time
        self.stats['total_time'] += total_time
        
        # Log performance
        fps = 1.0 / total_time
        print(f"Frame {self.stats['frames_processed']} | "
              f"FPS: {fps:.1f} | "
              f"Det: {detect_time*1000:.1f}ms | "
              f"Track: {track_time*1000:.1f}ms | "
              f"Behavior: {behavior_time*1000:.1f}ms")
        
        result = {
            'frame': vis_frame if self.visualize else frame,
            'detections': detections,
            'tracks': tracks,
            'threat_assessment': threat_assessment,
            'alert': alert,
            'fps': fps
        }
        
        return result
    
    def process_video(self, input_path, output_path=None):
        """Process a video file"""
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return
        
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        
        # Set up video writer if needed
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Save output if requested
            if output_path:
                out.write(result['frame'])
            
            # Display frame if requested
            if self.visualize:
                cv2.imshow('Aerial Crowd Analysis', result['frame'])
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        avg_fps = self.stats['frames_processed'] / max(self.stats['total_time'], 0.001)
        print(f"\nProcessed {self.stats['frames_processed']} frames in "
              f"{self.stats['total_time']:.1f}s ({avg_fps:.1f} FPS)")
        print(f"Alerts generated: {len(self.stats['alerts'])}")
        
        return self.stats
    
    def process_stream(self, source=0, output_path=None):
        """Process a live video stream"""
        # Open camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open camera/stream")
            return
        
        # Get video info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing stream from source: {source}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        # Set up video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{output_path}/stream_{timestamp}.mp4"
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Process frames
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Save output if requested
                if out:
                    out.write(result['frame'])
                
                # Display frame
                cv2.imshow('Aerial Crowd Analysis', result['frame'])
                
                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Stream processing interrupted by user")
        
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        avg_fps = self.stats['frames_processed'] / max(self.stats['total_time'], 0.001)
        print(f"\nProcessed {self.stats['frames_processed']} frames in "
              f"{self.stats['total_time']:.1f}s ({avg_fps:.1f} FPS)")
        print(f"Alerts generated: {len(self.stats['alerts'])}")
        
        return self.stats
    
    def save_hmm_model(self, output_path):
        """Save the trained HMM model"""
        self.hmm.save_model(output_path)
        print(f"HMM model saved to: {output_path}")

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description='Aerial Crowd Analysis Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None, help='Path to input video file')
    parser.add_argument('--output', type=str, default=None, help='Path to output video file')
    parser.add_argument('--stream', type=int, default=None, help='Camera source for live stream')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save HMM model')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AerialCrowdAnalysisPipeline(args.config)
    
    # Process video or stream
    if args.input:
        pipeline.process_video(args.input, args.output)
    elif args.stream is not None:
        pipeline.process_stream(args.stream, args.output)
    else:
        print("Error: No input source specified. Use --input or --stream.")
        return
    
    # Save HMM model if requested
    if args.save_model:
        pipeline.save_hmm_model(args.save_model)

if __name__ == '__main__':
    main() 