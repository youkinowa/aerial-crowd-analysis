import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
import os

class DetectionVisualizer:
    """Visualization tools for object detection and crowd analysis"""
    
    def __init__(self, class_names=None, colors=None):
        """
        Initialize visualizer
        
        Args:
            class_names: List of class names
            colors: Dictionary of colors for different elements
        """
        # Default class names if not provided
        self.class_names = class_names or ['person']
        
        # Default colors if not provided
        self.colors = colors or {
            'bbox': (0, 255, 0),           # Green for bounding boxes
            'trajectory': (255, 0, 0),     # Red for trajectories
            'hotspot': (0, 0, 255),        # Blue for hotspots
            'flow': (255, 255, 0),         # Yellow for flow vectors
            'text_bg': (0, 0, 0),          # Black for text background
            'text': (255, 255, 255)        # White for text
        }
        
        # Create colormap for heatmap visualization
        self.cmap = cm.jet
        self.norm = Normalize(vmin=0, vmax=1)
    
    def draw_detections(self, frame, detections, show_conf=True, thickness=2):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections [x1, y1, x2, y2, conf, class_id]
            show_conf: Whether to show confidence scores
            thickness: Line thickness
            
        Returns:
            Frame with detections drawn
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4]
            class_id = int(det[5]) if len(det) > 5 else 0
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), self.colors['bbox'], thickness)
            
            # Draw label with confidence if requested
            if show_conf:
                label = f"{class_name}: {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_frame, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 5), self.colors['text_bg'], -1)
                cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        return vis_frame
    
    def draw_trajectories(self, frame, tracks, draw_ids=True, min_length=2, thickness=2):
        """
        Draw tracking trajectories on frame
        
        Args:
            frame: Input frame
            tracks: Dictionary of tracks {id: track_data}
            draw_ids: Whether to draw track IDs
            min_length: Minimum trajectory length to draw
            thickness: Line thickness
            
        Returns:
            Frame with trajectories drawn
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        for track_id, track in tracks.items():
            # Skip trajectories that are too short
            if len(track['trajectory']) < min_length:
                continue
            
            # Convert trajectory points to integer coordinates
            points = np.array(track['trajectory'], dtype=np.int32)
            
            # Draw trajectory path
            cv2.polylines(vis_frame, [points], False, self.colors['trajectory'], thickness)
            
            # Draw ID if requested
            if draw_ids:
                last_point = track['trajectory'][-1]
                cv2.putText(vis_frame, str(track_id), (int(last_point[0]), int(last_point[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['trajectory'], 2)
        
        return vis_frame
    
    def draw_density_map(self, frame, density_map, alpha=0.6):
        """
        Draw crowd density heatmap on frame
        
        Args:
            frame: Input frame
            density_map: 2D array of density values [0-1]
            alpha: Transparency of the heatmap (0-1)
            
        Returns:
            Frame with density heatmap overlay
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Resize density map to match frame dimensions if needed
        if density_map.shape[:2] != frame.shape[:2]:
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
        
        # Create RGB colormap from density map
        colored_map = self.cmap(self.norm(density_map))
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        colored_map_bgr = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
        
        # Create mask for areas with density above threshold
        mask = density_map > 0.1
        
        # Blend heatmap with original frame
        vis_frame = np.where(mask[:, :, np.newaxis], 
                            cv2.addWeighted(vis_frame, 1-alpha, colored_map_bgr, alpha, 0),
                            vis_frame)
        
        return vis_frame
    
    def draw_crowd_flow(self, frame, flow_data, scale=20, thickness=2, min_magnitude=5):
        """
        Draw crowd flow vectors on frame
        
        Args:
            frame: Input frame
            flow_data: Dictionary with flow data including 'dominant_vector'
            scale: Scale factor for vector visualization
            thickness: Line thickness
            min_magnitude: Minimum vector magnitude to draw
            
        Returns:
            Frame with flow vectors drawn
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Check if flow data available
        if not flow_data or 'dominant_vector' not in flow_data:
            return vis_frame
        
        # Get dominant flow vector
        flow_vector = flow_data['dominant_vector']
        magnitude = flow_data.get('magnitude', np.linalg.norm(flow_vector))
        
        # Skip if magnitude is too small
        if magnitude < min_magnitude:
            return vis_frame
        
        # Draw flow vector from center of the frame
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        end_x = int(center_x + flow_vector[0] * scale)
        end_y = int(center_y + flow_vector[1] * scale)
        
        # Draw arrow
        cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), 
                      self.colors['flow'], thickness, tipLength=0.3)
        
        # Add flow info
        flow_text = f"Flow: {magnitude:.1f}, Dir: {np.degrees(flow_data.get('dominant_angle', 0)):.0f}°"
        cv2.putText(vis_frame, flow_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.6, self.colors['text_bg'], 4)
        cv2.putText(vis_frame, flow_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.6, self.colors['text'], 1)
        
        return vis_frame
    
    def draw_threat_assessment(self, frame, threat_assessment):
        """
        Draw threat assessment information on frame
        
        Args:
            frame: Input frame
            threat_assessment: Dictionary with threat assessment results
            
        Returns:
            Frame with threat assessment visualization
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Check if threat assessment available
        if not threat_assessment:
            return vis_frame
        
        # Get threat level and pattern
        threat_level = threat_assessment.get('threat_level', 0)
        behavior_pattern = threat_assessment.get('behavior_pattern', 'normal')
        
        # Color based on threat level (green to red)
        color = (
            0,  # B
            max(0, min(255, int(255 * (1 - threat_level)))),  # G
            max(0, min(255, int(255 * threat_level)))  # R
        )
        
        # Draw threat gauge at top of frame
        h, w = frame.shape[:2]
        gauge_width = int(w * threat_level)
        cv2.rectangle(vis_frame, (0, 0), (gauge_width, 20), color, -1)
        cv2.rectangle(vis_frame, (0, 0), (w, 20), (255, 255, 255), 1)
        
        # Add threat info text
        threat_text = f"Threat: {threat_level:.2f} | Pattern: {behavior_pattern}"
        cv2.putText(vis_frame, threat_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw hotspots
        for hotspot in threat_assessment.get('hotspots', []):
            bbox = hotspot['bbox']
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors['hotspot'], 2)
            cv2.putText(vis_frame, f"Hotspot", (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['hotspot'], 2)
        
        # Draw threat factors if available
        factors = threat_assessment.get('factors', {})
        if factors:
            y_pos = 40
            for factor, score in factors.items():
                if score > 0.1:  # Only show significant factors
                    factor_color = (
                        0,  # B
                        max(0, min(255, int(255 * (1 - score)))),  # G
                        max(0, min(255, int(255 * score)))  # R
                    )
                    factor_text = f"{factor}: {score:.2f}"
                    cv2.putText(vis_frame, factor_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, factor_color, 1)
                    y_pos += 20
        
        return vis_frame
    
    def draw_alert(self, frame, alert):
        """
        Draw alert information on frame
        
        Args:
            frame: Input frame
            alert: Dictionary with alert information
            
        Returns:
            Frame with alert visualization
        """
        # Make a copy of frame to avoid modifying the original
        vis_frame = frame.copy()
        
        # Check if alert available
        if not alert:
            return vis_frame
        
        # Get alert level and description
        level = alert.get('level', 'none').upper()
        description = alert.get('description', '')
        
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
        cv2.putText(vis_frame, description, (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_frame
    
    def visualize_results(self, frame, detections=None, tracks=None, density_map=None, 
                        flow_data=None, threat_assessment=None, alert=None):
        """
        Comprehensive visualization of detection and analysis results
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: Dictionary of tracks
            density_map: Crowd density map
            flow_data: Flow analysis data
            threat_assessment: Threat assessment results
            alert: Alert information
            
        Returns:
            Visualized frame with all elements
        """
        # Start with a copy of the original frame
        vis_frame = frame.copy()
        
        # Apply visualizations in order
        if density_map is not None:
            vis_frame = self.draw_density_map(vis_frame, density_map)
        
        if detections:
            vis_frame = self.draw_detections(vis_frame, detections)
        
        if tracks:
            vis_frame = self.draw_trajectories(vis_frame, tracks)
        
        if flow_data:
            vis_frame = self.draw_crowd_flow(vis_frame, flow_data)
        
        if threat_assessment:
            vis_frame = self.draw_threat_assessment(vis_frame, threat_assessment)
        
        if alert:
            vis_frame = self.draw_alert(vis_frame, alert)
        
        return vis_frame

class ResultsExporter:
    """Export visualization results to various formats"""
    
    def __init__(self, output_dir='output'):
        """
        Initialize exporter
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_frame(self, frame, filename):
        """Save a single frame to an image file"""
        full_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(full_path, frame)
        return full_path
    
    def create_density_heatmap(self, density_map, filename=None, show=False):
        """Create and optionally save a density heatmap visualization"""
        plt.figure(figsize=(10, 8))
        plt.imshow(density_map, cmap='jet')
        plt.colorbar(label='Density')
        plt.title('Crowd Density Heatmap')
        
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trajectory_analysis(self, trajectories, hmm_states=None, filename=None, show=False):
        """Plot trajectory analysis with optional HMM state coloring"""
        plt.figure(figsize=(12, 10))
        
        # Plot each trajectory
        for i, traj in enumerate(trajectories):
            traj = np.array(traj)
            
            if hmm_states is not None and i < len(hmm_states):
                # Color by state if available
                states = hmm_states[i]
                for j in range(len(traj) - 1):
                    if j < len(states):
                        state = states[j]
                        color = plt.cm.tab10(state % 10)
                        plt.plot(traj[j:j+2, 0], traj[j:j+2, 1], color=color, linewidth=2)
            else:
                # Otherwise use a simple color scheme
                plt.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.7, linewidth=1)
            
            # Mark start and end points
            plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=6)  # Start point
            plt.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=6)  # End point
        
        plt.title('Trajectory Analysis')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_threat_history(self, threat_history, filename=None, show=False):
        """Plot history of threat levels over time"""
        if not threat_history:
            return
        
        # Extract timestamps and threat levels
        timestamps = [entry['timestamp'] for entry in threat_history]
        threat_levels = [entry['threat_level'] for entry in threat_history]
        
        # Convert timestamps to datetime objects
        import datetime
        datetime_objects = [datetime.datetime.fromisoformat(ts) for ts in timestamps]
        
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_objects, threat_levels, 'r-', linewidth=2)
        plt.fill_between(datetime_objects, threat_levels, alpha=0.2, color='red')
        plt.title('Threat Level History')
        plt.xlabel('Time')
        plt.ylabel('Threat Level')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add colored regions for different threat levels
        plt.axhspan(0, 0.3, alpha=0.2, color='green', label='Low')
        plt.axhspan(0.3, 0.5, alpha=0.2, color='yellow', label='Medium')
        plt.axhspan(0.5, 0.7, alpha=0.2, color='orange', label='High')
        plt.axhspan(0.7, 1.0, alpha=0.2, color='red', label='Critical')
        
        plt.legend()
        
        # Format x-axis to show appropriate time format
        plt.gcf().autofmt_xdate()
        
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_to_video(self, frames, output_filename, fps=30):
        """Export a sequence of frames to a video file"""
        if not frames:
            return None
        
        # Get dimensions from first frame
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        full_path = os.path.join(self.output_dir, output_filename)
        out = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        # Release writer
        out.release()
        
        return full_path 