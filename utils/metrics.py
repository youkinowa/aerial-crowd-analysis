import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment
import json
import os
import datetime

class DetectionEvaluator:
    """Evaluation tools for object detection performance"""
    
    def __init__(self, iou_threshold=0.5, class_names=None):
        """
        Initialize evaluator
        
        Args:
            iou_threshold: IoU threshold for true positive
            class_names: List of class names for evaluation
        """
        self.iou_threshold = iou_threshold
        self.class_names = class_names or ['person']
        
        # Initialize metrics
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mean_iou': 0.0,
            'ious': [],
            'frames_evaluated': 0,
            'total_gt_objects': 0,
            'total_pred_objects': 0,
            'total_matches': 0,
            'class_stats': {}
        }
        
        # Initialize per-class stats
        for class_name in self.class_names:
            self.stats['class_stats'][class_name] = {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # No overlap
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def match_detections(self, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores=None):
        """
        Match ground truth boxes with predicted boxes
        
        Args:
            gt_boxes: Ground truth boxes [N, 4]
            gt_classes: Ground truth class IDs [N]
            pred_boxes: Predicted boxes [M, 4]
            pred_classes: Predicted class IDs [M]
            pred_scores: Predicted confidence scores [M]
            
        Returns:
            Matches, unmatched ground truths, unmatched predictions
        """
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                # Only consider matches with same class
                if gt_classes[i] == pred_classes[j]:
                    iou_matrix[i, j] = self.calculate_iou(gt_box, pred_box)
                else:
                    iou_matrix[i, j] = 0
        
        # Apply Hungarian algorithm to find optimal matching
        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter out matches with low IoU
        matches = []
        unmatched_gt = []
        unmatched_pred = list(range(len(pred_boxes)))
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if iou_matrix[gt_idx, pred_idx] >= self.iou_threshold:
                matches.append((gt_idx, pred_idx))
                if pred_idx in unmatched_pred:
                    unmatched_pred.remove(pred_idx)
            else:
                unmatched_gt.append(gt_idx)
        
        # Add all not matched gt indices
        for idx in range(len(gt_boxes)):
            if idx not in gt_indices:
                unmatched_gt.append(idx)
        
        return matches, unmatched_gt, unmatched_pred
    
    def update(self, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores=None):
        """
        Update metrics with a new set of detections
        
        Args:
            gt_boxes: Ground truth boxes [N, 4]
            gt_classes: Ground truth class IDs [N]
            pred_boxes: Predicted boxes [M, 4]
            pred_classes: Predicted class IDs [M]
            pred_scores: Predicted confidence scores [M]
        """
        # Match detections
        matches, unmatched_gt, unmatched_pred = self.match_detections(
            gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores
        )
        
        # Update overall stats
        self.stats['frames_evaluated'] += 1
        self.stats['total_gt_objects'] += len(gt_boxes)
        self.stats['total_pred_objects'] += len(pred_boxes)
        self.stats['total_matches'] += len(matches)
        
        # Calculate IoUs for matched boxes
        for gt_idx, pred_idx in matches:
            iou = self.calculate_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
            self.stats['ious'].append(iou)
            
            # Increment class-specific true positives
            class_id = gt_classes[gt_idx]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.stats['class_stats'][class_name]['true_positives'] += 1
        
        # Increment false positives for each unmatched prediction
        for pred_idx in unmatched_pred:
            class_id = pred_classes[pred_idx]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.stats['class_stats'][class_name]['false_positives'] += 1
        
        # Increment false negatives for each unmatched ground truth
        for gt_idx in unmatched_gt:
            class_id = gt_classes[gt_idx]
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            self.stats['class_stats'][class_name]['false_negatives'] += 1
        
        # Update totals
        self.stats['true_positives'] = sum(s['true_positives'] for s in self.stats['class_stats'].values())
        self.stats['false_positives'] = sum(s['false_positives'] for s in self.stats['class_stats'].values())
        self.stats['false_negatives'] = sum(s['false_negatives'] for s in self.stats['class_stats'].values())
    
    def compute_metrics(self):
        """Compute final metrics"""
        tp = self.stats['true_positives']
        fp = self.stats['false_positives']
        fn = self.stats['false_negatives']
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate mean IoU
        mean_iou = np.mean(self.stats['ious']) if self.stats['ious'] else 0
        
        # Update overall stats
        self.stats['precision'] = precision
        self.stats['recall'] = recall
        self.stats['f1_score'] = f1_score
        self.stats['mean_iou'] = mean_iou
        
        # Calculate per-class metrics
        for class_name, stats in self.stats['class_stats'].items():
            tp = stats['true_positives']
            fp = stats['false_positives']
            fn = stats['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            stats['precision'] = precision
            stats['recall'] = recall
            stats['f1_score'] = f1_score
        
        return self.stats
    
    def save_results(self, output_file):
        """Save evaluation results to a JSON file"""
        # Compute final metrics
        self.compute_metrics()
        
        # Add timestamp
        self.stats['timestamp'] = datetime.datetime.now().isoformat()
        
        # Convert numpy arrays to lists for JSON serialization
        results = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                  for k, v in self.stats.items()}
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_file

class TrackingEvaluator:
    """Evaluation tools for multi-object tracking performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.stats = {
            'frames': 0,
            'objects': 0,
            'matches': 0,
            'switches': 0,
            'misses': 0,
            'false_positives': 0,
            'mota': 0.0,  # Multi-Object Tracking Accuracy
            'motp': 0.0,  # Multi-Object Tracking Precision
            'mostly_tracked': 0,
            'partially_tracked': 0,
            'mostly_lost': 0,
            'id_switches': 0,
            'fragmentations': 0
        }
        
        # Track history: frame_id -> {gt_id -> pred_id}
        self.track_history = {}
        
        # Distances between matched objects
        self.distances = []
    
    def update(self, frame_id, gt_ids, gt_boxes, pred_ids, pred_boxes):
        """
        Update tracking metrics for a single frame
        
        Args:
            frame_id: Frame identifier
            gt_ids: List of ground truth object IDs
            gt_boxes: List of ground truth bounding boxes
            pred_ids: List of predicted object IDs
            pred_boxes: List of predicted bounding boxes
        """
        # Convert to numpy arrays
        gt_boxes = np.array(gt_boxes)
        pred_boxes = np.array(pred_boxes)
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self._calculate_iou(gt_box, pred_box)
        
        # Apply Hungarian algorithm to find optimal matching
        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        
        # Initialize frame tracking info
        self.track_history[frame_id] = {}
        
        # Process matches
        matches = 0
        switches = 0
        
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            if iou_matrix[gt_idx, pred_idx] > 0.5:  # IoU threshold
                gt_id = gt_ids[gt_idx]
                pred_id = pred_ids[pred_idx]
                
                # Record match
                self.track_history[frame_id][gt_id] = pred_id
                matches += 1
                
                # Calculate distance for MOTP
                self.distances.append(1 - iou_matrix[gt_idx, pred_idx])
                
                # Check for ID switch
                if frame_id > 0 and gt_id in self.track_history.get(frame_id - 1, {}):
                    if self.track_history[frame_id - 1][gt_id] != pred_id:
                        switches += 1
        
        # Update stats
        self.stats['frames'] += 1
        self.stats['objects'] += len(gt_ids)
        self.stats['matches'] += matches
        self.stats['switches'] += switches
        self.stats['misses'] += len(gt_ids) - matches
        self.stats['false_positives'] += len(pred_ids) - matches
    
    def compute_metrics(self):
        """Compute final tracking metrics"""
        # Multi-Object Tracking Accuracy (MOTA)
        total_objects = max(1, self.stats['objects'])
        mota = 1 - (self.stats['misses'] + self.stats['false_positives'] + self.stats['switches']) / total_objects
        self.stats['mota'] = max(0, mota)  # Clamp to [0, 1]
        
        # Multi-Object Tracking Precision (MOTP)
        self.stats['motp'] = 1 - np.mean(self.distances) if self.distances else 0
        
        # ID Switches and Fragmentations
        self.stats['id_switches'] = self.stats['switches']
        
        # Compute fragmentations (track interruptions)
        fragmentations = 0
        
        for gt_id in set().union(*[s.keys() for s in self.track_history.values()]):
            is_tracked = False
            track_fragments = 0
            
            for frame_id in sorted(self.track_history.keys()):
                if gt_id in self.track_history[frame_id]:
                    if not is_tracked:
                        is_tracked = True
                else:
                    if is_tracked:
                        is_tracked = False
                        track_fragments += 1
            
            fragmentations += track_fragments
        
        self.stats['fragmentations'] = fragmentations
        
        # Compute tracking ratios
        gt_ids = set().union(*[s.keys() for s in self.track_history.values()])
        
        mostly_tracked = 0
        partially_tracked = 0
        mostly_lost = 0
        
        for gt_id in gt_ids:
            tracked_frames = sum(1 for frame_id in self.track_history if gt_id in self.track_history[frame_id])
            total_frames = len(self.track_history)
            ratio = tracked_frames / total_frames if total_frames > 0 else 0
            
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio >= 0.2:
                partially_tracked += 1
            else:
                mostly_lost += 1
        
        self.stats['mostly_tracked'] = mostly_tracked
        self.stats['partially_tracked'] = partially_tracked
        self.stats['mostly_lost'] = mostly_lost
        
        return self.stats
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # No overlap
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou

class HMMEvaluator:
    """Evaluation tools for HMM behavior analysis"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.stats = {
            'sequences': 0,
            'total_samples': 0,
            'log_likelihood': 0,
            'perplexity': 0,
            'state_accuracy': 0,
            'confusion_matrix': None,
            'anomaly_detection': {
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'roc_auc': 0
            }
        }
        
        # Store all predictions and ground truths
        self.all_pred_states = []
        self.all_true_states = []
        self.all_pred_anomalies = []
        self.all_true_anomalies = []
        self.all_log_likelihoods = []
    
    def update_state_prediction(self, true_states, pred_states, log_likelihood=None):
        """
        Update metrics for state prediction
        
        Args:
            true_states: Ground truth state sequence
            pred_states: Predicted state sequence
            log_likelihood: Log likelihood of the sequence
        """
        # Ensure equal length
        min_len = min(len(true_states), len(pred_states))
        true_states = true_states[:min_len]
        pred_states = pred_states[:min_len]
        
        # Append to all predictions
        self.all_true_states.extend(true_states)
        self.all_pred_states.extend(pred_states)
        
        # Update stats
        self.stats['sequences'] += 1
        self.stats['total_samples'] += min_len
        
        # Track log likelihood
        if log_likelihood is not None:
            self.all_log_likelihoods.append(log_likelihood)
            self.stats['log_likelihood'] += log_likelihood
    
    def update_anomaly_detection(self, true_anomalies, pred_anomalies, anomaly_scores=None):
        """
        Update metrics for anomaly detection
        
        Args:
            true_anomalies: Ground truth anomaly labels (0/1)
            pred_anomalies: Predicted anomaly labels (0/1)
            anomaly_scores: Continuous anomaly scores for ROC AUC calculation
        """
        # Ensure equal length
        min_len = min(len(true_anomalies), len(pred_anomalies))
        true_anomalies = true_anomalies[:min_len]
        pred_anomalies = pred_anomalies[:min_len]
        
        # Append to all predictions
        self.all_true_anomalies.extend(true_anomalies)
        self.all_pred_anomalies.extend(pred_anomalies)
        
        # Calculate confusion matrix for this batch
        for true, pred in zip(true_anomalies, pred_anomalies):
            if true == 1 and pred == 1:
                self.stats['anomaly_detection']['true_positives'] += 1
            elif true == 0 and pred == 1:
                self.stats['anomaly_detection']['false_positives'] += 1
            elif true == 1 and pred == 0:
                self.stats['anomaly_detection']['false_negatives'] += 1
            elif true == 0 and pred == 0:
                self.stats['anomaly_detection']['true_negatives'] += 1
    
    def compute_metrics(self):
        """Compute final HMM evaluation metrics"""
        # Calculate average log likelihood and perplexity
        if self.all_log_likelihoods:
            avg_log_likelihood = np.mean(self.all_log_likelihoods)
            self.stats['log_likelihood'] = avg_log_likelihood
            self.stats['perplexity'] = np.exp(-avg_log_likelihood / self.stats['total_samples'])
        
        # Calculate state prediction accuracy and confusion matrix
        if self.all_true_states and self.all_pred_states:
            # Calculate accuracy
            matches = sum(t == p for t, p in zip(self.all_true_states, self.all_pred_states))
            self.stats['state_accuracy'] = matches / len(self.all_true_states) if self.all_true_states else 0
            
            # Calculate confusion matrix
            unique_states = sorted(set(self.all_true_states + self.all_pred_states))
            self.stats['confusion_matrix'] = confusion_matrix(
                self.all_true_states, self.all_pred_states, labels=unique_states).tolist()
        
        # Calculate anomaly detection metrics
        if self.all_true_anomalies and self.all_pred_anomalies:
            tp = self.stats['anomaly_detection']['true_positives']
            fp = self.stats['anomaly_detection']['false_positives']
            fn = self.stats['anomaly_detection']['false_negatives']
            tn = self.stats['anomaly_detection']['true_negatives']
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            self.stats['anomaly_detection']['precision'] = precision
            self.stats['anomaly_detection']['recall'] = recall
            self.stats['anomaly_detection']['f1_score'] = f1_score
            
            # Calculate ROC AUC if scores are available
            try:
                from sklearn.metrics import roc_auc_score
                if hasattr(self, 'anomaly_scores') and self.anomaly_scores:
                    self.stats['anomaly_detection']['roc_auc'] = roc_auc_score(
                        self.all_true_anomalies, self.anomaly_scores)
            except:
                pass
        
        return self.stats
    
    def save_results(self, output_file):
        """Save evaluation results to a JSON file"""
        # Compute final metrics
        self.compute_metrics()
        
        # Add timestamp
        self.stats['timestamp'] = datetime.datetime.now().isoformat()
        
        # Convert numpy arrays to lists for JSON serialization
        results = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                  for k, v in self.stats.items()}
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return output_file

class PerformanceMonitor:
    """Monitor for runtime performance metrics"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.stats = {
            'frames_processed': 0,
            'total_time': 0,
            'detection_time': 0,
            'tracking_time': 0,
            'behavior_time': 0,
            'visualization_time': 0,
            'avg_fps': 0,
            'avg_processing_times': {
                'detection': 0,
                'tracking': 0,
                'behavior': 0,
                'visualization': 0,
                'total': 0
            }
        }
    
    def update(self, frame_stats):
        """
        Update performance metrics with stats from a processed frame
        
        Args:
            frame_stats: Dictionary with frame processing stats
        """
        self.stats['frames_processed'] += 1
        self.stats['total_time'] += frame_stats.get('total_time', 0)
        self.stats['detection_time'] += frame_stats.get('detection_time', 0)
        self.stats['tracking_time'] += frame_stats.get('tracking_time', 0)
        self.stats['behavior_time'] += frame_stats.get('behavior_time', 0)
        self.stats['visualization_time'] += frame_stats.get('visualization_time', 0)
    
    def compute_metrics(self):
        """Compute average performance metrics"""
        frames = max(1, self.stats['frames_processed'])
        
        # Calculate averages
        self.stats['avg_fps'] = frames / max(0.001, self.stats['total_time'])
        
        self.stats['avg_processing_times']['detection'] = self.stats['detection_time'] / frames
        self.stats['avg_processing_times']['tracking'] = self.stats['tracking_time'] / frames
        self.stats['avg_processing_times']['behavior'] = self.stats['behavior_time'] / frames
        self.stats['avg_processing_times']['visualization'] = self.stats['visualization_time'] / frames
        self.stats['avg_processing_times']['total'] = self.stats['total_time'] / frames
        
        return self.stats
    
    def get_summary(self):
        """Get a text summary of performance metrics"""
        self.compute_metrics()
        
        summary = [
            f"Performance Summary:",
            f"  Frames processed: {self.stats['frames_processed']}",
            f"  Average FPS: {self.stats['avg_fps']:.2f}",
            f"  Average processing times (ms):",
            f"    Detection: {self.stats['avg_processing_times']['detection']*1000:.2f}",
            f"    Tracking: {self.stats['avg_processing_times']['tracking']*1000:.2f}",
            f"    Behavior analysis: {self.stats['avg_processing_times']['behavior']*1000:.2f}",
            f"    Visualization: {self.stats['avg_processing_times']['visualization']*1000:.2f}",
            f"    Total: {self.stats['avg_processing_times']['total']*1000:.2f}"
        ]
        
        return "\n".join(summary)
    
    def save_results(self, output_file):
        """Save performance results to a JSON file"""
        # Compute final metrics
        self.compute_metrics()
        
        # Add timestamp
        self.stats['timestamp'] = datetime.datetime.now().isoformat()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        return output_file 