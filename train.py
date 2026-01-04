import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import model components
from models.detection.backbone import load_enhanced_yolov8, EnhancedYOLOv8
from models.detection.preprocessing import LightingRobustPreprocessor
from models.detection.domain_adapt import LightingAdaptationCycleGAN
from models.behavior.hmm_model import CrowdBehaviorHMM, MultiPersonTracker
from data.datasets import AerialCrowdDataset, create_dataloaders, create_cyclegan_dataloaders
from data.augmentation import LightingAugmentations, PhysicallyInspiredAugmentations

class DetectionModelTrainer:
    """Trainer for the enhanced YOLOv8 detection model"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary or path to JSON configuration
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Set device
        self.device = torch.device(self.config['device'])
        
        # Initialize models, dataloaders, etc.
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizer()
        self._setup_logger()
    
    def _load_config(self, config):
        """Load configuration from file or dict"""
        default_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'base_model': 'yolov8n.pt',
            'pretrained_weights': None,
            'data_dir': 'data/crowd_dataset',
            'output_dir': 'output',
            'batch_size': 16,
            'img_size': 640,
            'epochs': 100,
            'lr': 0.001,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'save_interval': 10,
            'eval_interval': 5,
            'num_workers': 4,
            'warmup_epochs': 3,
            'freeze_backbone': False,
            'use_amp': True
        }
        
        if isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        elif isinstance(config, dict):
            default_config.update(config)
        
        return default_config
    
    def _setup_model(self):
        """Initialize and set up the detection model"""
        print(f"Initializing model on {self.device}...")
        
        # Load base model
        if self.config.get('pretrained_weights'):
            self.model = load_enhanced_yolov8(self.config['pretrained_weights'], device=self.device)
        else:
            self.model = EnhancedYOLOv8(self.config['base_model']).to(self.device)
        
        # Freeze backbone if specified
        if self.config['freeze_backbone']:
            print("Freezing backbone layers...")
            for param in self.model.backbone.parameters():
                param.requires_grad = False
    
    def _setup_dataloaders(self):
        """Set up training and validation dataloaders"""
        print(f"Creating dataloaders from {self.config['data_dir']}...")
        
        self.train_loader, self.val_loader = create_dataloaders(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config['num_workers']
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler"""
        # Filter trainable parameters
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                else:
                    pg0.append(v)  # all else
        
        # Use SGD with momentum
        self.optimizer = optim.SGD(
            pg0, lr=self.config['lr'], momentum=self.config['momentum'], nesterov=True
        )
        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.config['weight_decay']})
        self.optimizer.add_param_group({'params': pg2})
        
        # Learning rate scheduler with linear warmup and cosine annealing
        def lr_lambda(epoch):
            if epoch < self.config['warmup_epochs']:
                return epoch / self.config['warmup_epochs']
            else:
                # Cosine annealing
                return 0.5 * (1 + np.cos(np.pi * (epoch - self.config['warmup_epochs']) / 
                                         (self.config['epochs'] - self.config['warmup_epochs'])))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Initialize AMP scaler if using mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['use_amp'])
    
    def _setup_logger(self):
        """Set up logging and TensorBoard"""
        log_dir = os.path.join(self.config['output_dir'], 'logs', 
                               datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Save configuration
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        # Set up progress tracking
        start_time = time.time()
        num_batches = len(self.train_loader)
        
        for i, (imgs, targets) in enumerate(self.train_loader):
            # Move data to device
            imgs = torch.stack([img for img in imgs]).to(self.device)
            
            # Process targets (assuming YOLOv8 format)
            batch_targets = []
            for batch_idx, batch_target in enumerate(targets):
                # Convert target format as needed for your model
                batch_targets.append({
                    'boxes': batch_target['boxes'].to(self.device),
                    'labels': batch_target['labels'].to(self.device),
                    'image_id': batch_target['image_id'].to(self.device)
                })
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                outputs = self.model(imgs)
                loss = outputs[0]  # Assuming loss is returned as first element
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Log progress
            if i % 10 == 0:
                print(f"Epoch {epoch}/{self.config['epochs']} | Batch {i}/{num_batches} | "
                      f"Loss: {loss.item():.4f} | LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch}/{self.config['epochs']} completed in {elapsed:.2f}s | "
              f"Avg Loss: {avg_loss:.4f}")
        
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate model on validation set"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(self.val_loader):
                # Move data to device
                imgs = torch.stack([img for img in imgs]).to(self.device)
                
                # Process targets
                batch_targets = []
                for batch_idx, batch_target in enumerate(targets):
                    batch_targets.append({
                        'boxes': batch_target['boxes'].to(self.device),
                        'labels': batch_target['labels'].to(self.device),
                        'image_id': batch_target['image_id'].to(self.device)
                    })
                
                # Forward pass
                outputs = self.model(imgs)
                loss = outputs[0]
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.config['use_amp'] else None
        }, filename)
        
        print(f"Checkpoint saved to {filename}")
        
        # Save best model separately
        if is_best:
            best_filename = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_filename)
            print(f"Best model saved to {best_filename}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate model
            if epoch % self.config['eval_interval'] == 0:
                val_loss = self.validate(epoch)
                
                # Check if best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print(f"New best model with validation loss: {best_val_loss:.4f}")
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("Training completed!")
        self.writer.close()

class CycleGANTrainer:
    """Trainer for CycleGAN domain adaptation model"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary or path to JSON configuration
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Set device
        self.device = torch.device(self.config['device'])
        
        # Initialize models, dataloaders, etc.
        self._setup_model()
        self._setup_dataloaders()
        self._setup_logger()
    
    def _load_config(self, config):
        """Load configuration from file or dict"""
        default_config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'data_dir': 'data/crowd_dataset',
            'output_dir': 'output/cyclegan',
            'batch_size': 8,
            'img_size': 256,
            'epochs': 100,
            'save_interval': 10,
            'log_interval': 100,
            'sample_interval': 500,
            'num_workers': 4,
            'lambda_cycle': 10.0,
            'lambda_identity': 5.0
        }
        
        if isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        elif isinstance(config, dict):
            default_config.update(config)
        
        return default_config
    
    def _setup_model(self):
        """Initialize and set up the CycleGAN model"""
        print(f"Initializing CycleGAN model on {self.device}...")
        
        self.cycle_gan = LightingAdaptationCycleGAN(device=self.device)
    
    def _setup_dataloaders(self):
        """Set up training dataloader for CycleGAN"""
        print(f"Creating CycleGAN dataloader from {self.config['data_dir']}...")
        
        self.dataloader = create_cyclegan_dataloaders(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config['num_workers']
        )
        
        print(f"Training samples: {len(self.dataloader.dataset)}")
    
    def _setup_logger(self):
        """Set up logging and TensorBoard"""
        log_dir = os.path.join(self.config['output_dir'], 'logs', 
                               datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Create sample output directory
        self.sample_dir = os.path.join(self.config['output_dir'], 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(log_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = os.path.join(checkpoint_dir, f'cyclegan_epoch_{epoch}.pt')
        
        # Save model checkpoint
        self.cycle_gan.save_models(filename)
        
        print(f"CycleGAN checkpoint saved to {filename}")
    
    def save_sample_images(self, epoch, batch_idx, batch):
        """Save sample transformed images"""
        from torchvision.utils import save_image
        import torchvision.transforms.functional as F
        
        # Get sample images
        real_A = batch['A'][:4]  # Only use first 4 images
        real_B = batch['B'][:4]
        
        # Process images
        with torch.no_grad():
            fake_B = self.cycle_gan.G_AB(real_A)
            fake_A = self.cycle_gan.G_BA(real_B)
            
            # Reconstruct original
            rec_A = self.cycle_gan.G_BA(fake_B)
            rec_B = self.cycle_gan.G_AB(fake_A)
        
        # Save images
        save_dir = os.path.join(self.sample_dir, f'epoch_{epoch}_batch_{batch_idx}')
        os.makedirs(save_dir, exist_ok=True)
        
        save_image(real_A, os.path.join(save_dir, 'real_A.png'))
        save_image(real_B, os.path.join(save_dir, 'real_B.png'))
        save_image(fake_B, os.path.join(save_dir, 'fake_B.png'))
        save_image(fake_A, os.path.join(save_dir, 'fake_A.png'))
        save_image(rec_A, os.path.join(save_dir, 'rec_A.png'))
        save_image(rec_B, os.path.join(save_dir, 'rec_B.png'))
    
    def train(self):
        """Main training loop for CycleGAN"""
        print(f"Starting CycleGAN training for {self.config['epochs']} epochs...")
        
        total_batches = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                # Move data to device
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                
                # Train generators and discriminators
                losses = self.cycle_gan.train_step(
                    real_A, real_B, 
                    lambda_cycle=self.config['lambda_cycle'],
                    lambda_id=self.config['lambda_identity']
                )
                
                # Log batch progress
                if batch_idx % self.config['log_interval'] == 0:
                    print(f"Epoch {epoch}/{self.config['epochs']} | Batch {batch_idx}/{len(self.dataloader)} | "
                          f"G_loss: {losses['G_loss']:.4f} | D_A_loss: {losses['D_A_loss']:.4f} | "
                          f"D_B_loss: {losses['D_B_loss']:.4f}")
                    
                    self.writer.add_scalar('Train/G_loss', losses['G_loss'], total_batches)
                    self.writer.add_scalar('Train/D_A_loss', losses['D_A_loss'], total_batches)
                    self.writer.add_scalar('Train/D_B_loss', losses['D_B_loss'], total_batches)
                    self.writer.add_scalar('Train/cycle_loss', losses['cycle_loss'], total_batches)
                    self.writer.add_scalar('Train/identity_loss', losses['identity_loss'], total_batches)
                
                # Save sample images
                if batch_idx % self.config['sample_interval'] == 0:
                    self.save_sample_images(epoch, batch_idx, batch)
                
                total_batches += 1
            
            # Log epoch progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch}/{self.config['epochs']} completed in {epoch_time:.2f}s")
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
        
        print("CycleGAN training completed!")
        self.writer.close()

class HMMTrainer:
    """Trainer for HMM behavior model"""
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary or path to JSON configuration
        """
        # Load configuration
        self.config = self._load_config(config)
        
        # Initialize HMM model
        self.hmm = CrowdBehaviorHMM(
            n_states=self.config['n_states'],
            n_features=self.config['n_features'],
            covariance_type=self.config['covariance_type'],
            n_iter=self.config['n_iter']
        )
        
        # Initialize tracker for processing trajectories
        self.tracker = MultiPersonTracker(
            max_age=self.config['tracker_max_age'],
            min_hits=self.config['tracker_min_hits'],
            iou_threshold=self.config['tracker_iou_threshold']
        )
        
        # Output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _load_config(self, config):
        """Load configuration from file or dict"""
        default_config = {
            'n_states': 5,
            'n_features': 4,
            'covariance_type': 'diag',
            'n_iter': 100,
            'tracker_max_age': 30,
            'tracker_min_hits': 3,
            'tracker_iou_threshold': 0.3,
            'trajectory_files': [],
            'min_trajectory_length': 5,
            'output_dir': 'output/hmm',
            'model_name': 'hmm_model.pkl'
        }
        
        if isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        elif isinstance(config, dict):
            default_config.update(config)
        
        return default_config
    
    def load_trajectories(self):
        """Load trajectories from files or detection results"""
        all_trajectories = []
        
        # If trajectory files provided, load them
        if self.config['trajectory_files']:
            for traj_file in self.config['trajectory_files']:
                print(f"Loading trajectories from {traj_file}...")
                
                # Load trajectories from file (assuming JSON or numpy format)
                if traj_file.endswith('.json'):
                    with open(traj_file, 'r') as f:
                        data = json.load(f)
                        for trajectory in data:
                            if len(trajectory) >= self.config['min_trajectory_length']:
                                all_trajectories.append(trajectory)
                elif traj_file.endswith('.npy'):
                    trajectories = np.load(traj_file, allow_pickle=True)
                    for trajectory in trajectories:
                        if len(trajectory) >= self.config['min_trajectory_length']:
                            all_trajectories.append(trajectory)
        
        print(f"Loaded {len(all_trajectories)} trajectories")
        
        return all_trajectories
    
    def train(self):
        """Train HMM model on trajectories"""
        # Load trajectories
        trajectories = self.load_trajectories()
        
        if not trajectories:
            print("No trajectories found. Cannot train HMM model.")
            return
        
        # Preprocess trajectories
        print("Preprocessing trajectories...")
        features = self.hmm.preprocess_trajectories(trajectories)
        
        if features.size == 0:
            print("Error: No valid features extracted from trajectories.")
            return
        
        print(f"Extracted {features.shape[0]} feature vectors with {features.shape[1]} dimensions")
        
        # Train HMM model
        print(f"Training HMM model with {self.config['n_states']} states and {self.config['n_iter']} iterations...")
        start_time = time.time()
        
        self.hmm.fit(features)
        
        train_time = time.time() - start_time
        print(f"HMM training completed in {train_time:.2f}s")
        
        # Save model
        model_path = os.path.join(self.config['output_dir'], self.config['model_name'])
        self.hmm.save_model(model_path)
        print(f"HMM model saved to {model_path}")
        
        # Test model with sample trajectories
        print("Testing model on sample trajectories...")
        sample_trajectories = trajectories[:5] if len(trajectories) >= 5 else trajectories
        
        for i, trajectory in enumerate(sample_trajectories):
            # Extract features
            traj_features = self.hmm.preprocess_trajectories([trajectory])
            
            # Skip if no valid features
            if traj_features.size == 0:
                continue
            
            # Predict states
            states = self.hmm.predict_states(traj_features)
            
            # Detect anomalies
            anomalies = self.hmm.detect_anomalies(traj_features)
            
            # Analyze behavior
            behavior = self.hmm.analyze_behavior_pattern(states)
            
            print(f"Trajectory {i+1}:")
            print(f"  Pattern: {behavior['pattern']}")
            print(f"  Most common state: {behavior['most_common_state_label']}")
            print(f"  Anomalous: {anomalies['is_anomalous']}")
            print()
        
        return self.hmm

def main():
    """Main function to run training"""
    parser = argparse.ArgumentParser(description='Training Pipeline for Aerial Crowd Analysis')
    parser.add_argument('--mode', type=str, required=True, choices=['detection', 'cyclegan', 'hmm'],
                       help='Training mode: detection, cyclegan, or hmm')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Train based on selected mode
    if args.mode == 'detection':
        trainer = DetectionModelTrainer(args.config)
        trainer.train()
    elif args.mode == 'cyclegan':
        trainer = CycleGANTrainer(args.config)
        trainer.train()
    elif args.mode == 'hmm':
        trainer = HMMTrainer(args.config)
        trainer.train()
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main() 