import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AerialCrowdDataset(Dataset):
    """Dataset for aerial crowd detection with lighting condition augmentations"""
    def __init__(self, root_dir, split='train', img_size=640, augment=True):
        """
        Args:
            root_dir: Root directory with images and annotations
            split: 'train', 'val', or 'test'
            img_size: Input image size for the model
            augment: Whether to apply augmentations
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # Set up paths
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # Get image files
        self.img_files = sorted([f for f in os.listdir(self.img_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Basic transformations
        if self.split == 'train' and self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.RandomGamma(gamma_limit=(80, 120)),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
                ], p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3),
                    A.MotionBlur(blur_limit=3),
                ], p=0.2),
                A.GaussNoise(var_limit=(10, 50), p=0.2),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, 
                                num_flare_circles_lower=1, num_flare_circles_upper=3, p=0.2),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
        # Lighting condition augmentations (synthetic)
        self.lighting_conditions = {
            'normal': None,
            'fog': A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=1.0),
            'rain': A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=1.0),
            'shadow': A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=2, num_shadows_upper=5, p=1.0),
            'sunflare': A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=1.0),
            'low_light': A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), p=1.0),
            'over_exposed': A.RandomBrightnessContrast(brightness_limit=(0.3, 0.5), p=1.0)
        }
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        
        # Get label path
        label_file = self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        label_path = os.path.join(self.label_dir, label_file)
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read labels
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_id = int(data[0])
                    # YOLO format: class_id, x_center, y_center, width, height
                    box = [float(x) for x in data[1:5]]
                    boxes.append(box)
                    class_labels.append(class_id)
        
        # Apply transformations
        if boxes:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            # If no boxes, just transform the image
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            
        # Create targets
        targets = {
            'boxes': torch.tensor(boxes) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(class_labels) if class_labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'img_path': img_path
        }
        
        return image, targets
    
    def apply_lighting_condition(self, image, condition='normal'):
        """Apply synthetic lighting condition to an image"""
        if condition == 'normal' or condition not in self.lighting_conditions:
            return image
            
        transform = self.lighting_conditions[condition]
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Apply transform
            img_np = transform(image=img_np)['image']
            
            # Convert back to tensor
            return torch.from_numpy(img_np).permute(2, 0, 1) / 255.0
        else:
            return transform(image=image)['image']

class SyntheticLightingDataset(Dataset):
    """Dataset for generating paired images with normal and abnormal lighting conditions"""
    def __init__(self, base_dataset, lighting_conditions=['fog', 'shadow', 'low_light', 'over_exposed', 'sunflare']):
        self.base_dataset = base_dataset
        self.lighting_conditions = lighting_conditions
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.base_dataset[idx]
        
        # Select random abnormal lighting condition
        condition = random.choice(self.lighting_conditions)
        
        # Apply lighting condition
        abnormal_image = self.base_dataset.apply_lighting_condition(image, condition)
        
        return {
            'A': image,               # Normal lighting
            'B': abnormal_image,      # Abnormal lighting
            'condition': condition     # Which condition was applied
        }

def create_dataloaders(root_dir, batch_size=16, img_size=640, num_workers=4):
    """Create dataloaders for training and validation"""
    train_dataset = AerialCrowdDataset(root_dir, split='train', img_size=img_size, augment=True)
    val_dataset = AerialCrowdDataset(root_dir, split='val', img_size=img_size, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))  # Custom collate function for variable-sized targets
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader

def create_cyclegan_dataloaders(root_dir, batch_size=8, img_size=256, num_workers=4):
    """Create dataloaders for CycleGAN training"""
    # Create base dataset
    base_dataset = AerialCrowdDataset(root_dir, split='train', img_size=img_size, augment=False)
    
    # Create synthetic paired dataset
    synthetic_dataset = SyntheticLightingDataset(base_dataset)
    
    # Create dataloader
    dataloader = DataLoader(
        synthetic_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return dataloader 