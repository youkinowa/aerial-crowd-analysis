import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class AdaptiveImageEnhancement(nn.Module):
    """Adaptive image enhancement pipeline for robust lighting conditions"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.0):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma
        
    def apply_clahe(self, img):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge enhanced L with original A and B channels
        merged = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def adjust_gamma(self, img):
        """Apply gamma correction"""
        # Build lookup table
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using lookup table
        return cv2.LUT(img, table)
    
    def process_image(self, img):
        """Process image with adaptive enhancements"""
        # Convert to numpy if tensor
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(img)
        
        # Auto-adjust gamma based on image brightness
        avg_brightness = np.mean(img_np)
        self.gamma = 1.0 if avg_brightness > 127 else 0.7  # Adaptive gamma
        
        # Apply enhancements
        enhanced = self.apply_clahe(img_np)
        if avg_brightness < 100 or avg_brightness > 200:  # Low light or overexposed
            enhanced = self.adjust_gamma(enhanced)
        
        # Convert back to tensor if input was tensor
        if isinstance(img, torch.Tensor):
            enhanced = torch.from_numpy(enhanced.transpose(2, 0, 1) / 255.0).float()
            
        return enhanced
    
    def forward(self, x):
        """Forward pass for batch processing"""
        if isinstance(x, torch.Tensor):
            # Process batch of images
            if x.dim() == 4:  # Batch
                enhanced = []
                for i in range(x.shape[0]):
                    enhanced.append(self.process_image(x[i]))
                return torch.stack(enhanced)
            else:  # Single image
                return self.process_image(x)
        else:
            # Process numpy or PIL image
            return self.process_image(x)

class LightingRobustPreprocessor:
    """Complete preprocessing pipeline for robust lighting conditions"""
    def __init__(self):
        self.enhancement = AdaptiveImageEnhancement()
        self.augmentations = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])
    
    def preprocess(self, img, apply_augmentation=False):
        """Preprocess image for inference with lighting robustness"""
        # Apply adaptive enhancement
        enhanced = self.enhancement.process_image(img)
        
        # Apply augmentations during training
        if apply_augmentation and isinstance(enhanced, torch.Tensor):
            enhanced = self.augmentations(enhanced)
        
        return enhanced
    
    def __call__(self, img, apply_augmentation=False):
        return self.preprocess(img, apply_augmentation) 