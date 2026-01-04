import numpy as np
import cv2
import torch
import random
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LightingAugmentations:
    """Class for generating various lighting condition augmentations"""
    
    def __init__(self):
        # Define lighting augmentations with their corresponding parameters
        self.augmentations = {
            'fog': A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, p=1.0),
            'rain': A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, 
                                drop_color=(200, 200, 200), p=1.0),
            'shadow': A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=2, 
                                    num_shadows_upper=5, shadow_dimension=5, p=1.0),
            'sun_flare': A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, 
                                        num_flare_circles_lower=6, num_flare_circles_upper=10, 
                                        src_radius=40, src_color=(255, 255, 255), p=1.0),
            'low_light': A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), contrast_limit=(0.1, 0.3), p=1.0),
            'over_exposed': A.RandomBrightnessContrast(brightness_limit=(0.3, 0.5), contrast_limit=(-0.3, -0.1), p=1.0),
            'haze': A.Compose([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.2, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=(-0.1, -0.3), p=1.0)
            ])
        }
        
        # Define specialized augmentations for training robustness
        self.adaptive_clahe = A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
        self.gamma_correction = A.RandomGamma(gamma_limit=(70, 130), p=1.0)
    
    def apply_augmentation(self, image, augment_type=None):
        """Apply a specific augmentation or a random one if not specified"""
        if augment_type is None:
            augment_type = random.choice(list(self.augmentations.keys()))
            
        if augment_type not in self.augmentations:
            raise ValueError(f"Augmentation '{augment_type}' not supported")
            
        transform = self.augmentations[augment_type]
        
        # Convert image to proper format if it's a tensor
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy (C,H,W) -> (H,W,C)
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # Apply augmentation
            augmented = transform(image=image_np)['image']
            
            # Convert back to tensor
            return torch.from_numpy(augmented).permute(2, 0, 1).float() / 255.0
        else:
            # Apply directly to numpy array or PIL image
            return transform(image=np.array(image))['image']
    
    def apply_clahe(self, image):
        """Apply CLAHE for exposure normalization"""
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            augmented = self.adaptive_clahe(image=image_np)['image']
            return torch.from_numpy(augmented).permute(2, 0, 1).float() / 255.0
        else:
            return self.adaptive_clahe(image=np.array(image))['image']
    
    def generate_lighting_pair(self, image):
        """Generate a pair of normal and augmented images for domain adaptation training"""
        # Choose random lighting condition
        augment_type = random.choice(list(self.augmentations.keys()))
        
        # Apply augmentation
        augmented = self.apply_augmentation(image, augment_type)
        
        return image, augmented, augment_type

class PhysicallyInspiredAugmentations:
    """Class for physically-inspired augmentations that simulate lighting phenomena"""
    
    def __init__(self):
        pass
    
    def add_glare(self, image, intensity=0.8, size=100, position=None):
        """Add glare effect to simulate direct sunlight"""
        if isinstance(image, torch.Tensor):
            # Convert to numpy
            is_tensor = True
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            is_tensor = False
            image_np = np.array(image)
        
        h, w, c = image_np.shape
        
        # Random position if not specified
        if position is None:
            cx = random.randint(0, w-1)
            cy = random.randint(0, h-1)
        else:
            cx, cy = position
        
        # Create glare mask
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x*x + y*y <= size*size
        
        # Apply glare
        glare = image_np.copy()
        glare[mask] = np.clip(image_np[mask] + intensity * 255, 0, 255).astype(np.uint8)
        
        # Blend with original
        alpha = 0.7
        result = cv2.addWeighted(image_np, 1-alpha, glare, alpha, 0)
        
        if is_tensor:
            return torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
        else:
            return result
    
    def simulate_shadows(self, image, num_shadows=3, shadow_dimension=5, shadow_roi=(0, 0, 1, 1)):
        """Simulate shadows with custom implementation"""
        transform = A.RandomShadow(
            shadow_roi=shadow_roi,
            num_shadows_lower=num_shadows,
            num_shadows_upper=num_shadows,
            shadow_dimension=shadow_dimension,
            p=1.0
        )
        
        if isinstance(image, torch.Tensor):
            # Convert to numpy
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # Apply shadows
            shadowed = transform(image=image_np)['image']
            
            return torch.from_numpy(shadowed).permute(2, 0, 1).float() / 255.0
        else:
            return transform(image=np.array(image))['image']
    
    def simulate_time_of_day(self, image, time='noon'):
        """Simulate different times of day with color temperature changes"""
        # Color temperature adjustments for different times
        color_adjustments = {
            'dawn': A.Compose([
                A.RGBShift(r_shift_limit=10, g_shift_limit=0, b_shift_limit=-30),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=(-0.1, 0.1))
            ]),
            'noon': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=(0.1, 0.3))
            ]),
            'dusk': A.Compose([
                A.RGBShift(r_shift_limit=30, g_shift_limit=0, b_shift_limit=-20),
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0), contrast_limit=(-0.1, 0.1))
            ]),
            'night': A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.7, -0.5), contrast_limit=(-0.2, 0)),
                A.RGBShift(r_shift_limit=-10, g_shift_limit=-10, b_shift_limit=10)
            ])
        }
        
        transform = color_adjustments.get(time)
        if transform is None:
            raise ValueError(f"Time of day '{time}' not supported")
        
        if isinstance(image, torch.Tensor):
            # Convert to numpy
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # Apply transformation
            adjusted = transform(image=image_np)['image']
            
            return torch.from_numpy(adjusted).permute(2, 0, 1).float() / 255.0
        else:
            return transform(image=np.array(image))['image']

def get_lighting_augmentation_pipeline(p=0.5):
    """Create a pipeline of lighting augmentations for training"""
    return A.Compose([
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.5),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
        ], p=p),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=p)
    ]) 