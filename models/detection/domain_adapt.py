import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class ResidualBlock(nn.Module):
    """Residual Block for Generator"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """Generator for CycleGAN with 9 residual blocks"""
    def __init__(self, input_channels=3, output_channels=3, filters=64, n_blocks=9):
        super().__init__()
        
        # Initial Convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, filters, 7),
            nn.InstanceNorm2d(filters),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        for i in range(2):
            model += [
                nn.Conv2d(filters, filters*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(filters*2),
                nn.ReLU(inplace=True)
            ]
            filters *= 2
        
        # Residual blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(filters)]
        
        # Upsampling
        for i in range(2):
            model += [
                nn.ConvTranspose2d(filters, filters//2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(filters//2),
                nn.ReLU(inplace=True)
            ]
            filters //= 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(filters, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """PatchGAN discriminator for CycleGAN"""
    def __init__(self, input_channels=3, filters=64, n_layers=4):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.Conv2d(input_channels, filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Scale down layers
        for i in range(1, n_layers):
            prev_filters = filters
            filters = min(filters * 2, 512)
            model += [
                nn.Conv2d(prev_filters, filters, 4, stride=2, padding=1),
                nn.InstanceNorm2d(filters),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Output layer
        model += [nn.Conv2d(filters, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class LightingAdaptationCycleGAN:
    """CycleGAN implementation for domain adaptation across lighting conditions"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize generators and discriminators
        self.G_AB = Generator().to(device)  # Normal to abnormal lighting
        self.G_BA = Generator().to(device)  # Abnormal to normal lighting
        self.D_A = Discriminator().to(device)  # Discriminates normal lighting
        self.D_B = Discriminator().to(device)  # Discriminates abnormal lighting
        
        # Define losses
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=2e-4, betas=(0.5, 0.999)
        )
    
    def train_step(self, real_A, real_B, lambda_cycle=10.0, lambda_id=5.0):
        """Single training step for CycleGAN"""
        # Forward
        fake_B = self.G_AB(real_A)    # Generate abnormal lighting
        rec_A = self.G_BA(fake_B)     # Reconstruct normal lighting
        fake_A = self.G_BA(real_B)    # Generate normal lighting
        rec_B = self.G_AB(fake_A)     # Reconstruct abnormal lighting
        
        # Identity loss
        id_A = self.G_BA(real_A)
        id_B = self.G_AB(real_B)
        loss_id_A = self.criterion_identity(id_A, real_A) * lambda_id
        loss_id_B = self.criterion_identity(id_B, real_B) * lambda_id
        
        # GAN loss
        pred_fake_B = self.D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
        
        # Cycle loss
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * lambda_cycle
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * lambda_cycle
        
        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
        
        # Update generators
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        
        # Discriminator A
        self.optimizer_D.zero_grad()
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
        pred_fake_A = self.D_A(fake_A.detach())
        loss_D_fake_A = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        
        # Discriminator B
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
        pred_fake_B = self.D_B(fake_B.detach())
        loss_D_fake_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        
        # Update discriminators
        self.optimizer_D.step()
        
        return {
            'G_loss': loss_G.item(),
            'D_A_loss': loss_D_A.item(),
            'D_B_loss': loss_D_B.item(),
            'cycle_loss': (loss_cycle_A + loss_cycle_B).item(),
            'identity_loss': (loss_id_A + loss_id_B).item()
        }
    
    def transfer_lighting(self, img, target_domain='normal'):
        """Apply lighting transfer to an image"""
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img = img.unsqueeze(0)  # Add batch dimension
            else:
                # Convert numpy/PIL to tensor
                if not isinstance(img, torch.Tensor):
                    transform = torch.transforms.ToTensor()
                    img = transform(img).unsqueeze(0).to(self.device)
            
            if target_domain == 'normal':
                result = self.G_BA(img)
            else:  # abnormal lighting
                result = self.G_AB(img)
            
            return result.squeeze(0)  # Remove batch dimension
    
    def save_models(self, path):
        """Save model weights"""
        torch.save({
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict()
        }, path)
    
    def load_models(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(checkpoint['G_AB'])
        self.G_BA.load_state_dict(checkpoint['G_BA'])
        self.D_A.load_state_dict(checkpoint['D_A'])
        self.D_B.load_state_dict(checkpoint['D_B']) 