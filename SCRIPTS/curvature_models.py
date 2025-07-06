import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Manually calculate curvature of trajectories
class DifferentiableCurvature(nn.Module):
    def __init__(self, drop_edges=5):
        super().__init__()
        self.drop_edges = drop_edges
        
    def forward(self, trajectory):
        batch_size, n_points, n_dims = trajectory.shape
        
        # First derivative kernel
        kernel = torch.tensor([-0.5, 0.0, 0.5], dtype=trajectory.dtype, device=trajectory.device)
        kernel = kernel.view(1, 1, 3).repeat(n_dims, 1, 1)
        
        # Pad trajectory for convolution
        padded = F.pad(trajectory.transpose(1, 2), (1, 1), mode='replicate')
        
        # Compute velocity (first derivative)
        velocity = F.conv1d(padded, kernel, groups=n_dims).transpose(1, 2)
        
        # Second derivative kernel
        kernel2 = torch.tensor([1.0, -2.0, 1.0], dtype=trajectory.dtype, device=trajectory.device)
        kernel2 = kernel2.view(1, 1, 3).repeat(n_dims, 1, 1)
        
        # Compute acceleration (second derivative)
        acceleration = F.conv1d(padded, kernel2, groups=n_dims).transpose(1, 2)
        
        # Compute curvature for each trajectory
        curvatures = []
        for i in range(batch_size):
            v = velocity[i]  # (n_points, n_dims)
            a = acceleration[i]  # (n_points, n_dims)
            
            # Compute |v|
            v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-8
            
            # Normalize velocity
            v_normalized = v / v_norm
            
            # Compute perpendicular component of acceleration
            a_parallel = torch.sum(a * v_normalized, dim=1, keepdim=True) * v_normalized
            a_perp = a - a_parallel
            
            # Curvature = |a_perp| / |v|^2
            curvature = torch.norm(a_perp, dim=1) / (v_norm.squeeze() ** 2)
            curvatures.append(curvature)
        
        curvatures = torch.stack(curvatures)
        
        # Drop edges
        if self.drop_edges > 0:
            curvatures = curvatures[:, self.drop_edges:-self.drop_edges]
        
        return curvatures


class CurvatureModel(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, num_classes=3, drop_edges=5):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        self.curvature = DifferentiableCurvature(drop_edges=drop_edges)
        
        # Classifier on curvature features (400 timepoints - 2*drop_edges = 390 curvature values)
        curvature_dim = 390
        self.curvature_classifier = nn.Sequential(
            nn.Linear(curvature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def encode_sequence(self, x):
        batch_size, seq_len, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)
        z_flat = self.encoder(x_flat)
        z = z_flat.view(batch_size, seq_len, -1)
        return z
    
    def forward(self, x, return_all=False):
        z = self.encode_sequence(x)
        
        curvature_values = self.curvature(z)
        
        class_logits = self.curvature_classifier(curvature_values)
        
        batch_size, seq_len, feat_dim = x.shape
        z_flat = z.view(-1, z.shape[-1])
        x_recon_flat = self.decoder(z_flat)
        x_recon = x_recon_flat.view(batch_size, seq_len, feat_dim)
        
        if return_all:
            return class_logits, x_recon, z, curvature_values
        else:
            return class_logits, x_recon