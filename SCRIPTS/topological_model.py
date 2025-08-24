import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import VietorisRipsComplex

class TopologicalModel(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, num_classes=3):
        super().__init__()
        self.node_alphas = nn.Parameter(torch.randn(48) * 0.5)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim)
        )
        
        self.vr_complex = VietorisRipsComplex(dim=1)
        
        self.topo_features_dim = 12
        self.classifier = nn.Sequential(
            nn.Linear(self.topo_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def extract_topological_features(self, z_batch):
        batch_size = z_batch.shape[0]
        all_features = []
        
        for i in range(batch_size):
            # Normalize
            z_traj = z_batch[i]
            z_mean = z_traj.mean(dim=0, keepdim=True)
            z_std = z_traj.std(dim=0, keepdim=True)
            z_norm = (z_traj - z_mean) / (z_std + 1e-8)
            
            # Compute persistence
            persistence = self.vr_complex(z_norm.unsqueeze(0))
            features = []
            
            # Extract features for dim 0 and dim 1
            for dim_idx in range(2):
                if len(persistence) > dim_idx and len(persistence[dim_idx]) > 0:
                    pers_info = persistence[dim_idx][0]
                    if hasattr(pers_info, 'diagram') and pers_info.diagram is not None:
                        diagram = pers_info.diagram
                        if diagram.numel() > 0:
                            births = diagram[:, 0]
                            deaths = diagram[:, 1]
                            finite_mask = torch.isfinite(deaths)
                            
                            if finite_mask.sum() > 0:
                                pers = deaths[finite_mask] - births[finite_mask]
                                features.extend([
                                    pers.mean().item(),
                                    pers.std().item() if len(pers) > 1 else 0.0,
                                    pers.max().item(),
                                    pers.sum().item(),
                                    float(len(pers)),
                                    deaths[finite_mask].max().item()
                                ])
                            else:
                                features.extend([0.0] * 6)
                        else:
                            features.extend([0.0] * 6)
                    else:
                        features.extend([0.0] * 6)
                else:
                    features.extend([0.0] * 6)
            
            all_features.append(features)
        
        return torch.tensor(all_features, dtype=torch.float32, device=z_batch.device)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x_4d = x.view(batch_size, seq_len, 48, 16)
        # Use raw learned parameters with abs to ensure positive weights
        alphas = torch.abs(self.node_alphas).view(1, 1, 48, 1)
        x_weighted = (x_4d * alphas).view(batch_size, seq_len, input_dim)
        
        x_flat = x_weighted.view(-1, input_dim)
        z_flat = self.encoder(x_flat)
        z = z_flat.view(batch_size, seq_len, -1)
        
        x_recon_flat = self.decoder(z_flat)
        x_recon = x_recon_flat.view(batch_size, seq_len, input_dim)
        
        topo_features = self.extract_topological_features(z)
        
        logits = self.classifier(topo_features)
        
        return {
            'logits': logits,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': x_recon,
                'latent': z,
                'topo_features': topo_features
            }
        }