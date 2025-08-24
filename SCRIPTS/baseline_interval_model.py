import torch
import torch.nn as nn
import torch.nn.functional as F
from SCRIPTS.attention import NodeAttention


class BaselineIntervalModel(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, seq_len=400, num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        
        # Add node attention mechanism
        self.node_attention = NodeAttention(num_nodes=48, use_sigmoid=False)
        
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
        
        classifier_input_dim = latent_dim * seq_len
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def encode_sequence(self, x):
        batch_size, seq_len, _ = x.shape
        x_attended = self.node_attention(x)
        
        latent_sequence = []
        for t in range(seq_len):
            z_t = self.encoder(x_attended[:, t, :])
            latent_sequence.append(z_t)

        latent_sequence = torch.stack(latent_sequence, dim=1)
        return latent_sequence
    
    def decode_sequence(self, z):
        batch_size, seq_len, _ = z.shape
        reconstructions = []
        for t in range(seq_len):
            x_recon_t = self.decoder(z[:, t, :])
            reconstructions.append(x_recon_t)
        
        reconstructions = torch.stack(reconstructions, dim=1)
        return reconstructions
        
    def forward(self, x):
        batch_size = x.size(0)
        latent_sequence = self.encode_sequence(x)
        
        reconstructions = self.decode_sequence(latent_sequence)
        
        latent_flat = latent_sequence.view(batch_size, -1)
        
        logits = self.classifier(latent_flat)
        
        return {
            'logits': logits,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': reconstructions,
                'latent': latent_sequence
            }
        }
    
    def get_latent_representation(self, x):
        with torch.no_grad():
            return self.encode_sequence(x)