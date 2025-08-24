import torch
import torch.nn as nn
from SCRIPTS.attention import NodeAttention
import numpy as np


class TopologicalEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, feature_dim=12):
        super().__init__()
        self.node_attention = NodeAttention(num_nodes=48, use_sigmoid=True)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim)
        )

        self.topo_projection = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_attended = self.node_attention(x)
        
        x_flat = x_attended.reshape(-1, x_attended.shape[-1])
        z_flat = self.encoder(x_flat)
        z = z_flat.reshape(batch_size, seq_len, -1)
        
        # Pool across sequence and extract topological features
        z_pooled = z.mean(dim=1)
        topo_features = self.topo_projection(z_pooled)
        
        return topo_features


class RNNEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, hidden_dim=128, feature_dim=256):
        super().__init__()
        self.node_attention = NodeAttention(num_nodes=48, use_sigmoid=False)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.feature_projection = nn.Linear(hidden_dim * 2, feature_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_attended = self.node_attention(x)

        x_flat = x_attended.reshape(-1, x_attended.shape[-1])
        z_flat = self.encoder(x_flat)
        z = z_flat.reshape(batch_size, seq_len, -1)
        
        lstm_out, (hidden, cell) = self.lstm(z)
    
        hidden_final = hidden[-2:].transpose(0, 1).reshape(batch_size, -1)
        rnn_features = self.feature_projection(hidden_final)
        
        return rnn_features


class CurvatureEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, feature_dim=390):
        super().__init__()
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
        
        self.curvature_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Tanh()
        )
        
        self.feature_projection = nn.Linear(latent_dim, feature_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_attended = self.node_attention(x)
        
        x_flat = x_attended.reshape(-1, x_attended.shape[-1])
        z_flat = self.encoder(x_flat)
        z = z_flat.reshape(batch_size, seq_len, -1)
        
        curv_values = self.curvature_net(z) 
        curv_pooled = curv_values.mean(dim=1) 
        curv_features = self.feature_projection(curv_pooled)
        
        return curv_features


class CombinedModelV2(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, num_classes=3, dropout=0.3):
        super().__init__()
        
        # Initialize fresh encoders
        self.topological_encoder = TopologicalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            feature_dim=12
        )
        
        self.rnn_encoder = RNNEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=128,
            feature_dim=256
        )
        
        self.curvature_encoder = CurvatureEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            feature_dim=390
        )
        
        # Combined feature dimension
        total_features = 12 + 256 + 390  # 658 total
        
        # Unified classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.topo_decoder = nn.Sequential(
            nn.Linear(12, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        self.rnn_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
        self.curv_decoder = nn.Sequential(
            nn.Linear(390, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        print(f"CombinedModelV2 initialized with {total_features} combined features")
        print("All encoders are trainable and will be jointly optimized")
        
    def forward(self, x):
        topo_features = self.topological_encoder(x)
        rnn_features = self.rnn_encoder(x)
        curv_features = self.curvature_encoder(x)
        
        combined_features = torch.cat([
            topo_features,
            rnn_features,
            curv_features
        ], dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        # Reconstructions
        x_avg = x.mean(dim=1)
        topo_recon = self.topo_decoder(topo_features)
        rnn_recon = self.rnn_decoder(rnn_features)
        curv_recon = self.curv_decoder(curv_features)
        reconstruction = (topo_recon + rnn_recon + curv_recon) / 3
        
        return {
            'logits': logits,
            'aux': {
                'reconstruction': reconstruction,
                'features': {
                    'topological': topo_features,
                    'rnn': rnn_features,
                    'curvature': curv_features,
                    'combined': combined_features
                },
                'reconstructions': {
                    'topological': topo_recon,
                    'rnn': rnn_recon,
                    'curvature': curv_recon
                }
            }
        }
    
    def get_attention_weights(self):
        return {
            'topological': self.topological_encoder.node_attention.get_attention_weights(),
            'rnn': self.rnn_encoder.node_attention.get_attention_weights(),
            'curvature': self.curvature_encoder.node_attention.get_attention_weights()
        }
    
    def get_combined_attention_weights(self):
        attention_weights = []
        
        # Get attention from each encoder
        attention_weights.append(self.topological_encoder.node_attention.get_attention_weights())
        attention_weights.append(self.rnn_encoder.node_attention.get_attention_weights())
        attention_weights.append(self.curvature_encoder.node_attention.get_attention_weights())
        combined_attention = np.mean(attention_weights, axis=0)
        
        return combined_attention