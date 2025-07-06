import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeAttention(nn.Module):
    def __init__(self, num_nodes=48):
        super().__init__()
        self.alphas = nn.Parameter(torch.ones(num_nodes))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, 48, 16)
        alphas_expanded = self.alphas.view(1, 1, -1, 1)
        weighted_features = x * alphas_expanded
        
        return weighted_features.reshape(batch_size, seq_len, -1)

class AttentionLSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        
        self.node_attention = NodeAttention(num_nodes=48)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        x_weighted = self.node_attention(x)
        
        x_flat = x_weighted.reshape(-1, input_dim)
        z_flat = self.encoder(x_flat)
        z = z_flat.reshape(batch_size, seq_len, -1)
        
        x_recon_flat = self.decoder(z_flat)
        x_recon = x_recon_flat.reshape(batch_size, seq_len, input_dim)
        
        lstm_out, (h_n, _) = self.lstm(z)
        
        h_n = h_n.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
        h_n = torch.cat([h_n[-1, 0, :, :], h_n[-1, 1, :, :]], dim=1)
        
        logits = self.classifier(h_n)
        
        return x_recon, logits, z