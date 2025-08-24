import torch
import torch.nn as nn
import torch.nn.functional as F

#Simple MLP for classification
class BrainStateClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=3, dropout_rate=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], num_classes)) 
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        return {
            'logits': logits,
            'recon_loss': None,
            'class_loss': None,
            'aux': {}
        }

#Compress features to latent space
class BrainStateAutoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48):
        super().__init__()
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
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return {
            'logits': None,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': x_recon,
                'latent': z
            }
        }


# Combined autoencoder + classifier
class BrainStateFullModel(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, num_classes=3):
        super().__init__()
        
        self.autoencoder = BrainStateAutoencoder(input_dim, latent_dim)
        classifier_hidden = max(16, latent_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, num_classes)
        )
        
    def forward(self, x):
        autoencoder_output = self.autoencoder(x)
        x_recon = autoencoder_output['aux']['reconstruction']
        z = autoencoder_output['aux']['latent']
        logits = self.classifier(z)
        return {
            'logits': logits,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': x_recon,
                'latent': z
            }
        }


# Node-level attention on fNIRS channels
class NodeAttention(nn.Module):
    def __init__(self, num_nodes=48):
        super().__init__()
        self.alphas = nn.Parameter(torch.randn(num_nodes) * 0.1)
        
    def forward(self, x):
        # Always use softmax for proper probability distribution
        attention_weights = F.softmax(self.alphas, dim=0)
        
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 48, 16)  # Reshape to nodes x features
        alphas_expanded = attention_weights.view(1, -1, 1)
        weighted_features = x * alphas_expanded
        
        if len(x.shape) == 2:
            return weighted_features.reshape(batch_size, -1)
        return weighted_features


# Autoencoder w/ node attention
class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48):
        super().__init__()
        
        self.node_attention = NodeAttention()
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
        
    def forward(self, x):
        x = self.node_attention(x)
        if len(x.shape) > 2:
            x = x.reshape(x.size(0), -1)

        z = self.encoder(x)
        x_recon = self.decoder(z)
        return {
            'logits': None,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': x_recon,
                'latent': z
            }
        }


# Full model
class AttentionFullModel(nn.Module):
    def __init__(self, input_dim=768, latent_dim=48, num_classes=3):
        super().__init__()
        
        self.autoencoder = AttentionAutoencoder(input_dim, latent_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        autoencoder_output = self.autoencoder(x)
        x_recon = autoencoder_output['aux']['reconstruction']
        z = autoencoder_output['aux']['latent']
        logits = self.classifier(z)
        return {
            'logits': logits,
            'recon_loss': None,
            'class_loss': None,
            'aux': {
                'reconstruction': x_recon,
                'latent': z
            }
        }