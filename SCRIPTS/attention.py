import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAttention(nn.Module):    
    def __init__(self, num_nodes=48, use_sigmoid=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_sigmoid = use_sigmoid
        self.alphas = nn.Parameter(torch.randn(num_nodes) * 0.5)
        
    def forward(self, x):
        attention_weights = torch.abs(self.alphas)
            
        if len(x.shape) == 2:
            # Handle 2D input: (batch_size, 768)
            batch_size = x.size(0)
            x_reshaped = x.view(batch_size, self.num_nodes, -1)
            alphas_expanded = attention_weights.view(1, -1, 1)
            weighted_features = x_reshaped * alphas_expanded
            return weighted_features.reshape(batch_size, -1)
            
        elif len(x.shape) == 3:
            # Handle 3D input: (batch_size, seq_len, 768)
            batch_size, seq_len, _ = x.shape
            x_reshaped = x.view(batch_size, seq_len, self.num_nodes, -1)
            alphas_expanded = attention_weights.view(1, 1, -1, 1)
            weighted_features = x_reshaped * alphas_expanded
            return weighted_features.reshape(batch_size, seq_len, -1)
            
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 2D or 3D tensor.")

    def get_attention_weights(self):
        return torch.abs(self.alphas).detach().cpu().numpy()