import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from SCRIPTS.config import RNN_ATTENTION_DIR, create_directories, LOSS_WEIGHTS, TRAINING_CONFIG


def save_attention_weights(model, split_type, model_type='rnn', latent_dim=None, trial=None):
    create_directories()
    attention_weights = model.node_attention.get_attention_weights()
    
    if np.std(attention_weights) < 0.001:
        print(f"Warning: Very uniform weights detected!") #Debugging error w/ attention

    filename_parts = ['attention', split_type, model_type]
    if latent_dim is not None:
        filename_parts.append(f'{latent_dim}D')
    if trial is not None:
        filename_parts.append(f'trial{trial}')
    filename = '_'.join(filename_parts) + '.pth'
    
    save_path = RNN_ATTENTION_DIR / filename
    torch.save({
        'attention_weights': attention_weights,
        'model_type': model_type,
        'split_type': split_type,
        'latent_dim': latent_dim,
        'trial': trial
    }, save_path)
    print(f"✓ Attention weights saved to {save_path}")
    return save_path


#Weighted sampler for class balance
def create_balanced_sampler(dataset):
    labels = []
    for i in range(len(dataset)):
        _, label, _, _ = dataset[i]
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler, class_weights


def train_rnn_model(model, train_loader, test_loader, num_epochs=None, lr=None, recon_weight=None, device=None, split_type='subject', latent_dim=None, trial=None):
    num_epochs = num_epochs or TRAINING_CONFIG['epochs']
    lr = lr or TRAINING_CONFIG['learning_rate']
    recon_weight = recon_weight or LOSS_WEIGHTS['reconstruction']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'train_recon_loss': [], 'train_class_loss': [],
        'attention_weights': []
    }
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Reconstruction weight: {recon_weight}, Classification weight: {1-recon_weight}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_recon_loss = 0
        train_class_loss = 0
        
        for features, labels, _, _ in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            model_output = model(features)
            x_recon = model_output['aux']['reconstruction']
            logits = model_output['logits']
            loss_recon = criterion_recon(x_recon, features)
            loss_class = criterion_class(logits, labels)
            losses = {
                'reconstruction': loss_recon,
                'classification': loss_class
            }
            loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
            
            optimizer.zero_grad()
            loss.backward()
            
            #Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_recon.item()
            train_class_loss += loss_class.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels, _, _ in test_loader:
                features, labels = features.to(device), labels.to(device)
                
                model_output = model(features)
                x_recon = model_output['aux']['reconstruction']
                logits = model_output['logits']
                
                # Compute individual losses
                loss_recon = criterion_recon(x_recon, features)
                loss_class = criterion_class(logits, labels)
                
                # Use unified loss aggregation
                losses = {
                    'reconstruction': loss_recon,
                    'classification': loss_class
                }
                loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_acc'].append(val_acc)
        history['train_recon_loss'].append(train_recon_loss / len(train_loader))
        history['train_class_loss'].append(train_class_loss / len(train_loader))
        
        # Save attention weights
        if hasattr(model, 'node_attention'):
            attention_weights = model.node_attention.alphas.detach().cpu().numpy()
            history['attention_weights'].append(attention_weights.copy())
        
        # Progress update
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
            print(f"  Recon Loss: {train_recon_loss/len(train_loader):.4f}, "
                  f"Class Loss: {train_class_loss/len(train_loader):.4f}")
    
    # Final evaluation
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, 
                              target_names=['Rest', 'Improv', 'Scale']))
    
    # Save attention weights
    save_attention_weights(model, split_type, 'rnn', latent_dim, trial)
    
    return model, history