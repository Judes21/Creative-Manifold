import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from SCRIPTS.config import TOPOLOGICAL_ATTENTION_DIR, create_directories, LOSS_WEIGHTS, TRAINING_CONFIG


def save_attention_weights(model, split_type, model_type='topological', latent_dim=None, trial=None):
    create_directories()
    attention_weights = torch.abs(model.node_alphas).detach().cpu().numpy()
    filename_parts = ['attention', split_type, model_type]
    if latent_dim is not None:
        filename_parts.append(f'{latent_dim}D')
    if trial is not None:
        filename_parts.append(f'trial{trial}')
    filename = '_'.join(filename_parts) + '.pth'
    
    save_path = TOPOLOGICAL_ATTENTION_DIR / filename
    torch.save({
        'attention_weights': attention_weights,
        'model_type': model_type,
        'split_type': split_type,
        'latent_dim': latent_dim,
        'trial': trial
    }, save_path)
    print(f"✓ Attention weights saved to {save_path}")
    return save_path


def train_topological_model(model, train_loader, test_loader, num_epochs=None, lr=None, device=None, split_type='subject', latent_dim=None, trial=None):
    num_epochs = num_epochs or TRAINING_CONFIG['epochs']
    lr = lr or TRAINING_CONFIG['learning_rate']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'recon_loss': [], 'class_loss': [],
        'attention_weights': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_recon_loss = 0
        train_class_loss = 0
        train_topo_loss = 0
        
        for features, labels, _, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            model_output = model(features)
            x_recon = model_output['aux']['reconstruction']
            logits = model_output['logits']
            z = model_output['aux']['latent']
            topo_features = model_output['aux']['topo_features']
        
            loss_recon = criterion_recon(x_recon, features)
            loss_class = criterion_class(logits, labels)
            loss_topo = torch.mean(topo_features ** 2)
            
            losses = {
                'reconstruction': loss_recon,
                'classification': loss_class,
                'topological': loss_topo
            }
            total_loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_recon_loss += loss_recon.item()
            train_class_loss += loss_class.item()
            train_topo_loss += loss_topo.item()
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
                features = features.to(device)
                labels = labels.to(device)
                
                model_output = model(features)
                x_recon = model_output['aux']['reconstruction']
                logits = model_output['logits']
                z = model_output['aux']['latent']
                topo_features = model_output['aux']['topo_features']
                
                # Compute individual losses
                loss_recon = criterion_recon(x_recon, features)
                loss_class = criterion_class(logits, labels)
                loss_topo = torch.mean(topo_features ** 2)
                losses = {
                    'reconstruction': loss_recon,
                    'classification': loss_class,
                    'topological': loss_topo
                }
                total_loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
                
                val_loss += total_loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_acc'].append(val_acc)
        history['recon_loss'].append(train_recon_loss / len(train_loader))
        history['class_loss'].append(train_class_loss / len(train_loader))
        
        if hasattr(model, 'node_alphas'):
            attention_weights = F.softmax(model.node_alphas, dim=0).detach().cpu().numpy()
            history['attention_weights'].append(attention_weights.copy())
        
        # Progress update
        if (epoch + 1) % 10 == 0:
            unique_preds = np.unique(val_preds)
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")
            print(f"  Recon Loss: {train_recon_loss/len(train_loader):.4f}, "
                  f"Class Loss: {train_class_loss/len(train_loader):.4f}")
            print(f"  Predicting classes: {unique_preds}")
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, 
                              target_names=['Rest', 'Improv', 'Scale']))
    
    # Save attention weights
    save_attention_weights(model, split_type, 'topological', latent_dim, trial)
    
    return model, history, val_preds, val_labels