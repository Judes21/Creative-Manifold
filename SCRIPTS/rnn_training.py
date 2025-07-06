import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def train_rnn_model(model, train_loader, test_loader, num_epochs=50, lr=0.001, recon_weight=0.3, device=None):
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
            
            x_recon, logits, z = model(features)
            
            loss_recon = criterion_recon(x_recon, features)
            loss_class = criterion_class(logits, labels)
            loss = recon_weight * loss_recon + (1 - recon_weight) * loss_class
            
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
                
                x_recon, logits, _ = model(features)
                
                loss_recon = criterion_recon(x_recon, features)
                loss_class = criterion_class(logits, labels)
                loss = recon_weight * loss_recon + (1 - recon_weight) * loss_class
                
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
    
    return model, history