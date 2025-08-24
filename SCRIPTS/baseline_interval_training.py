import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
from SCRIPTS.config import BASELINE_INTERVAL_ATTENTION_DIR, create_directories, LOSS_WEIGHTS, TRAINING_CONFIG


def save_attention_weights(model, split_type, model_type='baseline_interval', latent_dim=None, trial=None):
    create_directories()
    attention_weights = model.node_attention.get_attention_weights()
    
    filename_parts = ['attention', split_type, model_type]
    if latent_dim is not None:
        filename_parts.append(f'{latent_dim}D')
    if trial is not None:
        filename_parts.append(f'trial{trial}')
    filename = '_'.join(filename_parts) + '.pth'
    
    save_path = BASELINE_INTERVAL_ATTENTION_DIR / filename
    torch.save({
        'attention_weights': attention_weights,
        'model_type': model_type,
        'split_type': split_type,
        'latent_dim': latent_dim,
        'trial': trial
    }, save_path)
    print(f"✓ Attention weights saved to {save_path}")
    return save_path


def train_baseline_interval_model(model, train_loader, test_loader, num_epochs=None, 
                                  lr=None, recon_weight=None, device='cuda', split_type='subject', latent_dim=None, trial=None):
    num_epochs = num_epochs or TRAINING_CONFIG['epochs']
    lr = lr or TRAINING_CONFIG['learning_rate']
    recon_weight = recon_weight or LOSS_WEIGHTS['reconstruction']
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, verbose=True
    )
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_recon_loss': [], 'train_class_loss': [],
        'val_loss': [], 'val_acc': [], 'val_recon_loss': [], 'val_class_loss': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_class_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features, labels, _, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            model_output = model(features)
            reconstructions = model_output['aux']['reconstruction']
            logits = model_output['logits']
            
            # Compute individual losses
            loss_recon = criterion_recon(reconstructions, features)
            loss_class = criterion_class(logits, labels)
            
            # Use unified loss aggregation
            losses = {
                'reconstruction': loss_recon,
                'classification': loss_class
            }
            loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_recon.item()
            train_class_loss += loss_class.item()
            
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_class_loss = 0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                features, labels, _, _ = batch
                features = features.to(device)
                labels = labels.to(device)
                
                model_output = model(features)
                reconstructions = model_output['aux']['reconstruction']
                logits = model_output['logits']
                
                # Compute individual losses
                loss_recon = criterion_recon(reconstructions, features)
                loss_class = criterion_class(logits, labels)
                
                # Use unified loss aggregation
                losses = {
                    'reconstruction': loss_recon,
                    'classification': loss_class
                }
                loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
                
                val_loss += loss.item()
                val_recon_loss += loss_recon.item()
                val_class_loss += loss_class.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Epoch metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_recon_loss'].append(train_recon_loss / len(train_loader))
        history['train_class_loss'].append(train_class_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_recon_loss'].append(val_recon_loss / len(test_loader))
        history['val_class_loss'].append(val_class_loss / len(test_loader))
        history['val_acc'].append(val_acc)
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, "
                  f"Acc: {train_acc:.2f}%, "
                  f"Recon: {train_recon_loss/len(train_loader):.4f}, "
                  f"Class: {train_class_loss/len(train_loader):.4f}")
            print(f"  Val   - Loss: {val_loss/len(test_loader):.4f}, "
                  f"Acc: {val_acc:.2f}%, "
                  f"Recon: {val_recon_loss/len(test_loader):.4f}, "
                  f"Class: {val_class_loss/len(test_loader):.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with validation accuracy: {best_val_acc:.2f}%")
    
    val_predictions = []
    val_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            features, labels, _, _ = batch
            features = features.to(device)
            
            model_output = model(features)
            logits = model_output['logits']
            _, predicted = torch.max(logits.data, 1)
            
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    print("\nFinal Classification Report:")
    print(classification_report(val_labels, val_predictions, 
                              target_names=['Rest', 'Improv', 'Scale']))
    
    # Save attention weights
    save_attention_weights(model, split_type, 'baseline_interval', latent_dim, trial)
    
    return model, history


def evaluate_baseline_interval_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    
    predictions = []
    labels = []
    latent_representations = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, batch_labels, _, _ = batch
            features = features.to(device)
            
            model_output = model(features)
            reconstructions = model_output['aux']['reconstruction']
            latent_sequence = model_output['aux']['latent']
            logits = model_output['logits']
            
            _, predicted = torch.max(logits.data, 1)
            
            predictions.extend(predicted.cpu().numpy())
            labels.extend(batch_labels.numpy())
            latent_representations.append(latent_sequence.cpu().numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = accuracy_score(labels, predictions)
    
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, predictions))
    
    return accuracy, predictions, labels, latent_representations