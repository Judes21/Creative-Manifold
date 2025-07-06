import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_baseline_interval_model(model, train_loader, test_loader, num_epochs=100, 
                                  lr=0.001, recon_weight=0.3, device='cuda'):
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
            reconstructions, latent_sequence, logits = model(features)
            
            loss_recon = criterion_recon(reconstructions, features)
            loss_class = criterion_class(logits, labels)
            loss = recon_weight * loss_recon + (1 - recon_weight) * loss_class
            
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
                
                reconstructions, latent_sequence, logits = model(features)
                
                loss_recon = criterion_recon(reconstructions, features)
                loss_class = criterion_class(logits, labels)
                loss = recon_weight * loss_recon + (1 - recon_weight) * loss_class
                
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
            
            _, _, logits = model(features)
            _, predicted = torch.max(logits.data, 1)
            
            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    print("\nFinal Classification Report:")
    print(classification_report(val_labels, val_predictions, 
                              target_names=['Rest', 'Improv', 'Scale']))
    
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
            
            reconstructions, latent_sequence, logits = model(features)
            
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