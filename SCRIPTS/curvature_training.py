import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix


class IntervalDataset(Dataset):
    def __init__(self, intervals_list):
        self.intervals = intervals_list

    def __len__(self):
        return len(self.intervals)
    
    def __getitem__(self, idx):
        features, label, subject_id, interval_name = self.intervals[idx]
        return (
            torch.FloatTensor(features),  # (400, 768)
            torch.LongTensor([label]).squeeze(),
            subject_id,
            interval_name
        )

def train_curvature_model(
    train_intervals, 
    test_intervals,
    latent_dim=48,
    num_epochs=50,
    batch_size=16,
    learning_rate=0.001,
    recon_weight=0.1,
    device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and loaders
    train_dataset = IntervalDataset(train_intervals)
    test_dataset = IntervalDataset(test_intervals)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
############## 

    from SCRIPTS.curvature_models import CurvatureModel
    
    model = CurvatureModel(
        input_dim=768,
        latent_dim=latent_dim,
        num_classes=3
    ).to(device)
    
    class_criterion = nn.CrossEntropyLoss()
    recon_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'train_class_loss': [],
        'train_recon_loss': [],
        'curvature_stats': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_class_loss = 0
        train_recon_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels, _, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            class_logits, reconstruction = model(features)
            
            class_loss = class_criterion(class_logits, labels)
            recon_loss = recon_criterion(reconstruction, features)
            total_loss = class_loss + recon_weight * recon_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_class_loss += class_loss.item()
            train_recon_loss += recon_loss.item()
            
            _, predicted = torch.max(class_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_curvatures = []
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for features, labels, _, _ in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                class_logits, reconstruction, latent, curvatures = model(features, return_all=True)
                
                # Compute losses
                class_loss = class_criterion(class_logits, labels)
                recon_loss = recon_criterion(reconstruction, features)
                total_loss = class_loss + recon_weight * recon_loss
                
                test_loss += total_loss.item()
                
                _, predicted = torch.max(class_logits.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Store for final evaluation
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
                # Store curvatures for analysis
                all_curvatures.append(curvatures.cpu().numpy())
        
        # Compute metrics
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)
        avg_train_class_loss = train_class_loss / len(train_loader)
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        history['train_class_loss'].append(avg_train_class_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        
        # Compute curvature statistics
        all_curvatures = np.concatenate(all_curvatures)
        curv_stats = {
            'mean': np.mean(all_curvatures),
            'std': np.std(all_curvatures),
            'max': np.max(all_curvatures),
            'min': np.min(all_curvatures)
        }
        history['curvature_stats'].append(curv_stats)
        
        # Progress update
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f} (Class: {avg_train_class_loss:.4f}, Recon: {avg_train_recon_loss:.4f})')
            print(f'  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            print(f'  Curvature: mean={curv_stats["mean"]:.4f}, std={curv_stats["std"]:.4f}')
    
    # Final evaluation
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"  Test Accuracy: {history['test_acc'][-1]:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=['Rest', 'Improv', 'Scale']))
    
    return model, history

# Evaluate the trained model and return detailed results
def evaluate_model(model, test_loader, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_labels = []
    all_curvatures = []
    all_latents = []
    
    with torch.no_grad():
        for features, labels, subjects, intervals in test_loader:
            features = features.to(device)
            
            class_logits, _, latent, curvatures = model(features, return_all=True)
            
            _, predicted = torch.max(class_logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_curvatures.append(curvatures.cpu().numpy())
            all_latents.append(latent.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_curvatures = np.concatenate(all_curvatures)
    all_latents = np.concatenate(all_latents)
    
    # Calculate accuracy
    accuracy = 100 * np.mean(all_preds == all_labels)
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'curvatures': all_curvatures,
        'latents': all_latents
    }