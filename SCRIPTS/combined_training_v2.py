import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
from SCRIPTS.config import (
    COMBINED_ATTENTION_DIR, COMBINED_MODEL_RESULTS_DIR, create_directories,
    LOSS_WEIGHTS, TRAINING_CONFIG
)
from SCRIPTS.combined_model_v2 import CombinedModelV2


def save_combined_attention_weights(model, split_type, latent_dim=None, trial=None):
    create_directories()
    combined_attention = model.get_combined_attention_weights()
    
    filename_parts = ['attention', split_type, 'combined']
    if latent_dim is not None:
        filename_parts.append(f'{latent_dim}D')
    if trial is not None:
        filename_parts.append(f'trial{trial}')
    filename = '_'.join(filename_parts) + '.pth'
    save_path = COMBINED_ATTENTION_DIR / filename
    
    torch.save({
        'attention_weights': combined_attention,
        'model_type': 'combined',
        'split_type': split_type,
        'latent_dim': latent_dim,
        'trial': trial
    }, save_path)
    
    print(f"✓ Combined attention weights saved to {save_path}")
    return save_path


def train_combined_model_v2(
    train_loader,
    test_loader,
    latent_dim=48,
    split_type='subject',
    num_epochs=None,
    learning_rate=None,
    device=None,
    trial=None
):
    num_epochs = num_epochs or TRAINING_CONFIG['epochs']
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Training Combined Model V2 with end-to-end optimization")
    print(f"Latent dim: {latent_dim}, Split: {split_type}, Device: {device}")
    
    model = CombinedModelV2(
        input_dim=768,
        latent_dim=latent_dim,
        num_classes=3,
        dropout=TRAINING_CONFIG['dropout']
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    assert trainable_params == total_params, "Some parameters are frozen!"
    
    # Optimizers and loss functions
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINING_CONFIG.get('weight_decay', 1e-4)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=10, factor=0.5, verbose=True
    )
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'train_class_loss': [], 'train_recon_loss': [],
        'val_class_loss': [], 'val_recon_loss': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nTraining for {num_epochs} epochs with joint optimization...")
    print(f"Loss weights: Classification={LOSS_WEIGHTS['classification']:.1f}, "
          f"Reconstruction={LOSS_WEIGHTS['reconstruction']:.1f}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_class_loss = 0
        train_recon_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features, labels, _, _ = batch
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            model_output = model(features)
            logits = model_output['logits']
            reconstruction = model_output['aux']['reconstruction']
            
            loss_class = criterion_class(logits, labels)
            loss_recon = criterion_recon(reconstruction, features.mean(dim=1))  # Compare to average
            losses = {
                'classification': loss_class,
                'reconstruction': loss_recon
            }
            total_loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
            
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            train_loss += total_loss.item()
            train_class_loss += loss_class.item()
            train_recon_loss += loss_recon.item()
            
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        val_loss = 0
        val_class_loss = 0
        val_recon_loss = 0
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
                logits = model_output['logits']
                reconstruction = model_output['aux']['reconstruction']
                
                # Compute losses
                loss_class = criterion_class(logits, labels)
                loss_recon = criterion_recon(reconstruction, features.mean(dim=1))
                
                losses = {
                    'classification': loss_class,
                    'reconstruction': loss_recon
                }
                total_loss = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items())
                
                val_loss += total_loss.item()
                val_class_loss += loss_class.item()
                val_recon_loss += loss_recon.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_class_loss'].append(train_class_loss / len(train_loader))
        history['train_recon_loss'].append(train_recon_loss / len(train_loader))
        
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_acc'].append(val_acc)
        history['val_class_loss'].append(val_class_loss / len(test_loader))
        history['val_recon_loss'].append(val_recon_loss / len(test_loader))
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Progress update
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, "
                  f"Acc: {train_acc:.2f}%, "
                  f"Class: {train_class_loss/len(train_loader):.4f}, "
                  f"Recon: {train_recon_loss/len(train_loader):.4f}")
            print(f"  Val   - Loss: {val_loss/len(test_loader):.4f}, "
                  f"Acc: {val_acc:.2f}%, "
                  f"Class: {val_class_loss/len(test_loader):.4f}, "
                  f"Recon: {val_recon_loss/len(test_loader):.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation
    model.eval()
    final_predictions = []
    final_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, labels, _, _ = batch
            features = features.to(device)
            
            model_output = model(features)
            logits = model_output['logits']
            _, predicted = torch.max(logits.data, 1)
            
            final_predictions.extend(predicted.cpu().numpy())
            final_labels.extend(labels.numpy())
    
    final_accuracy = 100 * accuracy_score(final_labels, final_predictions)
    
    print(f"\nFinal Combined Model Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"  Final Test Accuracy: {final_accuracy:.1f}%")
    
    print("\nClassification Report:")
    print(classification_report(final_labels, final_predictions,
                              target_names=['Rest', 'Improv', 'Scale']))
    
    save_combined_attention_weights(model, split_type, latent_dim, trial)
    
    history['final_accuracy'] = final_accuracy
    history['best_val_accuracy'] = best_val_acc
    history['classification_report'] = classification_report(
        final_labels, final_predictions,
        target_names=['Rest', 'Improv', 'Scale'],
        output_dict=True
    )
    
    return model, history