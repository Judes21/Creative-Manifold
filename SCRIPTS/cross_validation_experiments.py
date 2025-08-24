import os
import torch
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from SCRIPTS.config import (
    CROSS_VALIDATION_RESULTS_DIR, COMBINED_FNIRS, COMBINED_SCATTERING, BASELINE_RESULTS_DIR, TOPOLOGICAL_RESULTS_DIR, CURVATURE_RESULTS_DIR, RNN_RESULTS_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, BASELINE_INTERVAL_RESULTS_DIR, CHECKPOINTS_DIR, COMBINED_MODEL_RESULTS_DIR
)
from SCRIPTS.dataprep import prepare_data, prepare_interval_data
from SCRIPTS.models import BrainStateClassifier, BrainStateFullModel, AttentionFullModel
from SCRIPTS.train import train_and_evaluate, train_autoencoder, train_attention_model
from SCRIPTS.topological_model import TopologicalModel
from SCRIPTS.topological_training import train_topological_model
from SCRIPTS.rnn_model import AttentionLSTMClassifier
from SCRIPTS.rnn_training import train_rnn_model
from SCRIPTS.curvature_models import CurvatureModel
from SCRIPTS.curvature_training import train_curvature_model
from SCRIPTS.baseline_interval_model import BaselineIntervalModel
from SCRIPTS.baseline_interval_training import train_baseline_interval_model
from SCRIPTS.combined_model_v2 import CombinedModelV2
from SCRIPTS.combined_training_v2 import train_combined_model_v2

#Point-wise baseline MLP
def run_baseline_cross_validation(data_path, feature_prefix, split_type, num_trials=5, input_dim=768, num_epochs=DEFAULT_EPOCHS):
    accuracies = []
    confusion_matrices = []
    classification_reports = []
    
    for trial in range(num_trials):
        print(f"\n=== Baseline MLP Trial {trial+1}/{num_trials} ({split_type} split) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader = prepare_data(
            data_path=data_path,
            feature_prefix=feature_prefix,
            include_metadata=False,
            split_type=split_type,
            random_state=random_seed
        )
        
        model = BrainStateClassifier(input_dim=input_dim)
        model, preds, labels, metrics = train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            num_epochs=num_epochs
        )

        accuracy = accuracy_score(labels, preds)
        accuracies.append(accuracy)
        cm = confusion_matrix(labels, preds)
        confusion_matrices.append(cm)
        
        cr = classification_report(labels, preds, 
                                  target_names=['Rest', 'Improv', 'Scale'],
                                  output_dict=True)
        classification_reports.append(cr)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_cm = np.mean(confusion_matrices, axis=0)

    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'confusion_matrices': confusion_matrices,
        'avg_confusion_matrix': avg_cm,
        'classification_reports': classification_reports,
        'split_type': split_type
    }
    
    save_path = BASELINE_RESULTS_DIR / f'baseline_MLP_{feature_prefix.strip("_")}_{split_type}_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Baseline Results ({split_type} split) ===")
    print(f"Accuracy: {mean_accuracy:.2%} ± {std_accuracy:.2%}")
    
    return results


# Baseline autoencoder
def run_autoencoder_cross_validation(data_path, feature_prefix, split_type, latent_dim=48, num_trials=5, num_epochs=DEFAULT_EPOCHS):
    accuracies = []
    histories = []
    
    for trial in range(num_trials):
        print(f"\n=== Autoencoder Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader = prepare_data(
            data_path=data_path,
            feature_prefix=feature_prefix,
            include_metadata=True,  
            split_type=split_type,
            random_state=random_seed
        )
        
        model, history = train_autoencoder(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            latent_dim=latent_dim,
            split_name=f"CV{trial+1}-{split_type}"
        )
        
        accuracies.append(history['test_accuracy'])
        histories.append(history)
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'split_type': split_type,
        'latent_dim': latent_dim
    }
    
    save_path = BASELINE_RESULTS_DIR / f'baseline_autoencoder_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Autoencoder Results ({split_type} split, {latent_dim}D) ===")
    print(f"Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    return results


# Autoencoder w/ attention
def run_attention_cross_validation(data_path, feature_prefix, split_type, latent_dim=48, num_trials=5, num_epochs=DEFAULT_EPOCHS):
    accuracies = []
    histories = []
    attention_weights = []
    
    for trial in range(num_trials):
        print(f"\n=== Attention Model Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader = prepare_data(
            data_path=data_path,
            feature_prefix=feature_prefix,
            include_metadata=True,
            split_type=split_type,
            random_state=random_seed
        )
        
        model, history = train_attention_model(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            latent_dim=latent_dim,
            split_name=f"CV{trial+1}-{split_type}"
        )
        
        accuracies.append(history['test_accuracy'])
        histories.append(history)
        if 'attention_weights' in history:
            attention_weights.append(history['attention_weights'][-1])  # Final weights
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'attention_weights': attention_weights,
        'split_type': split_type,
        'latent_dim': latent_dim
    }
    
    save_path = BASELINE_RESULTS_DIR / f'baseline_attention_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Attention Results ({split_type} split, {latent_dim}D) ===")
    print(f"Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    
    return results


# Topological model
def run_topological_cross_validation(data_path, split_type, latent_dim=48, num_trials=5, num_epochs=DEFAULT_EPOCHS, save_best_model=False):
    accuracies = []
    histories = []
    
    for trial in range(num_trials):
        print(f"\n=== Topological Model Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader, info = prepare_interval_data(
            scattering_data_path=data_path,
            split_type=split_type,
            batch_size=DEFAULT_BATCH_SIZE,
            random_state=random_seed
        )
        
        model = TopologicalModel(input_dim=768, latent_dim=latent_dim, num_classes=3)
        model, history, val_preds, val_labels = train_topological_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=DEFAULT_LEARNING_RATE,
            split_type=split_type,
            latent_dim=latent_dim,
            trial=trial
        )
        
        accuracy = 100 * np.mean(np.array(val_preds) == np.array(val_labels))
        accuracies.append(accuracy)
        histories.append(history)
        
        if save_best_model:
            checkpoint_path = CHECKPOINTS_DIR / f'topological_{split_type}_{latent_dim}d_trial{trial}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'trial': trial,
                'split_type': split_type,
                'latent_dim': latent_dim,
                'model_type': 'topological'
            }, checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'split_type': split_type,
        'latent_dim': latent_dim
    }
    
    save_path = TOPOLOGICAL_RESULTS_DIR / f'topological_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Topological Results ({split_type} split, {latent_dim}D) ===")
    print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
    
    return results


# RNN model
def run_rnn_cross_validation(data_path, split_type, num_trials=5, num_epochs=DEFAULT_EPOCHS, latent_dim=48, save_best_model=False):
    accuracies = []
    histories = []

    for trial in range(num_trials):
        print(f"\n=== RNN Model Trial {trial+1}/{num_trials} ({split_type} split, latent_dim={latent_dim}) ===")
        random_seed = 42 + trial

        train_loader, test_loader, info = prepare_interval_data(
            scattering_data_path=data_path,
            split_type=split_type,
            batch_size=DEFAULT_BATCH_SIZE,
            random_state=random_seed
        )

        model = AttentionLSTMClassifier(
            input_dim=768,
            latent_dim=latent_dim,
            hidden_dim=128
        )

        model, history = train_rnn_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=DEFAULT_LEARNING_RATE,
            split_type=split_type,
            latent_dim=latent_dim,
            trial=trial
        )

        accuracies.append(history['val_acc'][-1])
        histories.append(history)
        
        if save_best_model:
            checkpoint_path = CHECKPOINTS_DIR / f'rnn_{split_type}_{latent_dim}d_trial{trial}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': history['val_acc'][-1],
                'trial': trial,
                'split_type': split_type,
                'latent_dim': latent_dim,
                'model_type': 'rnn'
            }, checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'split_type': split_type,
        'latent_dim': latent_dim
    }

    save_path = RNN_RESULTS_DIR / f'rnn_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")

    print(f"\n=== RNN Results ({split_type} split, latent_dim={latent_dim}) ===")
    print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")

    return results

# Curvature model
def run_curvature_cross_validation(data_path, split_type, latent_dim=48, num_trials=5, num_epochs=DEFAULT_EPOCHS, save_best_model=False):
    accuracies = []
    histories = []
    
    for trial in range(num_trials):
        print(f"\n=== Curvature Model Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader, info = prepare_interval_data(
            scattering_data_path=data_path,
            split_type=split_type,
            batch_size=16, 
            random_state=random_seed
        )
        
        train_intervals = []
        for batch in train_loader:
            features, labels, subjects, intervals = batch
            for i in range(len(features)):
                train_intervals.append((
                    features[i].numpy(),
                    labels[i].item(),
                    subjects[i],
                    intervals[i]
                ))
        
        test_intervals = []
        for batch in test_loader:
            features, labels, subjects, intervals = batch
            for i in range(len(features)):
                test_intervals.append((
                    features[i].numpy(),
                    labels[i].item(),
                    subjects[i],
                    intervals[i]
                ))
        
        model, history = train_curvature_model(
            train_intervals=train_intervals,
            test_intervals=test_intervals,
            latent_dim=latent_dim,
            num_epochs=num_epochs,
            batch_size=16,
            learning_rate=DEFAULT_LEARNING_RATE,
            split_type=split_type,
            trial=trial
        )
        
        accuracies.append(history['test_acc'][-1])
        histories.append(history)
        
        if save_best_model:
            checkpoint_path = CHECKPOINTS_DIR / f'curvature_{split_type}_{latent_dim}d_trial{trial}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': history['test_acc'][-1],
                'trial': trial,
                'split_type': split_type,
                'latent_dim': latent_dim,
                'model_type': 'curvature'
            }, checkpoint_path)
            print(f"Model checkpoint saved: {checkpoint_path}")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'split_type': split_type,
        'latent_dim': latent_dim
    }
    
    save_path = CURVATURE_RESULTS_DIR / f'curvature_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Curvature Results ({split_type} split, {latent_dim}D) ===")
    print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
    
    return results


# Baseline interval model
def run_baseline_interval_cross_validation(data_path, split_type, latent_dim=48, num_trials=5, 
                                         num_epochs=DEFAULT_EPOCHS, recon_weight=0.3):
    
    accuracies = []
    histories = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for trial in range(num_trials):
        print(f"\n=== Baseline Interval Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
        random_seed = 42 + trial
        
        train_loader, test_loader, info = prepare_interval_data(
            scattering_data_path=data_path,
            split_type=split_type,
            batch_size=16,
            random_state=random_seed
        )
    
        model = BaselineIntervalModel(
            input_dim=768,
            latent_dim=latent_dim,
            seq_len=400,
            num_classes=3
        )

        model, history = train_baseline_interval_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            lr=DEFAULT_LEARNING_RATE,
            recon_weight=recon_weight,
            device=device,
            split_type=split_type,
            latent_dim=latent_dim,
            trial=trial
        )
        
        final_accuracy = history['val_acc'][-1]
        accuracies.append(final_accuracy)
        histories.append(history)
        
        print(f"Trial {trial+1} Final Test Accuracy: {final_accuracy:.1f}%")
    
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    results = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'histories': histories,
        'split_type': split_type,
        'latent_dim': latent_dim
    }
    
    save_path = BASELINE_INTERVAL_RESULTS_DIR / f'baseline_interval_{split_type}_{latent_dim}d_cv_results.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to: {save_path}")
    
    print(f"\n=== Baseline Interval Results ({split_type} split, {latent_dim}D) ===")
    print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
    
    return results


# Combined model
def run_combined_cross_validation(data_path, split_types=['subject', 'time'], 
                                 latent_dims=[8, 48], num_trials=5, num_epochs=50, save_best_model=False):
    
    all_results = {}
    
    for split_type in split_types:
        for latent_dim in latent_dims:
            accuracies = []
            histories = []
            
            for trial in range(num_trials):
                print(f"\n=== Combined Model Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
                random_seed = 42 + trial
                
                train_loader, test_loader, info = prepare_interval_data(
                    scattering_data_path=data_path,
                    split_type=split_type,
                    batch_size=DEFAULT_BATCH_SIZE,
                    random_state=random_seed
                )
                
                model, history = train_combined_model(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    latent_dim=latent_dim,
                    split_type=split_type,
                    num_epochs=num_epochs,
                    lr=DEFAULT_LEARNING_RATE,
                    trial=trial
                )
                
                accuracies.append(history['val_acc'][-1])
                histories.append(history)
                
                if save_best_model:
                    checkpoint_path = CHECKPOINTS_DIR / f'combined_{split_type}_{latent_dim}d_trial{trial}.pth'
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'accuracy': history['val_acc'][-1],
                        'trial': trial,
                        'split_type': split_type,
                        'latent_dim': latent_dim,
                        'model_type': 'combined'
                    }, checkpoint_path)
                    print(f"Model checkpoint saved: {checkpoint_path}")
            
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            results = {
                'accuracies': accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'histories': histories,
                'split_type': split_type,
                'latent_dim': latent_dim
            }
            
            save_path = COMBINED_MODEL_RESULTS_DIR / f'combined_{split_type}_{latent_dim}d_cv_results.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to: {save_path}")
            
            print(f"\n=== Combined Results ({split_type} split, {latent_dim}D) ===")
            print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
            
            # Store in all_results
            result_key = f"{split_type}_{latent_dim}d"
            all_results[result_key] = results
    
    return all_results


# Combined model V2
def run_combined_v2_cross_validation(data_path, split_types=['subject', 'time'], 
                                    latent_dims=[8, 48], num_trials=5, num_epochs=100):
    all_results = {}
    
    for split_type in split_types:
        for latent_dim in latent_dims:
            accuracies = []
            histories = []
            
            for trial in range(num_trials):
                print(f"\n=== Combined Model V2 Trial {trial+1}/{num_trials} ({split_type} split, {latent_dim}D) ===")
                random_seed = 42 + trial
                
                train_loader, test_loader, info = prepare_interval_data(
                    scattering_data_path=data_path,
                    split_type=split_type,
                    batch_size=DEFAULT_BATCH_SIZE,
                    random_state=random_seed
                )
                
                model, history = train_combined_model_v2(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    latent_dim=latent_dim,
                    split_type=split_type,
                    num_epochs=num_epochs,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    trial=trial
                )
                
                accuracies.append(history['final_accuracy'])
                histories.append(history)
            
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            results = {
                'accuracies': accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'histories': histories,
                'split_type': split_type,
                'latent_dim': latent_dim
            }
            
            save_path = COMBINED_MODEL_RESULTS_DIR / f'combined_v2_{split_type}_{latent_dim}d_cv_results.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to: {save_path}")
            
            print(f"\n=== Combined V2 Results ({split_type} split, {latent_dim}D) ===")
            print(f"Accuracy: {mean_accuracy:.1f}% ± {std_accuracy:.1f}%")
            
            # Store in all_results
            result_key = f"{split_type}_{latent_dim}d"
            all_results[result_key] = results
    
    return all_results