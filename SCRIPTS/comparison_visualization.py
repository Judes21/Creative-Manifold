import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns

from SCRIPTS.config import (
    BASELINE_RESULTS_DIR, 
    TOPOLOGICAL_RESULTS_DIR, 
    CURVATURE_RESULTS_DIR, 
    RNN_RESULTS_DIR, 
    FINAL_VISUALIZATION_RESULTS_DIR,
    BASELINE_INTERVAL_RESULTS_DIR
)


def compare_pointwise_methods(output_path=None):
    if output_path is None:
        output_path = FINAL_VISUALIZATION_RESULTS_DIR / "method_comparisons" / "pointwise_comparison.png"

    #Group methods by colors
    method_groups = {
        'Baseline': {
            'methods': ['Raw fNIRS', 'Scattering'],
            'colors': ['#1f77b4', '#2ca02c']
        },
        'Autoencoder': {
            'methods': ['Autoencoder 8D', 'Autoencoder 48D'],
            'colors': ['#ff7f0e', '#d62728']
        },
        'Attention': {
            'methods': ['Attention 8D', 'Attention 48D'],
            'colors': ['#9467bd', '#8c564b']
        }
    }
    
    methods_ordered = []
    all_colors = []
    for group in ['Baseline', 'Autoencoder', 'Attention']:
        methods_ordered.extend(method_groups[group]['methods'])
        all_colors.extend(method_groups[group]['colors'])
    
    results = {method: {'subject': None, 'time': None} for method in methods_ordered}
    
    # File mappings
    file_mappings = {
        'Raw fNIRS': ('baseline_MLP_node', BASELINE_RESULTS_DIR),
        'Scattering': ('baseline_MLP_feature', BASELINE_RESULTS_DIR),
        'Autoencoder 8D': ('baseline_autoencoder', BASELINE_RESULTS_DIR, '_8d'),
        'Autoencoder 48D': ('baseline_autoencoder', BASELINE_RESULTS_DIR, '_48d'),
        'Attention 8D': ('baseline_attention', BASELINE_RESULTS_DIR, '_8d'),
        'Attention 48D': ('baseline_attention', BASELINE_RESULTS_DIR, '_48d')
    }
    
    for method, file_info in file_mappings.items():
        if len(file_info) == 2:
            prefix, directory = file_info
            suffix = ''
        else:
            prefix, directory, suffix = file_info
            
        for split_type in ['subject', 'time']:
            filepath = directory / f'{prefix}_{split_type}{suffix}_cv_results.pkl'
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                results[method][split_type] = data
            else:
                print(f"Warning: {filepath} not found")
    
    subject_means = []
    subject_stds = []
    time_means = []
    time_stds = []
    
    for method in methods_ordered:
        if results[method]['subject'] is not None:
            subj_mean = results[method]['subject']['mean_accuracy']
            subj_std = results[method]['subject']['std_accuracy']
            # Convert to percentage if needed
            if subj_mean <= 1:
                subj_mean *= 100
                subj_std *= 100
            subject_means.append(subj_mean)
            subject_stds.append(subj_std)
        else:
            subject_means.append(0)
            subject_stds.append(0)
            
        if results[method]['time'] is not None:
            time_mean = results[method]['time']['mean_accuracy']
            time_std = results[method]['time']['std_accuracy']
            if time_mean <= 1:
                time_mean *= 100
                time_std *= 100
            time_means.append(time_mean)
            time_stds.append(time_std)
        else:
            time_means.append(0)
            time_stds.append(0)
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15, 
                          left=0.08, right=0.92, top=0.85, bottom=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(methods_ordered))
    width = 0.75
    
    # Subject-withheld
    bars1 = ax1.bar(x, subject_means, width, 
                     yerr=subject_stds, 
                     capsize=6,
                     color=all_colors, 
                     edgecolor='white', 
                     linewidth=1.5,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    
    ax1.set_title('Subject-Withheld Cross-Validation', fontsize=14, pad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    for bar, val, std in zip(bars1, subject_means, subject_stds):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + std + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    #Time-withheld
    bars2 = ax2.bar(x, time_means, width, 
                     yerr=time_stds, 
                     capsize=6,
                     color=all_colors, 
                     edgecolor='white', 
                     linewidth=1.5,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    
    ax2.set_title('Time-Withheld Cross-Validation', fontsize=14, pad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    for bar, val, std in zip(bars2, time_means, time_stds):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + std + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    group_positions = {
        'Baseline': (0, 2),
        'Autoencoder': (2, 4),
        'Attention': (4, 6)
    }
    
    for ax in [ax1, ax2]:
        for i, (group, (start, end)) in enumerate(group_positions.items()):
            color = ['#f8f8f8', '#f0f0f0', '#f8f8f8'][i % 3]
            ax.axvspan(start - 0.5, end - 0.5, alpha=0.5, color=color, zorder=0)
    
    fig.suptitle('Point-wise Method Performance Comparison', fontsize=16, fontweight='bold')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Point-wise comparison saved to: {output_path}")
    return results


def compare_interval_methods(baseline_interval_results=None, output_path=None):
    if output_path is None:
        output_path = FINAL_VISUALIZATION_RESULTS_DIR / "method_comparisons" / "interval_comparison.png"
    
    # Group methods by colors
    method_groups = {
        'Baseline Interval': {
            'methods': ['Baseline Interval 8D', 'Baseline Interval 48D'],
            'colors': ['#1f77b4', '#2ca02c']
        },
        'Topological': {
            'methods': ['Topological 8D', 'Topological 48D'],
            'colors': ['#ff7f0e', '#d62728']
        },
        'RNN': {
            'methods': ['RNN 8D', 'RNN 48D'],
            'colors': ['#9467bd', '#8c564b']
        },
        'Curvature': {
            'methods': ['Curvature 8D', 'Curvature 48D'],
            'colors': ['#e377c2', '#7f7f7f']
        }
    }
    
    methods_ordered = []
    all_colors = []
    for group in ['Baseline Interval', 'Topological', 'RNN', 'Curvature']:
        methods_ordered.extend(method_groups[group]['methods'])
        all_colors.extend(method_groups[group]['colors'])
    
    results = {method: {'subject': None, 'time': None} for method in methods_ordered}

    if baseline_interval_results:
        results['Baseline Interval 8D'] = {
            'subject': baseline_interval_results[8]['subject'],
            'time': baseline_interval_results[8]['time']
        }
        results['Baseline Interval 48D'] = {
            'subject': baseline_interval_results[48]['subject'],
            'time': baseline_interval_results[48]['time']
        }
    else:
        baseline_file = BASELINE_RESULTS_DIR / 'baseline_interval_cv_results.pkl'
        if baseline_file.exists():
            with open(baseline_file, 'rb') as f:
                baseline_data = pickle.load(f)
            if isinstance(baseline_data, dict) and 8 in baseline_data:
                results['Baseline Interval 8D'] = {
                    'subject': baseline_data[8]['subject'],
                    'time': baseline_data[8]['time']
                }
                results['Baseline Interval 48D'] = {
                    'subject': baseline_data[48]['subject'],
                    'time': baseline_data[48]['time']
                }
    
    # File mappings for other methods
    interval_methods = {
        'Topological 8D': (TOPOLOGICAL_RESULTS_DIR, 'topological_subject_8d_cv_results.pkl', 'topological_time_8d_cv_results.pkl'),
        'Topological 48D': (TOPOLOGICAL_RESULTS_DIR, 'topological_subject_48d_cv_results.pkl', 'topological_time_48d_cv_results.pkl'),
        'RNN 8D': (RNN_RESULTS_DIR, 'rnn_subject_8d_cv_results.pkl', 'rnn_time_8d_cv_results.pkl'),
        'RNN 48D': (RNN_RESULTS_DIR, 'rnn_subject_48d_cv_results.pkl', 'rnn_time_48d_cv_results.pkl'),
        'Curvature 8D': (CURVATURE_RESULTS_DIR, 'curvature_subject_8d_cv_results.pkl', 'curvature_time_8d_cv_results.pkl'),
        'Curvature 48D': (CURVATURE_RESULTS_DIR, 'curvature_subject_48d_cv_results.pkl', 'curvature_time_48d_cv_results.pkl')
    }
    
    for method_name, (results_dir, subject_file, time_file) in interval_methods.items():
        subject_path = results_dir / subject_file
        if subject_path.exists():
            with open(subject_path, 'rb') as f:
                results[method_name]['subject'] = pickle.load(f)
        else:
            print(f"Warning: {subject_path} not found")
        
        time_path = results_dir / time_file
        if time_path.exists():
            with open(time_path, 'rb') as f:
                results[method_name]['time'] = pickle.load(f)
        else:
            print(f"Warning: {time_path} not found")
    
    subject_means = []
    subject_stds = []
    time_means = []
    time_stds = []
    
    for method in methods_ordered:
        if results[method]['subject'] is not None:
            subj_mean = results[method]['subject']['mean_accuracy']
            subj_std = results[method]['subject']['std_accuracy']
            subject_means.append(subj_mean)
            subject_stds.append(subj_std)
        else:
            subject_means.append(0)
            subject_stds.append(0)
            
        if results[method]['time'] is not None:
            time_mean = results[method]['time']['mean_accuracy']
            time_std = results[method]['time']['std_accuracy']
            time_means.append(time_mean)
            time_stds.append(time_std)
        else:
            time_means.append(0)
            time_stds.append(0)
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.15,
                          left=0.06, right=0.94, top=0.85, bottom=0.15)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    x = np.arange(len(methods_ordered))
    width = 0.75

    # Subject-withheld
    bars1 = ax1.bar(x, subject_means, width, 
                     yerr=subject_stds, 
                     capsize=6,
                     color=all_colors, 
                     edgecolor='white', 
                     linewidth=1.5,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    
    ax1.set_title('Subject-Withheld Cross-Validation', fontsize=14, pad=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    for bar, val, std in zip(bars1, subject_means, subject_stds):
        if val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + std + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Time-withheld
    bars2 = ax2.bar(x, time_means, width, 
                     yerr=time_stds, 
                     capsize=6,
                     color=all_colors, 
                     edgecolor='white', 
                     linewidth=1.5,
                     error_kw={'linewidth': 1.5, 'ecolor': 'black', 'alpha': 0.7})
    
    ax2.set_title('Time-Withheld Cross-Validation', fontsize=14, pad=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods_ordered, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)
    
    for bar, val, std in zip(bars2, time_means, time_stds):
        if val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + std + 1.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    group_positions = {
        'Baseline Interval': (0, 2),
        'Topological': (2, 4),
        'RNN': (4, 6),
        'Curvature': (6, 8)
    }
    
    for ax in [ax1, ax2]:
        for i, (group, (start, end)) in enumerate(group_positions.items()):
            color = ['#f8f8f8', '#f0f0f0', '#f8f8f8', '#f0f0f0'][i % 4]
            ax.axvspan(start - 0.5, end - 0.5, alpha=0.5, color=color, zorder=0)
    
    fig.suptitle('Interval-based Method Performance Comparison', fontsize=16, fontweight='bold')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Interval comparison saved to: {output_path}")
    return results