import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import torch
import phate
import tphate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

from SCRIPTS.config import (
    TASK_SEGMENTS, EXTENDED_LATENT_RESULTS_DIR, LATENT_VIZ_RESULTS_DIR,
    ALPHA, MARKER_SIZE, T_VALUE
)
from SCRIPTS.dataprep import compute_tphate_embedding


def load_latent_embeddings(results_dir, latent_dims):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_dict = {}
    
    for latent_dim in latent_dims:
        embeddings_dict[latent_dim] = {}
        
        # Load data
        subject_path = results_dir / f'subject_withheld_model_latent{latent_dim}.pth'
        if subject_path.exists():
            subject_results = torch.load(subject_path, map_location=device)
            all_subject_embeddings = []
            all_subject_labels = []
            all_subject_ids = []
            
            for subject, embs in subject_results['history']['embeddings'].items():
                all_subject_embeddings.append(np.array(embs))
                all_subject_labels.extend(subject_results['history']['labels'][subject])
                all_subject_ids.extend([subject] * len(subject_results['history']['labels'][subject]))
            
            subject_latent_space = np.vstack(all_subject_embeddings)
            embeddings_dict[latent_dim]['subject'] = {
                'embeddings': subject_latent_space,
                'labels': all_subject_labels,
                'subjects': all_subject_ids
            }
        
        time_path = results_dir / f'time_withheld_model_latent{latent_dim}.pth'
        if time_path.exists():
            time_results = torch.load(time_path, map_location=device)
            all_time_embeddings = []
            all_time_labels = []
            all_time_ids = []
            
            for subject, embs in time_results['history']['embeddings'].items():
                all_time_embeddings.append(np.array(embs))
                all_time_labels.extend(time_results['history']['labels'][subject])
                all_time_ids.extend([subject] * len(time_results['history']['labels'][subject]))
            
            time_latent_space = np.vstack(all_time_embeddings)
            embeddings_dict[latent_dim]['time'] = {
                'embeddings': time_latent_space,
                'labels': all_time_labels,
                'subjects': all_time_ids
            }
    return embeddings_dict


def visualize_latent_space(latent_space, labels, subjects, split_type, latent_dim, include_rest=True, output_dir=None):
    if output_dir is None:
        output_dir = LATENT_VIZ_RESULTS_DIR
        
    latent_space = np.array(latent_space)
    labels = np.array(labels)
    subjects = np.array(subjects)
    
    if not include_rest:
        non_rest_mask = labels != 0
        latent_space = latent_space[non_rest_mask]
        labels = labels[non_rest_mask]
        subjects = subjects[non_rest_mask]
        print(f"Filtered shape (rest excluded): {latent_space.shape}")
        
    print(f"PHATE embedding for {split_type} split, {latent_dim}D latent space...")
    
    # Compute PHATE embedding
    phate_op = phate.PHATE(
        n_components=3,
        t=T_VALUE,
        random_state=42,
        n_jobs=-1,
        verbose=True
    )
    embedding = phate_op.fit_transform(latent_space)
    
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    if include_rest:
        task_colors = ['#2ecc71', '#e74c3c', '#3498db'] 
        task_names = ['Rest', 'Improv', 'Scale']
    else:
        task_colors = ['#e74c3c', '#3498db']
        task_names = ['Improv', 'Scale']

    # Plot by task
    for i, (task, color) in enumerate(zip(task_names, task_colors)):
        label_idx = i if include_rest else i+1
        mask = labels == label_idx
        ax1.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=color,
            label=task,
            alpha=ALPHA,
            s=MARKER_SIZE
        )
    
    title_suffix = " (Rest Excluded)" if not include_rest else ""
    ax1.set_title(f'{split_type.capitalize()} Split: {latent_dim}D Latent Space by Task{title_suffix}', fontsize=14)
    ax1.set_xlabel('PHATE 1', fontsize=12)
    ax1.set_ylabel('PHATE 2', fontsize=12)
    ax1.set_zlabel('PHATE 3', fontsize=12)
    ax1.legend(fontsize=10)
    
    # Plot by subject
    unique_subjects = sorted(np.unique(subjects))
    n_subjects = len(unique_subjects)
    subject_colors = [plt.cm.hsv(i/n_subjects) for i in range(n_subjects)]

    for i, subject in enumerate(unique_subjects):
        mask = subjects == subject
        ax2.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            c=[subject_colors[i]],
            label=f'Subject {subject}',
            alpha=ALPHA,
            s=MARKER_SIZE
        )
    
    ax2.set_title(f'{split_type.capitalize()} Split: {latent_dim}D Latent Space by Subject{title_suffix}', fontsize=14)
    ax2.set_xlabel('PHATE 1', fontsize=12) 
    ax2.set_ylabel('PHATE 2', fontsize=12)
    ax2.set_zlabel('PHATE 3', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rest_suffix = "_with_rest" if include_rest else "_no_rest"
    filename = f"{split_type}_latent{latent_dim}d_space{rest_suffix}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latent_trajectories(embeddings, labels, subject_id, output_path=None):
    # Use first 3 dimensions for visualization
    embed_3d = embeddings[:, :3] if embeddings.shape[1] > 3 else embeddings
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {0: 'blue', 1: 'red', 2: 'green'}
    labels_text = {0: 'Rest', 1: 'Improv', 2: 'Scale'}
    
    for i in range(len(embed_3d) - 1):
        ax.plot(embed_3d[i:i+2, 0], 
                embed_3d[i:i+2, 1], 
                embed_3d[i:i+2, 2],
                color=colors[labels[i]], alpha=0.6, linewidth=2)
    

    ax.scatter(*embed_3d[0], color='black', s=100, marker='o', label='Start')
    ax.scatter(*embed_3d[-1], color='black', s=100, marker='s', label='End')
    

    legend_elements = [Line2D([0], [0], color=colors[i], lw=4, label=labels_text[i]) 
                      for i in range(3)]
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='black', lw=0, markersize=10, label='Start'),
        Line2D([0], [0], marker='s', color='black', lw=0, markersize=10, label='End')
    ])
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f'Latent Space Trajectory - Subject {subject_id}')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


def plot_tphate_trajectories(embeddings_dict, split_type, latent_dim, save_path=None):
    n_subjects = len(embeddings_dict)
    n_cols = 4
    n_rows = (n_subjects + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
    
    for idx, (subject, data) in enumerate(embeddings_dict.items()):
        embeddings = np.array(data['embeddings'])
        labels = np.array(data['labels'])
        n_timepoints = len(embeddings)
        
        # Compute T-PHATE for this subject
        print(f"Computing T-PHATE for subject {subject}...")
        tphate_coords = compute_tphate_embedding(embeddings, n_components=3)
        
        
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        timepoints = np.arange(n_timepoints)
        scatter = ax.scatter(tphate_coords[:, 0], 
                           tphate_coords[:, 1], 
                           tphate_coords[:, 2],
                           c=timepoints, 
                           cmap='viridis', 
                           alpha=0.8, 
                           s=10)
        
        ax.plot(tphate_coords[:, 0], 
               tphate_coords[:, 1], 
               tphate_coords[:, 2], 
               'k-', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'Subject {subject}')
        ax.set_xlabel('T-PHATE 1')
        ax.set_ylabel('T-PHATE 2')
        ax.set_zlabel('T-PHATE 3')
        
    plt.suptitle(f'T-PHATE Trajectories - {split_type} split, {latent_dim}D latent space', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


class TaskIntervalVisualizer:
    def __init__(self, latent_dim=48, split_type='subject', device=None):
        self.latent_dim = latent_dim
        self.split_type = split_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_data = {}
        self.load_models()
        
    def load_models(self):
        if self.split_type in ['subject', 'both']:
            subject_path = EXTENDED_LATENT_RESULTS_DIR / f'subject_withheld_model_latent{self.latent_dim}.pth'
            if subject_path.exists():
                self.models_data['subject'] = torch.load(subject_path, map_location=self.device)
                print(f"Loaded subject-withheld model (latent dim: {self.latent_dim})")
            else:
                print(f"Warning: Subject model not found at {subject_path}")
                
        if self.split_type in ['time', 'both']:
            time_path = EXTENDED_LATENT_RESULTS_DIR / f'time_withheld_model_latent{self.latent_dim}.pth'
            if time_path.exists():
                self.models_data['time'] = torch.load(time_path, map_location=self.device)
                print(f"Loaded time-withheld model (latent dim: {self.latent_dim})")
            else:
                print(f"Warning: Time model not found at {time_path}")
    
    def get_available_subjects(self, split=None):
        if split is None:
            split = 'subject' if 'subject' in self.models_data else 'time'
            
        if split not in self.models_data:
            return []
            
        return list(self.models_data[split]['history']['embeddings'].keys())
    
    def extract_task_interval(self, embeddings, times, interval_name):
        # Find the matching segment
        for start, end, task_name in TASK_SEGMENTS:
            if task_name == interval_name:
                # Find indices within this time range
                mask = (times >= start) & (times < end)
                interval_embeddings = embeddings[mask]
                interval_times = times[mask]
                
                # Normalize times to 0-1 for coloring
                if len(interval_times) > 0:
                    interval_times_norm = (interval_times - interval_times.min()) / \
                                         (interval_times.max() - interval_times.min())
                else:
                    interval_times_norm = interval_times
                    
                return interval_embeddings, interval_times_norm
                
        print(f"Warning: Interval '{interval_name}' not found")
        return None, None
    
    def apply_visualization_method(self, data, method='phate', **kwargs):
        if data is None or len(data) == 0:
            return None
            
        if method == 'phate':
            phate_op = phate.PHATE(
                n_components=3, 
                n_jobs=-1, 
                random_state=42,
                knn=6, 
                decay=40,  
                t=5,  
                **kwargs
            )
            return phate_op.fit_transform(data)
            
        if method == 'tphate':
            if len(data) < 10:
                
                phate_op = phate.PHATE(
                    n_components=3, 
                    n_jobs=-1, 
                    random_state=42,
                    knn=6,
                    decay=40,
                    t=5
                )
                return phate_op.fit_transform(data)
            else:
                tphate_op = tphate.TPHATE(
                    n_components=3, 
                    n_jobs=-1,
                    t=1,  
                    smooth_window=1, 
                    knn=3, 
                    decay=20, 
                    mds='metric',
                    mds_solver='sgd',
                    random_state=42,
                    **kwargs
                )
                return tphate_op.fit_transform(data)
            
        if method == 'pca':
            pca = PCA(n_components=3, random_state=42)
            return pca.fit_transform(data)
            
        if method == 'tsne':
            n = len(data)
            perp = min(30, max(5, n//4)) if n > 15 else min(n-1, 5)
            tsne = TSNE(
                n_components=3, 
                perplexity=perp,
                random_state=42,
                n_iter=1000,
                **kwargs
            )
            return tsne.fit_transform(data)
            
        raise ValueError(f"Unknown method: {method}")
    
    def visualize_intervals(
        self,
        subject_id=None,
        rest_intervals=('Rest 1', 'Rest 8'),
        improv_intervals=('Improv 1', 'Improv 4'),
        scale_intervals=('Scale 1', 'Scale 4'),
        save_path=None,
        figsize=(20, 16),
        verbose=False
    ):
        if self.split_type == 'both':
            split = 'subject' if 'subject' in self.models_data else 'time'
        else:
            split = self.split_type
        if split not in self.models_data:
            raise ValueError(f"No model data for split: {split}")

        subjects = self.get_available_subjects(split)
        if not subjects:
            raise ValueError("No subjects available")
        if subject_id is None or subject_id not in subjects:
            subject_id = subjects[0]
            print(f"Using subject: {subject_id}")

        hist = self.models_data[split]['history']
        E = np.array(hist['embeddings'][subject_id])
        T = np.array(hist['time_indices'][subject_id])

        methods = ['phate', 'tphate', 'pca', 'tsne']
        method_names = ['PHATE', 't-PHATE', 'PCA', 't-SNE']
        tasks = ['REST', 'IMPROV', 'SCALE']
        intervals_list = [rest_intervals, improv_intervals, scale_intervals]

        fig = plt.figure(figsize=figsize, constrained_layout=False)
        
        gs = fig.add_gridspec(4, 7, left=0.08, right=0.95, top=0.92, bottom=0.05,
                             width_ratios=[1]*6 + [0.05], hspace=0.3, wspace=0.25)
        
        fig.suptitle(
            f'Task Interval Visualizations — Subject {subject_id} (Latent Dim: {self.latent_dim})',
            fontsize=20, y=0.98
        )

        scatter_ref = None

        # Plot every subplot
        for i, (m, mname) in enumerate(zip(methods, method_names)):
            for j, (intervals, task) in enumerate(zip(intervals_list, tasks)):
                for k, interval in enumerate(intervals):
                    ax = fig.add_subplot(gs[i, j*2 + k], projection='3d')
                    data, times_norm = self.extract_task_interval(E, T, interval)
                    
                    if data is not None and len(data) > 3:  # Need at least 4 points
                        if verbose and i == 0 and k == 0: 
                            print(f"\n{interval}: {len(data)} timepoints")
                            print(f"  Data shape: {data.shape}")
                            print(f"  Data variance: {np.var(data, axis=0).mean():.4f}")
                        
                        if verbose and m == 'tphate' and k == 0: 
                            if len(data) < 10:
                                print(f"  Note: Using PHATE instead of t-PHATE for {interval} (only {len(data)} points)")
                        
                        coords = self.apply_visualization_method(data, m)
                        if coords is not None:
                            scat = ax.scatter(
                                coords[:,0], coords[:,1], coords[:,2],
                                c=times_norm, cmap='viridis',
                                s=40, alpha=0.8, vmin=0, vmax=1,
                                edgecolor='none'
                            )
                            ax.plot(
                                coords[:,0], coords[:,1], coords[:,2],
                                'k-', alpha=0.3, lw=0.8
                            )
                            if scatter_ref is None:
                                scatter_ref = scat

                    ax.set_title(interval, fontsize=11, pad=5)
                    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                    ax.xaxis.pane.fill = False
                    ax.yaxis.pane.fill = False
                    ax.zaxis.pane.fill = False
                    ax.grid(True, alpha=0.3)
                    ax.view_init(elev=20, azim=45)

        # Add method labels on the left
        for i, mname in enumerate(method_names):
            y_pos = 0.815 - i * 0.217
            fig.text(0.02, y_pos, mname, rotation=90, ha='center', va='center',
                    fontsize=16, fontweight='bold')

        task_x_positions = [0.23, 0.50, 0.77]
        for j, (task, x_pos) in enumerate(zip(tasks, task_x_positions)):
            fig.text(x_pos, 0.95, task, ha='center', va='bottom',
                    fontsize=18, fontweight='bold')

        if scatter_ref is not None:
            cax = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(scatter_ref, cax=cax, orientation='vertical')
            cbar.set_label('Time (normalized)', fontsize=14, labelpad=10)
            cbar.ax.tick_params(labelsize=11)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        if save_path:
            dir_path = os.path.dirname(save_path)
            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {save_path}")

        plt.show()
        plt.close(fig)