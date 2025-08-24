import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
from SCRIPTS.config import FINAL_VISUALIZATION_RESULTS_DIR, ATTENTION_RESULTS_DIR

def plot_attention_weights(weights, split_type="", output_dir=None):
    if output_dir is None:
        output_dir = FINAL_VISUALIZATION_RESULTS_DIR / "attention_weights"
        
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(weights)), weights)
    plt.title(f'Node Attention Weights ({split_type.capitalize()} Split)')
    plt.xlabel('Node Index')
    plt.ylabel('Weight')
    plt.xticks(range(0, len(weights), 5))
    plt.grid(True, alpha=0.3)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{split_type}_attention_weights.png', dpi=300)
    plt.show()

def compare_attention_with_pca(attention_weights, pca_loadings, split_name, output_dir=None):
    if output_dir is None:
        output_dir = FINAL_VISUALIZATION_RESULTS_DIR / "attention_weights"
    
    # Looking at first principal component
    first_pc_loadings = pca_loadings[0]
    abs_loadings = np.abs(first_pc_loadings)
    normalized_loadings = abs_loadings / np.sum(abs_loadings)
    
    # Calculate correlation
    correlation = np.corrcoef(attention_weights, normalized_loadings)[0, 1]
    print(f"\nCorrelation between {split_name} and first PC loadings: {correlation:.4f}")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(attention_weights)), attention_weights)
    plt.title(f'{split_name.capitalize()} Attention Weights', fontsize=14)
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.xticks(range(0, len(attention_weights), 5))
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(normalized_loadings)), normalized_loadings)
    plt.title(f'First Principal Component Loadings', fontsize=14)
    plt.xlabel('Node Index', fontsize=12)
    plt.ylabel('Loading', fontsize=12)
    plt.xticks(range(0, len(normalized_loadings), 5))
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{split_name}_vs_first_pc_comparison.png', dpi=300)
    plt.show()
    
    # Find top nodes
    attention_top = np.argsort(attention_weights)[-5:][::-1]
    pc_top = np.argsort(normalized_loadings)[-5:][::-1]
    common_nodes = set(attention_top).intersection(set(pc_top))
    
    print(f"\nCommon top channels: {sorted(common_nodes)}")
    return correlation, attention_top, pc_top

def plot_attention_weights_with_regions(weights, brain_regions_csv_path, split_type="", output_dir=None):
    if output_dir is None:
        output_dir = FINAL_VISUALIZATION_RESULTS_DIR / "attention_weights"
    brain_regions_df = pd.read_csv(brain_regions_csv_path)
    
    # Region to color mapping
    unique_regions = brain_regions_df['Primary_Region'].unique()
    n_regions = len(unique_regions)
    if n_regions <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, n_regions))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_regions))
    
    region_to_color = {region: colors[i] for i, region in enumerate(unique_regions)}
    fig, ax = plt.subplots(figsize=(24, 14))

    x_positions = []
    bar_colors = []
    bar_labels = []
    bar_weights = []
    
    # Group by hemisphere
    left_data = brain_regions_df[brain_regions_df['Side'] == 'Left'].copy()
    right_data = brain_regions_df[brain_regions_df['Side'] == 'Right'].copy()
    
    left_data = left_data.sort_values('Primary_Region')
    right_data = right_data.sort_values('Primary_Region')
    combined_data = pd.concat([left_data, right_data])
    
    x_pos = 0
    hemisphere_gap = 2
    
    for idx, row in combined_data.iterrows():
        if x_pos > 0 and x_pos == len(left_data):
            x_pos += hemisphere_gap
        
        channel = row['Channel']
        region = row['Primary_Region']
        side = row['Side']
        
        
        x_positions.append(x_pos)
        bar_colors.append(region_to_color[region])
        bar_weights.append(weights[channel])
        
        ba_number = region.split(' - ')[0]
        bar_labels.append(f"Ch{channel}\nBA{ba_number}")
        
        x_pos += 1
    
    # Create chart
    bars = ax.bar(x_positions, bar_weights, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Channel / Brodmann Area', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('Attention Weight', fontsize=16, fontweight='bold')
    ax.set_title(f'Node Attention Weights by Brain Region ({split_type.capitalize()} Split)', 
                 fontsize=20, fontweight='bold', pad=25)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8, va='top')
    ax.tick_params(axis='x', pad=15, length=10)  
    
    current_region = None
    region_boundaries = []
    for i, row in enumerate(combined_data.itertuples()):
        if row.Primary_Region != current_region:
            if current_region is not None:
                region_boundaries.append(x_positions[i] - 0.5)
            current_region = row.Primary_Region
    
    for boundary in region_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Hemisphere labels
    left_center = (x_positions[len(left_data)-1] + x_positions[0]) / 2
    right_center = (x_positions[-1] + x_positions[len(left_data)+hemisphere_gap]) / 2
    
    ax.text(left_center, ax.get_ylim()[1] * 0.95, 'LEFT HEMISPHERE', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    
    ax.text(right_center, ax.get_ylim()[1] * 0.95, 'RIGHT HEMISPHERE', 
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
    
    # Legend
    legend_elements = []
    for region in unique_regions:
        ba_number = region.split(' - ')[0]
        region_name = region.split(' - ')[1]
        legend_label = f"BA{ba_number} ({region_name})"
        legend_elements.append(mpatches.Patch(color=region_to_color[region], label=legend_label))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=12, title='Brodmann Areas', title_fontsize=14, frameon=True)
    
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(bar_weights) * 1.1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    #Save figure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{split_type}_attention_weights_brain_regions.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print(f"\n=== Attention Weight Summary by Brain Region ({split_type} split) ===")
    region_weights = {}
    
    for idx, row in brain_regions_df.iterrows():
        region = row['Primary_Region']
        if region not in region_weights:
            region_weights[region] = {'left': [], 'right': []}
        region_weights[region][row['Side'].lower()].append(weights[row['Channel']])
    
    for region, sides in region_weights.items():
        ba_number = region.split(' - ')[0]
        region_name = region.split(' - ')[1]
        print(f"\nBA{ba_number} ({region_name}):")
        if sides['left']:
            print(f"  Left:  mean={np.mean(sides['left']):.4f}, max={np.max(sides['left']):.4f}, n={len(sides['left'])}")
        if sides['right']:
            print(f"  Right: mean={np.mean(sides['right']):.4f}, max={np.max(sides['right']):.4f}, n={len(sides['right'])}")


def compare_attention_splits(
    model_name="NeuroMuse",
    model_dir="combined", 
    latent_dim="8D",
    split_types="both",  # "subject", "time", or "both"
    trial=0,
    output_dir=None,
    brain_regions_csv_path=None
):
    
    if output_dir is None:
        output_dir = FINAL_VISUALIZATION_RESULTS_DIR / "attention_weights"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if brain_regions_csv_path is None:
        brain_regions_csv_path = Path("../DATA/POS_DATA/xyz_primary_regions_snapped.csv")
    
    # Load brain regions
    try:
        brain_regions_df = pd.read_csv(brain_regions_csv_path)
    except FileNotFoundError:
        print(f"Warning: Brain regions file not found at {brain_regions_csv_path}")
        brain_regions_df = None
    
    if brain_regions_df is not None:
        unique_regions = brain_regions_df['Primary_Region'].unique()
        n_regions = len(unique_regions)
        if n_regions <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_regions))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(n_regions, 20)))
        region_colormap = {region: colors[i] for i, region in enumerate(unique_regions)}
    else:
        region_colormap = None
    
    splits_to_load = []
    if split_types == "both":
        splits_to_load = ["subject", "time"]
    elif split_types in ["subject", "time"]:
        splits_to_load = [split_types]
    else:
        raise ValueError(f"Invalid split_types: {split_types}. Must be 'subject', 'time', or 'both'")
    
    attention_data = {}
    for split in splits_to_load:
        filename = f"attention_{split}_{model_dir}_{latent_dim}_trial{trial}.pth"
        filepath = ATTENTION_RESULTS_DIR / model_dir / filename
        
        try:
            data = torch.load(filepath, map_location='cpu')
            weights = data['attention_weights']
            attention_data[split] = weights
            print(f"Loaded {split} weights: range=[{np.min(weights):.3f}, {np.max(weights):.3f}]")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            attention_data[split] = None
    
    if split_types == "both":
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'{model_name} Attention Weights - {latent_dim} (Trial {trial})', 
                     fontsize=16, fontweight='bold')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
        fig.suptitle(f'{model_name} Attention Weights - {latent_dim} {split_types.capitalize()} Split (Trial {trial})', 
                     fontsize=16, fontweight='bold')
    
    for idx, (split, ax) in enumerate(zip(splits_to_load, axes)):
        weights = attention_data.get(split)
        
        if weights is None or brain_regions_df is None:
            ax.text(0.5, 0.5, f'{split.capitalize()} Split\nNo Data Available', 
                    transform=ax.transAxes, ha='center', va='center', 
                    fontsize=14, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        
        left_data = brain_regions_df[brain_regions_df['Side'] == 'Left'].copy()
        right_data = brain_regions_df[brain_regions_df['Side'] == 'Right'].copy()
        
        left_data = left_data.sort_values('Primary_Region')
        right_data = right_data.sort_values('Primary_Region')
        combined_data = pd.concat([left_data, right_data])
        
        x_positions = []
        bar_colors = []
        bar_labels = []
        bar_weights = []
        
        x_pos = 0
        hemisphere_gap = 1.5
        
        for _, row in combined_data.iterrows():
            if x_pos > 0 and x_pos == len(left_data):
                x_pos += hemisphere_gap
            
            channel = row['Channel']
            region = row['Primary_Region']
            
            x_positions.append(x_pos)
            bar_colors.append(region_colormap[region])
            bar_weights.append(weights[channel])
            bar_labels.append(f"{channel}")
            
            x_pos += 1
        
        bars = ax.bar(x_positions, bar_weights, color=bar_colors, 
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Labels and formatting
        ax.set_title(f'{split.capitalize()}-Withheld Split', fontsize=12, fontweight='bold')
        ax.set_xlabel('Channel', fontsize=10)
        ax.set_ylabel('Attention Weight', fontsize=10)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=7)
        
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        weight_range = max_weight - min_weight
        
        y_min = max(0, min_weight - 0.1 * weight_range)
        y_max = max_weight + 0.15 * weight_range
        ax.set_ylim(y_min, y_max)
        
        # Hemisphere labels
        if len(left_data) > 0 and len(x_positions) > 0:
            left_center = (x_positions[len(left_data)-1] + x_positions[0]) / 2
            ax.text(left_center, y_max * 0.95, 'LEFT', 
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        
        if len(right_data) > 0 and len(x_positions) > len(left_data):
            right_start_idx = len(left_data) + (1 if len(left_data) > 0 else 0)
            if right_start_idx < len(x_positions):
                right_center = (x_positions[-1] + x_positions[right_start_idx]) / 2
                ax.text(right_center, y_max * 0.95, 'RIGHT', 
                        ha='center', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5))
        
        ax.grid(False)
        ax.set_axisbelow(True)
        
        # Print top 5 nodes
        print(f"\nTop 5 nodes for {split}-withheld split:")
        top_5_idx = np.argsort(weights)[-10:][::-1] ## CHANGE BACK TO FIVE IF YOU WANT
        for rank, idx in enumerate(top_5_idx, 1):
            channel_info = brain_regions_df[brain_regions_df['Channel'] == idx]
            if not channel_info.empty:
                region = channel_info.iloc[0]['Primary_Region']
                side = channel_info.iloc[0]['Side']
                ba = region.split(' - ')[0] if ' - ' in region else region[:4]
                print(f"  {rank}. Channel {idx} ({side[0]}-{ba}): {weights[idx]:.4f}")
            else:
                print(f"  {rank}. Channel {idx}: {weights[idx]:.4f}")
        
        # Calculate and print mean attention per region
        print(f"\nMean attention weights by region for {split}-withheld split:")
        region_weights = {}
        for _, row in brain_regions_df.iterrows():
            region = row['Primary_Region']
            if region not in region_weights:
                region_weights[region] = []
            region_weights[region].append(weights[row['Channel']])
        
        region_stats = []
        for region, region_weight_list in region_weights.items():
            ba = region.split(' - ')[0] if ' - ' in region else region[:10]
            region_stats.append({
                'region': ba,
                'mean': np.mean(region_weight_list),
                'std': np.std(region_weight_list),
                'n': len(region_weight_list)
            })
        
        region_stats.sort(key=lambda x: x['mean'], reverse=True)
        for stats in region_stats[:5]:  # Show top 5 regions
            print(f"  {stats['region']}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['n']}")
    
    # Add legend
    if brain_regions_df is not None and region_colormap is not None:
        legend_elements = []
        for region in sorted(brain_regions_df['Primary_Region'].unique()):
            ba_number = region.split(' - ')[0] if ' - ' in region else region[:10]
            region_name = region.split(' - ')[1] if ' - ' in region else region[10:]
            legend_label = f"BA{ba_number}: {region_name[:20]}"
            legend_elements.append(mpatches.Patch(color=region_colormap[region], 
                                                 label=legend_label))
        
        if split_types == "both":
            fig.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1.02, 0.5), fontsize=10, 
                      title='Brodmann Areas', title_fontsize=12, 
                      frameon=True, ncol=1)
        else:
            ax.legend(handles=legend_elements, loc='best', 
                     fontsize=9, title='Brodmann Areas', 
                     title_fontsize=11, frameon=True, ncol=1)
    
    plt.tight_layout()
    
    filename = f'{model_name.lower().replace(" ", "_")}_{latent_dim}_{split_types}_attention.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFigure saved to: {output_dir / filename}")
    
    return fig