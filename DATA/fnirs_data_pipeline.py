import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from SCRIPTS.config import RAW_FNIRS_DIR, PREPARED_FNIRS_DIR

# Remove metadata
def clean_fnirs_file(file_path, output_path=None):
    df = pd.read_csv(file_path, header=None, skiprows=41, low_memory=False)
    df = df.drop(columns=[0])
    columns_to_remove = [25, 26, 27, 28, 29]
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if columns_to_remove:
        df = df.drop(columns=columns_to_remove, axis=1)
    if output_path:
        df.to_csv(output_path, index=False, header=False)

    return df

# Calculate oxy - deoxy hemoglobin difference
def calculate_oxy_deoxy_difference(oxy_path, deoxy_path, output_path=None):
    try:
        df_oxy = pd.read_csv(oxy_path, header=None)
        df_deoxy = pd.read_csv(deoxy_path, header=None)
    
        if df_oxy.shape[1] != df_deoxy.shape[1]:
            min_columns = min(df_oxy.shape[1], df_deoxy.shape[1])
            print(f"  Aligning columns: using first {min_columns} columns")
            df_oxy = df_oxy.iloc[:, :min_columns]
            df_deoxy = df_deoxy.iloc[:, :min_columns]

        df_diff = df_oxy - df_deoxy
        
        if output_path:
            df_diff.to_csv(output_path, index=False, header=False)
        return df_diff
        
    except Exception as e:
        print(f"Error calculating difference: {e}")
        return None

# Combine Probe 1 & 2
def combine_probes(probe1_df, probe2_df):
    combined_df = pd.concat([probe1_df, probe2_df], axis=1)
    if combined_df.shape[1] != 48:
        print(f"Warning: Expected 48 channels, got {combined_df.shape[1]}")
    
    return combined_df

# Process a single subject's fNIRS data
def process_single_subject(subject_id, input_dir=None, output_dir=None, save_intermediate=False):
    if input_dir is None:
        input_dir = RAW_FNIRS_DIR
    if output_dir is None:
        output_dir = PREPARED_FNIRS_DIR
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    print(f"\nProcessing subject: {subject_id}")
    
    subject_dir = input_dir / subject_id
    if not subject_dir.exists():
        print(f"  Subject directory not found: {subject_dir}")
        return None
    
    files = {
        'probe1_oxy': subject_dir / f"{subject_id}_HBA_Probe1_Oxy.csv",
        'probe1_deoxy': subject_dir / f"{subject_id}_HBA_Probe1_Deoxy.csv",
        'probe2_oxy': subject_dir / f"{subject_id}_HBA_Probe2_Oxy.csv",
        'probe2_deoxy': subject_dir / f"{subject_id}_HBA_Probe2_Deoxy.csv"
    }

    for name, path in files.items():
        if not path.exists():
            print(f"  Missing file: {name} - {path}")
            return None
    
    cleaned_data = {}
    for name, path in files.items():
        if save_intermediate:
            clean_path = output_dir / subject_id / f"{path.stem}_cleaned.csv"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_data[name] = clean_fnirs_file(path, clean_path)
        else:
            cleaned_data[name] = clean_fnirs_file(path)
    
    probe1_diff = calculate_oxy_deoxy_difference(
        cleaned_data['probe1_oxy'], 
        cleaned_data['probe1_deoxy']
    )
    
    probe2_diff = calculate_oxy_deoxy_difference(
        cleaned_data['probe2_oxy'], 
        cleaned_data['probe2_deoxy']
    )
    
    if probe1_diff is None or probe2_diff is None:
        print("  Error calculating differences")
        return None
    
    combined_data = combine_probes(probe1_diff, probe2_diff)
    
    output_path = output_dir / f"{subject_id}_Combined_Probe_Difference.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(output_path, index=False, header=False)
    print(f"  Saved to: {output_path}")
    print(f"  Final shape: {combined_data.shape}")
    return combined_data

 # Process all subjects in the raw data directory
def process_all_subjects(input_dir=None, output_dir=None, save_intermediate=False):
    if input_dir is None:
        input_dir = RAW_FNIRS_DIR
    if output_dir is None:
        output_dir = PREPARED_FNIRS_DIR
    input_dir = Path(input_dir)
    
    subject_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    subject_ids = [d.name for d in subject_dirs]
    print(f"Found {len(subject_ids)} subjects to process")
    
    successful = 0
    failed = []
    
    for subject_id in sorted(subject_ids):
        result = process_single_subject(subject_id, input_dir, output_dir, save_intermediate)
        if result is not None:
            successful += 1
        else:
            failed.append(subject_id)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Successful: {successful}/{len(subject_ids)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    
    return successful, failed