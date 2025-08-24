import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tphate
from SCRIPTS.config import (
    PREPARED_FNIRS_DIR, SCATTERING_DIR, THRESHOLD, TASK_SEGMENTS, 
    EXPECTED_FNIRS_SHAPE, EXPECTED_SCATTERING_SHAPE
)
from SCRIPTS.scattering import construct_adjacency_matrix, create_task_labels, generate_timepoint_feature


# Dataset for brain state classification from fNIRS data
class BrainStateDataset(Dataset):
    def __init__(self, features, labels, subject_ids=None, time_indices=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        self.time_indices = None if time_indices is None else torch.LongTensor(time_indices)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.subject_ids is None and self.time_indices is None:
            return self.features[idx], self.labels[idx]
        elif self.time_indices is None:
            return self.features[idx], self.labels[idx], self.subject_ids[idx]
        else:
            return (self.features[idx], 
                    self.labels[idx], 
                    self.subject_ids[idx],
                    self.time_indices[idx])


class TaskIntervalDataset(Dataset):
    def __init__(self, intervals_list):
        # intervals_list: List of tuples (features, label, subject_id, interval_name)
        self.intervals = intervals_list
 
    def __len__(self):
        return len(self.intervals)
    
    def __getitem__(self, idx):
        features, label, subject_id, interval_name = self.intervals[idx]
        return (
            torch.FloatTensor(features),
            torch.LongTensor([label]).squeeze(),
            subject_id,
            interval_name
        )


def clean_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def verify_data_match(subject_channel_coords, subject_data_matrices):
    xyz_subject_ids = sorted([subj_id for subj_id, _ in subject_channel_coords])
    fNIRS_subject_ids = sorted([subject_id for subject_id, _ in subject_data_matrices])
    
    if xyz_subject_ids == fNIRS_subject_ids:
        return True
    else:
        missing_in_fNIRS = set(xyz_subject_ids) - set(fNIRS_subject_ids)
        missing_in_XYZ = set(fNIRS_subject_ids) - set(xyz_subject_ids)

        if missing_in_fNIRS:
            print(f"Missing in fNIRS data: {missing_in_fNIRS}")
        if missing_in_XYZ:
            print(f"Missing in XYZ data: {missing_in_XYZ}")
        return False


def load_fnirs_data():
    file_pattern = str(PREPARED_FNIRS_DIR / "*_Combined_Probe_Difference.csv")
    subject_data_matrices = []
    
    for file_path in glob.glob(file_pattern):
        subject_id = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        
        if df.shape != EXPECTED_FNIRS_SHAPE:
            print(f"Excluding Subject {subject_id}: shape {df.shape} ≠ {EXPECTED_FNIRS_SHAPE}")
            continue
        
        cleaned_df = clean_data(df)
        subject_data_matrices.append((subject_id, cleaned_df.values))

    print(f"\nProcessed {len(subject_data_matrices)} subjects")
    if len(subject_data_matrices) == 20:
        print("All subjects successfully processed")
    else:
        print(f"Only {len(subject_data_matrices)}/20 subjects processed")
    
    return subject_data_matrices


def generate_scatter_coefficients(subject_channel_coords, subject_data_matrices):
    fnirs_dict = {subj_id: data for subj_id, data in subject_data_matrices}
    xyz_dict = {subj_id: coords for subj_id, coords in subject_channel_coords}
    common_subjects = set(fnirs_dict) & set(xyz_dict)

    for subject_id in sorted(common_subjects):
        print(f"Processing subject {subject_id}")
        fnirs_data = fnirs_dict[subject_id]
        coords = xyz_dict[subject_id]
        
        # Build adjacency matrix
        adj_matrix = construct_adjacency_matrix(coords, THRESHOLD)
        ro = fnirs_data.T  # Transpose to get channels x time
        
        # Generate scattering features
        features = generate_timepoint_feature(adj_matrix, ro)
        features = features.T  # Back to time x features
        
        output_file = SCATTERING_DIR / f"subject_{subject_id}_scattering_coeffs.csv"
        np.savetxt(output_file, features, delimiter=',')
    print(f"Generated features for {len(common_subjects)} subjects")


def combine_fnirs_data(data_dir=None, output_file=None):
    if data_dir is None:
        data_dir = PREPARED_FNIRS_DIR
    if output_file is None:
        output_file = PREPARED_FNIRS_DIR / "combined_fnirs_data.csv"
    
    file_pattern = str(data_dir / "*_Combined_Probe_Difference.csv")
    _, general_labels, timepoints = create_task_labels(TASK_SEGMENTS)
    all_data = []

    for file_path in glob.glob(file_pattern):
        subject_id = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        
        if df.shape != EXPECTED_FNIRS_SHAPE:
            print(f"Excluding Subject {subject_id}: shape {df.shape} ≠ {EXPECTED_FNIRS_SHAPE}")
            continue
            
        cleaned_df = clean_data(df)
        subject_data = pd.DataFrame(cleaned_df.values)
        subject_data['subject_id'] = subject_id
        subject_data['time_index'] = timepoints
        subject_data['task'] = general_labels
        all_data.append(subject_data)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Rename columns
    all_columns = combined_df.columns.tolist()
    node_cols = [f'node_{i}' for i in range(len(all_columns) - 3)]
    new_columns = node_cols + ['subject_id', 'time_index', 'task']
    combined_df.columns = new_columns
    
    print(f"\nDataset: {len(combined_df)} samples, {len(node_cols)} features")    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    return combined_df


def combine_scattering_data(scattering_dir=None, output_file=None):
    if scattering_dir is None:
        scattering_dir = SCATTERING_DIR
    if output_file is None:
        output_file = SCATTERING_DIR / "combined_scattering_data.csv"
    
    scattering_dir = Path(scattering_dir)
    scattering_files = list(scattering_dir.glob('subject_*_scattering_coeffs.csv'))
    _, general_labels, timepoints = create_task_labels(TASK_SEGMENTS)
    all_data = []
    
    print(f"Processing {len(scattering_files)} subjects...")
    for file_path in scattering_files:
        subject_id = file_path.stem.split('_')[1]
        print(f"Processing subject {subject_id}...")
        
        subject_coeffs = np.loadtxt(file_path, delimiter=',')
        subject_df = pd.DataFrame(subject_coeffs)
        subject_df['subject_id'] = subject_id
        subject_df['time_index'] = timepoints
        subject_df['task'] = general_labels
        all_data.append(subject_df)

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Name features consistently
    feature_cols = [f'feature_{i}' for i in range(768)]
    combined_df.columns = feature_cols + ['subject_id', 'time_index', 'task']
    
    print(f"\nDataset: {len(combined_df)} samples, {len(feature_cols)} features")
    print(f"Task distribution:\n{combined_df['task'].value_counts()}")
    combined_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    return combined_df


def compute_tphate_embedding(latent_trajectory, n_components=3, t=5, smooth_window=4):
    tphate_op = tphate.TPHATE(
        n_components=n_components,
        n_jobs=-1, 
        verbose=0, 
        t=t, 
        mds='metric', 
        mds_solver='sgd', 
        smooth_window=smooth_window
    )
    embedding = tphate_op.fit_transform(latent_trajectory)
    return embedding


#Preps data for basic MLP/baseline models (point wise)
def prepare_data(
    data_path,
    feature_prefix,
    include_metadata=False,
    test_size=0.2,
    random_state=42,
    split_type="subject" 
):    
    df = pd.read_csv(data_path)
    df['subject_id'] = df['subject_id'].astype(str)
    df['time_index'] = df['time_index'].astype(int)
    
    # Exclude Pre and Sham
    df = df[df['task'].isin(['Rest', 'Improv', 'Scale'])].reset_index(drop=True)
    df['label'] = df['task'].map({'Rest': 0, 'Improv': 1, 'Scale': 2})

    subject_ids = df['subject_id'].values
    time_indices = df['time_index'].values
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feature_cols].values
    y = df['label'].values

    if split_type == "subject":
        subs = np.unique(subject_ids)
        np.random.seed(random_state)
        n_test = max(1, int(len(subs) * test_size))
        test_subs = np.random.choice(subs, n_test, replace=False)
        
        mask_test = np.isin(subject_ids, test_subs)
        mask_train = ~mask_test
        
        # Check data leakage
        train_subs = np.unique(subject_ids[mask_train])
        actual_test_subs = np.unique(subject_ids[mask_test])
        assert len(set(train_subs) & set(actual_test_subs)) == 0, "Data leakage detected!"

    elif split_type == "time":
        mask_train = np.zeros(len(df), dtype=bool)
        mask_test = np.zeros(len(df), dtype=bool)
        for s in np.unique(subject_ids):
            idxs = np.where(subject_ids == s)[0]
            sorted_idxs = idxs[np.argsort(time_indices[idxs])]
            cutoff = int(len(sorted_idxs) * (1 - test_size))
            mask_train[sorted_idxs[:cutoff]] = True
            mask_test[sorted_idxs[cutoff:]] = True

    else:
        raise ValueError("split_type must be 'subject' or 'time'")

    # Split data
    X_train, X_test = X[mask_train], X[mask_test]
    y_train, y_test = y[mask_train], y[mask_test]
    subj_train, subj_test = subject_ids[mask_train], subject_ids[mask_test]
    time_train, time_test = time_indices[mask_train], time_indices[mask_test]

    # Handle missing data and normalize
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Weighted sampler for class balance
    class_wts = 1.0 / np.bincount(y_train)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=class_wts[y_train],
        num_samples=len(y_train),
        replacement=True
    )
            
    # Create datasets & loaders
    if include_metadata:
        train_ds = BrainStateDataset(X_train, y_train, subj_train, time_train)
        test_ds = BrainStateDataset(X_test, y_test, subj_test, time_test)
    else:
        train_ds = BrainStateDataset(X_train, y_train)
        test_ds = BrainStateDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    print(f"\nSplit: {split_type} — train {len(train_ds)}, test {len(test_ds)}")
    return train_loader, test_loader


def prepare_interval_data(
    scattering_data_path,
    split_type='subject',
    test_size=0.2,
    batch_size=32,
    random_state=42
):
    df = pd.read_csv(scattering_data_path, low_memory=False)
    valid_subjects = []
    for subject in df['subject_id'].unique():
        subject_df = df[df['subject_id'] == subject]
        if len(subject_df) == 7850:
            valid_subjects.append(subject)
    
    print(f"Found {len(valid_subjects)} valid subjects")
    df = df[df['subject_id'].isin(valid_subjects)]
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    all_intervals = []
    
    # Extract 400-timepoint intervals for each task segment
    for subject in valid_subjects:
        subject_df = df[df['subject_id'] == subject].sort_values('time_index')
        
        for start_time, end_time, interval_name in TASK_SEGMENTS:
            # Skip Pre and Sham
            if interval_name in ['Pre', 'Sham']:
                continue
                
            interval_mask = (subject_df['time_index'] >= start_time) & \
                           (subject_df['time_index'] < end_time)
            interval_data = subject_df[interval_mask]
            
            if len(interval_data) != 400:
                print(f"Warning: {interval_name} has {len(interval_data)} points for subject {subject}")
                continue
            
            features = interval_data[feature_cols].values
            
            if 'Rest' in interval_name:
                label = 0
            elif 'Improv' in interval_name:
                label = 1
            elif 'Scale' in interval_name:
                label = 2
            else:
                continue
            
            all_intervals.append((features, label, str(subject), interval_name))
    print(f"\nTotal intervals extracted: {len(all_intervals)} (expected: {len(valid_subjects) * 18})")
    
    all_features = np.array([x[0] for x in all_intervals])
    all_labels = np.array([x[1] for x in all_intervals])
    all_subjects = [x[2] for x in all_intervals]
    all_interval_names = [x[3] for x in all_intervals]
    
    # Create train/test split
    if split_type == 'subject':
        unique_subjects = list(set(all_subjects))
        np.random.seed(random_state)
        np.random.shuffle(unique_subjects)
        n_test = max(1, int(len(unique_subjects) * test_size))
        test_subjects = unique_subjects[:n_test]
        train_subjects = unique_subjects[n_test:]

        print(f"\nSubject split:")
        print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"  Test subjects ({len(test_subjects)}): {test_subjects}")
        
        assert len(set(train_subjects) & set(test_subjects)) == 0, "Subject overlap detected!"
        
        train_mask = np.array([subj in train_subjects for subj in all_subjects])
        test_mask = ~train_mask
        
    elif split_type == 'time':
        train_indices = []
        test_indices = []

        for subject in set(all_subjects):
            subject_indices = [i for i, s in enumerate(all_subjects) if s == subject]
            n_subject = len(subject_indices)
            n_train = int(n_subject * (1 - test_size))
            
            train_indices.extend(subject_indices[:n_train])
            test_indices.extend(subject_indices[n_train:])
        
        train_mask = np.zeros(len(all_intervals), dtype=bool)
        train_mask[train_indices] = True
        test_mask = ~train_mask
    
    else:
        raise ValueError("split_type must be 'subject' or 'time'")
    
    # Split intervals & normalize
    train_intervals = [all_intervals[i] for i in range(len(all_intervals)) if train_mask[i]]
    test_intervals = [all_intervals[i] for i in range(len(all_intervals)) if test_mask[i]]
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_intervals)} intervals")
    print(f"  Test: {len(test_intervals)} intervals")
    
    train_features = np.array([x[0] for x in train_intervals])
    test_features = np.array([x[0] for x in test_intervals])
    
    n_train, n_time, n_feat = train_features.shape
    n_test = test_features.shape[0]
    
    train_flat = train_features.reshape(-1, n_feat)
    test_flat = test_features.reshape(-1, n_feat)
    
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_flat)
    test_flat = scaler.transform(test_flat)  # Apply training scaler to test
    
    train_features = train_flat.reshape(n_train, n_time, n_feat)
    test_features = test_flat.reshape(n_test, n_time, n_feat)
    
    train_intervals_norm = [
        (train_features[i], train_intervals[i][1], train_intervals[i][2], train_intervals[i][3])
        for i in range(len(train_intervals))
    ]
    test_intervals_norm = [
        (test_features[i], test_intervals[i][1], test_intervals[i][2], test_intervals[i][3])
        for i in range(len(test_intervals))
    ]
    
    # Create datasets & loaders
    train_dataset = TaskIntervalDataset(train_intervals_norm)
    test_dataset = TaskIntervalDataset(test_intervals_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, {
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
        'train_subjects': train_subjects if split_type == 'subject' else None,
        'test_subjects': test_subjects if split_type == 'subject' else None
    }


def verify_data_loading(scattering_data_path):
    print("\n=== VERIFYING DATA EXTRACTION ===")
    
    # Test both split types
    for split_type in ['subject', 'time']:
        print(f"\n--- Testing {split_type} split ---")
        train_loader, test_loader, info = prepare_interval_data(
            scattering_data_path,
            split_type=split_type,
            batch_size=4
        )
        
        # Verify no data leakage for subject split
        if split_type == 'subject' and info['train_subjects'] and info['test_subjects']:
            train_subjects_in_loader = set()
            test_subjects_in_loader = set()
            
            for _, _, subjects, _ in train_loader:
                train_subjects_in_loader.update(subjects)
            for _, _, subjects, _ in test_loader:
                test_subjects_in_loader.update(subjects)
                
            overlap = train_subjects_in_loader & test_subjects_in_loader
            if overlap:
                print(f"WARNING: Data leakage detected! Overlapping subjects: {overlap}")
            else:
                print("✓ No data leakage - train and test subjects are completely separate")
    
    return True