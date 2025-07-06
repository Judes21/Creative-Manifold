import os
from pathlib import Path

# BASE PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
RESULTS_ROOT = PROJECT_ROOT / "RESULTS"

# DATA DIRECTORIES
PREPARED_FNIRS_DIR = DATA_ROOT / "PREPARED_FNIRS_DATA"
POS_DATA_DIR = DATA_ROOT / "POS_DATA"
POS_FILES_PATTERN = str(POS_DATA_DIR / "*subject*.pos")
SCATTERING_DIR = DATA_ROOT / "SCATTERING_COEFFICIENTS"
COMBINED_FNIRS = PREPARED_FNIRS_DIR / "combined_fnirs_data.csv"
COMBINED_SCATTERING = SCATTERING_DIR / "combined_scattering_data.csv"


# RESULTS DIRECTORIES
CROSS_VALIDATION_RESULTS_DIR = RESULTS_ROOT / "cross_validation_results"
LATENT_VIZ_RESULTS_DIR = RESULTS_ROOT / "latent_space_visualization_results"
ATTENTION_RESULTS_DIR = RESULTS_ROOT / "attention_results"
TRAJECTORY_COMPARISON_DIR = RESULTS_ROOT / "trajectory_comparison_results"
FINAL_VISUALIZATION_RESULTS_DIR = RESULTS_ROOT / "final_visualization_results"
BASELINE_RESULTS_DIR = CROSS_VALIDATION_RESULTS_DIR / "baseline_results"
RNN_RESULTS_DIR = CROSS_VALIDATION_RESULTS_DIR / "rnn_results"
TOPOLOGICAL_RESULTS_DIR = CROSS_VALIDATION_RESULTS_DIR / "topological_results"
CURVATURE_RESULTS_DIR = CROSS_VALIDATION_RESULTS_DIR / "curvature_results"
WEIGHT_VIZ_RESULTS_DIR = FINAL_VISUALIZATION_RESULTS_DIR / "attention_weights"
METHOD_COMPARISON_VIZ_RESULTS_DIR = FINAL_VISUALIZATION_RESULTS_DIR / "method_comparisons"
BASELINE_INTERVAL_RESULTS_DIR = CROSS_VALIDATION_RESULTS_DIR  / "baseline_interval_results"


# PARAMETERS
THRESHOLD = 0.33  # ask Smita about this. changed from 0.6.
T_VALUE = 200
GRAPHS_PER_ROW = 3
GRAPHS_PER_COL = 3
MARKER_SIZE = 50
ALPHA = 0.7
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100
DEFAULT_DROPOUT_RATE = 0.2
EXPECTED_FNIRS_SHAPE = (7850, 48)  #48 fnirs channels
EXPECTED_SCATTERING_SHAPE = (7850, 768)  #768 scattering coefficients

# TASK STRUCTURE (based on Tachibana's experimental paradigm)
TASK_SEGMENTS = [
    (0, 250, 'Pre'),           
    (250, 650, 'Rest 1'),      
    (650, 1050, 'Sham'),      
    (1050, 1450, 'Rest 2'),    
    (1450, 1850, 'Improv 1'), 
    (1850, 2250, 'Rest 3'),
    (2250, 2650, 'Scale 1'),
    (2650, 3050, 'Rest 4'),
    (3050, 3450, 'Improv 2'),
    (3450, 3850, 'Rest 5'),
    (3850, 4250, 'Scale 2'),
    (4250, 4650, 'Rest 6'),
    (4650, 5050, 'Improv 3'),
    (5050, 5450, 'Rest 7'),
    (5450, 5850, 'Scale 3'),
    (5850, 6250, 'Rest 8'),
    (6250, 6650, 'Improv 4'),
    (6650, 7050, 'Rest 9'),
    (7050, 7450, 'Scale 4'),
    (7450, 7850, 'Rest 10')
]

# Left hemisphere 
HOLDER1_PAIRS = [
    (1, 2), (2, 3), (3, 4), (1, 5), (2, 6), (3, 7), (4, 8),
    (5, 6), (6, 7), (7, 8), (5, 9), (6, 10), (7, 11), (8, 12),
    (9, 10), (10, 11), (11, 12), (9, 13), (10, 14), (11, 15), (12, 16),
    (13, 14), (14, 15), (15, 16)
]

# Right hemisphere
HOLDER2_PAIRS = [
    (17, 18), (18, 19), (19, 20), (17, 21), (18, 22), (19, 23), (20, 24),
    (21, 22), (22, 23), (23, 24), (21, 25), (22, 26), (23, 27), (24, 28),
    (25, 26), (26, 27), (27, 28), (25, 29), (26, 30), (27, 31), (28, 32),
    (29, 30), (30, 31), (31, 32)
]

# UTILITY FUNCTIONS
def create_directories():
    directories = [
        PREPARED_FNIRS_DIR,
        POS_DATA_DIR,
        SCATTERING_DIR,
        CROSS_VALIDATION_RESULTS_DIR,
        BASELINE_RESULTS_DIR,
        LATENT_VIZ_RESULTS_DIR,
        ATTENTION_RESULTS_DIR,
        WEIGHT_VIZ_RESULTS_DIR,
        FINAL_VISUALIZATION_RESULTS_DIR,
        TRAJECTORY_COMPARISON_DIR,
        TOPOLOGICAL_RESULTS_DIR,
        RNN_RESULTS_DIR,
        CURVATURE_RESULTS_DIR,
        METHOD_COMPARISON_VIZ_RESULTS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
def get_task_label(time_point):
    for start, end, task in TASK_SEGMENTS:
        if start <= time_point < end:
            return task
    return 'Unknown'

def get_task_class(task_name):
    if 'Rest' in task_name or task_name == 'Pre':
        return 0
    elif 'Improv' in task_name:
        return 1
    elif 'Scale' in task_name:
        return 2
    else:
        return -1  # Sham or unknown