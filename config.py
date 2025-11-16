# File: config.py
# Central configuration file for the hybrid drowsiness project.

# --- 1. DATA FILE PATHS ---
# Input path for the large, labeled HRV dataset
BIG_DATASET_PATH = "C:\\Sarthak\\THI\\sem 3\\engineered_features_dataset.csv"

# Input path for the small, raw context dataset
SMALL_DATASET_RAW_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\heart_rate_cleaned.csv"

# Intermediate and output paths for the small dataset pipeline
SMALL_DATASET_CLEANED_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\heart_rate_context_cleaned.csv"
SMALL_DATASET_LABELED_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\heart_rate_labeled_data.csv"

# --- 2. MODEL OUTPUT PATHS ---
MODEL_1_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\physiological_model.joblib"
MODEL_2_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\contextual_model.joblib"
MODEL_2_COLUMNS_PATH = "C:\\Sarthak\\THI\\sem 3\\time context\\contextual_model_columns.pkl"

# --- 3. MODEL 2 HEURISTIC RULES ---
LOW_HR_THRESHOLD = 60
RISKY_TIMES = ['Afternoon', 'Evening', 'Night', 'Early Morning']

# --- 4. MODEL 1 LABEL MAPPING ---
# Ensure these match the integer labels in your KSS dataset
TARGET_COLUMN = "drowsiness_label"
KSS_LABELS = {
    "ALERT": 0,
    "DROWSY": 1,
    "VERY_DROWSY": 2
}