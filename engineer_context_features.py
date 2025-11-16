# File: engineer_context_features.py
import pandas as pd
import numpy as np
import sys
import config

def main():
    print(f"--- [Task 2] START: Engineering features from {config.SMALL_DATASET_CLEANED_PATH} ---")
    
    try:
        df = pd.read_csv(config.SMALL_DATASET_CLEANED_PATH)
    except FileNotFoundError:
        print(f"Error: Cleaned data file not found at '{config.SMALL_DATASET_CLEANED_PATH}'")
        print("Please run 'clean_context_data.py' first.")
        sys.exit()

    # Apply the rule from the config file
    condition_hr = df['HR Min'] < config.LOW_HR_THRESHOLD
    condition_time = df['Time of Day'].isin(config.RISKY_TIMES)

    # Apply the rule to create the target label
    df['is_drowsy'] = np.where(condition_hr & condition_time, 1, 0)

    print(f"Heuristic rule applied. 'Drowsy' (1) count: {df['is_drowsy'].sum()}")

    # Save the final labeled data
    df.to_csv(config.SMALL_DATASET_LABELED_PATH, index=False)
    
    print(f"SUCCESS: Labeled data saved to '{config.SMALL_DATASET_LABELED_PATH}'")
    print(f"--- [Task 2] END ---")

if __name__ == "__main__":
    main()