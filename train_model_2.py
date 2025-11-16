# File: train_model_2.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import sys
import config

def main():
    print(f"--- [Task 4] START: Training Contextual Model (Model 2) ---")
    
    try:
        df = pd.read_csv(config.SMALL_DATASET_LABELED_PATH)
    except FileNotFoundError:
        print(f"Error: Labeled data file not found at '{config.SMALL_DATASET_LABELED_PATH}'")
        print("Please run 'engineer_context_features.py' first.")
        sys.exit()

    # Prepare data for model
    feature_columns = ['HR Min', 'HR Max', 'Time of Day']
    target_column = 'is_drowsy'
    X_unencoded = df[feature_columns]
    y = df[target_column]

    # One-Hot Encode the features
    X = pd.get_dummies(X_unencoded, columns=['Time of Day'], drop_first=False)

    # Train Model (100% of Data)
    print("Training final model on 100% of contextual data...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X, y)

    # Save Model
    joblib.dump(final_model, config.MODEL_2_PATH)
    print(f"SUCCESS: Final Model 2 saved to '{config.MODEL_2_PATH}'")

    # Save the column list (critical for production)
    with open(config.MODEL_2_COLUMNS_PATH, 'wb') as f:
        pickle.dump(X.columns.to_list(), f)
    print(f"SUCCESS: Model columns saved to '{config.MODEL_2_COLUMNS_PATH}'")
    print(f"--- [Task 4] END ---")

if __name__ == "__main__":
    main()