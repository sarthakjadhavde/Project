# File: train_model_1.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import sys
import config

def main():
    print(f"--- [Task 3] START: Training Physiological Model (Model 1) ---")

    try:
        df = pd.read_csv(config.BIG_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Big data file not found at '{config.BIG_DATASET_PATH}'")
        sys.exit()

    print(f"Loaded {len(df)} rows from {config.BIG_DATASET_PATH}")
    
    try:
        y = df[config.TARGET_COLUMN]
        X = df.drop(columns=[config.TARGET_COLUMN])
        X = X.filter(regex='HRV_') # Ensure we only use HRV features
    except KeyError:
        print(f"Error: Target column '{config.TARGET_COLUMN}' not found.")
        sys.exit()

    # --- Validation Step (from your report) ---
    print("\n--- Running 70/30 Validation ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    validation_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    validation_model.fit(X_train, y_train)
    y_pred = validation_model.predict(X_test)
    
    class_names = [str(c) for c in sorted(y.unique())]
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Final Training (100% of Data) ---
    print(f"\n--- Training Final Model on 100% of data ---")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    final_model.fit(X, y)

    # Save the final model
    joblib.dump(final_model, config.MODEL_1_PATH)
    
    print(f"SUCCESS: Final Model 1 saved to '{config.MODEL_1_PATH}'")
    print(f"--- [Task 3] END ---")

if __name__ == "__main__":
    main()