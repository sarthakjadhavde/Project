# File: test_model_2.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys
import config

def main():
    print(f"--- [Validation] START: Testing Model 2 Logic ---")
    
    try:
        df = pd.read_csv(config.SMALL_DATASET_LABELED_PATH)
    except FileNotFoundError:
        print(f"Error: Labeled file not found at '{config.SMALL_DATASET_LABELED_PATH}'")
        sys.exit()

    # Prepare data
    feature_columns = ['HR Min', 'HR Max', 'Time of Day']
    target_column = 'is_drowsy'
    X_unencoded = df[feature_columns]
    y = df[target_column]
    X = pd.get_dummies(X_unencoded, columns=['Time of Day'], drop_first=False)

    # Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Splitting data: {len(X_train)} train, {len(X_test)} test")

    # Train on 70%
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test on 30%
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- TEST ACCURACY: {accuracy * 100:.2f}% ---")
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["Alert (0)", "Drowsy (1)"]))
    
    if accuracy == 1.0:
        print("\nSUCCESS: 100% accuracy confirms the model perfectly learned our rule.")
    
    print(f"--- [Validation] END ---")

if __name__ == "__main__":
    main()