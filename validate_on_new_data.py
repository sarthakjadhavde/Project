# File: validate_on_new_data.py
# Runs the full hybrid model on new, LABELED data and reports accuracy.
import pandas as pd
import joblib
import pickle
import sys
import config
from hybrid_logic import make_final_prediction
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- !! CONFIGURE THIS !! ---
INPUT_TEST_FILE = "new_labeled_test_set.csv" 
TRUE_LABEL_COLUMN = "KSS_Label" # The column with the true answers (0, 1, 2)
# ---

def main():
    print(f"--- [Validate] START: Validating on {INPUT_TEST_FILE} ---")

    # Load Models
    try:
        model_1 = joblib.load(config.MODEL_1_PATH)
        model_2 = joblib.load(config.MODEL_2_PATH)
        with open(config.MODEL_2_COLUMNS_PATH, 'rb') as f:
            model_2_columns = pickle.load(f)
    except Exception as e:
        print(f"Error loading model files: {e}")
        sys.exit()

    # Load New Labeled Data
    try:
        df_test = pd.read_csv(INPUT_TEST_FILE)
    except Exception as e:
        print(f"Error loading test file '{INPUT_TEST_FILE}': {e}")
        sys.exit()

    # Get features and labels
    hrv_feature_cols = [col for col in df_test.columns if 'HRV_' in col]
    context_feature_cols = ['HR Min', 'HR Max', 'Time of Day']
    y_true = df_test[TRUE_LABEL_COLUMN]
    y_pred = []
    
    # Map string predictions back to integers
    label_map = {v: k for k, v in config.KSS_LABELS.items()} # {0: "ALERT", ...}
    pred_to_int_map = {v: k for k, v in label_map.items()} # {"ALERT": 0, ...}

    # --- Prediction Loop ---
    for index, row in df_test.iterrows():
        # Get inputs
        hrv_data = row[hrv_feature_cols].values
        context_data = row[context_feature_cols].to_dict()
        
        # Prep context data
        context_df = pd.DataFrame([context_data])
        context_encoded = pd.get_dummies(context_df, columns=['Time of Day'])
        context_final = context_encoded.reindex(columns=model_2_columns, fill_value=0)
        
        # Run Models
        physio_probs = model_1.predict_proba([hrv_data])[0]
        context_risk = model_2.predict(context_final)[0]
        
        # Get final decision
        final_decision_str = make_final_prediction(physio_probs, context_risk)
        y_pred.append(pred_to_int_map.get(final_decision_str, -1))

    # --- Show Final Report ---
    print("\n--- [FINAL HYBRID MODEL VALIDATION REPORT] ---")
    accuracy = accuracy_score(y_true, y_pred)
    target_names = [label_map[i] for i in sorted(y_true.unique())]
    report = classification_report(y_true, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"Total Accuracy: {accuracy * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(report)
    print("\n--- Confusion Matrix (Rows=True, Cols=Pred) ---")
    print(matrix)
    print(f"--- [Validate] END ---")

if __name__ == "__main__":
    main()