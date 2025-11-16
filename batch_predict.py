# File: batch_predict.py
# Runs the full hybrid model on new, UNLABELED data.
import pandas as pd
import joblib
import pickle
import sys
import config
from hybrid_logic import make_final_prediction

# --- !! CONFIGURE THIS !! ---
# This is the file you want to make predictions on.
# It MUST have all HRV_ features AND the context features.
INPUT_TEST_FILE = "new_data_for_prediction.csv" 
# ---

OUTPUT_PREDICTIONS_FILE = "hybrid_predictions.csv" 

def main():
    print(f"--- [Predict] START: Running batch prediction on {INPUT_TEST_FILE} ---")

    # Load Models
    try:
        model_1 = joblib.load(config.MODEL_1_PATH)
        model_2 = joblib.load(config.MODEL_2_PATH)
        with open(config.MODEL_2_COLUMNS_PATH, 'rb') as f:
            model_2_columns = pickle.load(f)
    except Exception as e:
        print(f"Error loading model files: {e}")
        sys.exit()

    # Load New Data
    try:
        df_test = pd.read_csv(INPUT_TEST_FILE)
    except Exception as e:
        print(f"Error loading test file '{INPUT_TEST_FILE}': {e}")
        sys.exit()

    # Get feature lists
    hrv_feature_cols = [col for col in df_test.columns if 'HRV_' in col]
    context_feature_cols = ['HR Min', 'HR Max', 'Time of Day']

    # --- Prediction Loop ---
    final_predictions = []
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
        final_decision = make_final_prediction(physio_probs, context_risk)
        final_predictions.append(final_decision)

    # Save Results
    results_df = df_test.copy()
    results_df['FINAL_PREDICTION'] = final_predictions
    results_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    
    print(f"SUCCESS: Predictions saved to '{OUTPUT_PREDICTIONS_FILE}'")
    print(f"--- [Predict] END ---")

if __name__ == "__main__":
    main()