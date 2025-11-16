# 🧠 Hybrid Drowsiness Detection System

This project is a machine learning pipeline designed to detect driver drowsiness. It addresses a key challenge in sensor-based detection: **ambiguity**.

While physiological data (like Heart Rate Variability) is a strong predictor, it often struggles to distinguish between a "calm, alert" state and a "drowsy, low-energy" state.

This project solves that problem by implementing a **two-stage hybrid model**.

## 1. The Hybrid Model Strategy

Our system combines two different "expert" models to make a final, more accurate decision:

### 🤖 Model 1: The Physiological Expert
* **What it is:** A complex Random Forest model.
* **Trained on:** A large dataset with 90+ Heart Rate Variability (HRV) features and their corresponding KSS (drowsiness) labels.
* **Job:** To find the deep, complex physiological patterns of drowsiness.
* **Output:** A set of probabilities (e.g., `Alert: 40%, Drowsy: 50%, Very Drowsy: 10%`).
* **Weakness:** It can be "ambiguous" (like the 50/40 split above), just as identified in our initial research.

### 🧠 Model 2: The Contextual Expert
* **What it is:** A simple, rule-based Random Forest model.
* **Trained on:** A smaller dataset with context features (`HR Min`, `Time of Day`).
* **Job:** To apply a simple, heuristic rule that we defined:
    > `IF (HR Min < 60) AND (Time of Day is 'Risky') THEN 1 (Risky)`
    > `ELSE 0 (Not Risky)`
* **Output:** A single "risk" flag (`0` or `1`).

### ⚖️ The Hybrid Logic (The "Brain")
The final prediction is made by `hybrid_logic.py`, which acts as the "brain":

> **IF** Model 1 is "ambiguous" (e.g., `Drowsy: 50%, Alert: 40%`)
> **AND** Model 2 flags a `Risk: 1`...
> **THEN** The system resolves the ambiguity and confidently predicts **"Drowsy"**.

This allows the system to use real-world context to solve the exact problem our sensor model struggles with.

## 2. 🗂️ File Structure & Workflow

This project is broken into 9 individual files, representing a clear data processing "assembly line."

### ⚙️ Core Modules
* **`config.py`**: A central file to hold all file paths, model rules (like `LOW_HR_THRESHOLD`), and settings.
* **`hybrid_logic.py`**: The non-runnable "brain" file. It contains the `make_final_prediction` function that all prediction scripts import.

### 🏭 Training Pipeline (The "Factory")
These scripts are run *once* to build the models.
1.  **`clean_context_data.py`**: **(Task 1)** Loads the raw `heart_rate_cleaned.csv`, cleans it, and saves `context_cleaned.csv`.
2.  **`engineer_context_features.py`**: **(Task 2)** Loads `context_cleaned.csv`, applies the heuristic rule from `config.py`, and saves the final *labeled* file, `context_labeled.csv`.
3.  **`train_model_1.py`**: **(Task 3)** Trains the complex Model 1 on your `big_hrv_kss_dataset.csv`. It validates the model and saves the final `physiological_model.joblib`.
4.  **`train_model_2.py`**: **(Task 4)** Trains the simple Model 2 on `context_labeled.csv`. It saves both `contextual_model.joblib` and the crucial `contextual_model_columns.pkl`.

### 🔬 Validation & Production (The "Application")
These scripts use the models you just built.
* **`test_model_2.py`**: A validation script. You can run this after Task 4 to *prove* that Model 2 successfully learned your heuristic rule (it should give 100% accuracy).
* **`batch_predict.py`**: A production script. It loads *new, unlabeled data*, runs the full hybrid model, and saves a CSV with the final predictions.
* **`validate_on_new_data.py`**: A production script for testing. It loads *new, labeled data*, runs the hybrid model, and prints a full accuracy report and confusion matrix.

## 3. 🛠️ Installation

You will need the following Python libraries. You can install them Via pip:

```bash
pip install pandas numpy scikit-learn