import pandas as pd
import numpy as np
import sys
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# --- Import functions from your other scripts ---
try:
    from load_data import load_data
    from feature_engineering import engineer_features
except ImportError:
    print("[ERROR] Could not import from 'load_data.py' or 'feature_engineering.py'.")
    print("        Make sure all three scripts are in the same directory.")
    sys.exit(1)

# --- 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! CRITICAL CONFIGURATION: YOU MUST CHANGE THESE !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# 1. PATH_TO_YOUR_MODEL: 
#    Point this to the folder you downloaded with 'git clone'.
#    e.g., r"F:\models\Llama-3.1-8B"
#
LOCAL_MODEL_PATH = r"F:\Llama-3.1-8B"

# 2. DATASET PATHS: 
#    Point these to your new split files.
#
TRAIN_CSV_PATH = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\new_dataset\train_dataset.csv"
TEST_CSV_PATH = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\new_dataset\test_dataset.csv"
VAL_CSV_PATH = r"F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\new_dataset\validation_dataset.csv"


# 3. ADAPTER_PATH: 
#    Where to save the small, fine-tuned model adapter.

ADAPTER_SAVE_PATH = r"F:\LLM_Models\LLama_KSS_adapter"


def format_row_as_text(row, feature_cols):
    """
    Converts a single row of tabular data (features + label) into a 
    text-based prompt string for the LLM to learn.
    """
    # System prompt explains the task to the LLM
    system_prompt = (
        "[INST] You are a medical analyst. Predict the KSS drowsiness value "
        "(0=Alert, 1=Drowsy, 2=Very Drowsy) based on the following HRV features. "
        "Respond with *only* the numerical KSS value. [/INST]\n\n"
    )
    
    # Create the feature list as a string
    feature_string = "Features:\n"
    for col in feature_cols:
         # Handle potential non-numeric values gracefully in prompt
        feature_val = row.get(col)
        if isinstance(feature_val, (int, float)):
             feature_string += f"- {col}: {feature_val:.4f}\n"
        else:
             feature_string += f"- {col}: {str(feature_val)}\n"
    
    # Get the label
    label = int(row['drowsiness'])
    
    # Combine into the final format: "PROMPT" -> "ANSWER"
    # The model learns to generate the text after "KSS Value: "
    full_text = f"{system_prompt}{feature_string}\nKSS Value: {label}"
    
    return full_text

def create_inference_prompt(row, feature_cols):
    """
    Creates the prompt string for *prediction*, stopping right before
    the answer is needed.
    """
    system_prompt = (
        "[INST] You are a medical analyst. Predict the KSS drowsiness value "
        "(0=Alert, 1=Drowsy, 2=Very Drowsy) based on the following HRV features. "
        "Respond with *only* the numerical KSS value. [/INST]\n\n"
    )
    
    feature_string = "Features:\n"
    for col in feature_cols:
        # Handle potential non-numeric values gracefully in prompt
        feature_val = row.get(col)
        if isinstance(feature_val, (int, float)):
             feature_string += f"- {col}: {feature_val:.4f}\n"
        else:
             feature_string += f"- {col}: {str(feature_val)}\n"

    # Note: We stop at "KSS Value: " and let the model complete it.
    prompt = f"{system_prompt}{feature_string}\nKSS Value: "
    return prompt


def load_and_prepare_data():
    """
    Loads pre-split data, engineers features, and formats it for the LLM.
    """
    # --- 1. Process Training Data ---
    print("\n[INFO] Loading training data...")
    train_raw_df = load_data(TRAIN_CSV_PATH)
    if train_raw_df is None:
        print("[ERROR] Failed to load training data.")
        return None, None, None

    print("[INFO] Engineering features for training data...")
    X_train, y_train = engineer_features(train_raw_df)
    if X_train is None:
        print("[ERROR] Feature engineering failed on training data.")
        return None, None, None
    
    # Get the single, definitive list of features from the training set
    FEATURE_COLUMNS = X_train.columns.tolist()
    print(f"[INFO] Found {len(FEATURE_COLUMNS)} features from training data.")
    
    print("[INFO] Serializing training data...")
    train_df = X_train.copy()
    train_df['drowsiness'] = y_train
    train_df['text'] = train_df.apply(lambda row: format_row_as_text(row, FEATURE_COLUMNS), axis=1)

    # --- 2. Process Validation Data ---
    print("\n[INFO] Loading validation data...")
    val_raw_df = load_data(VAL_CSV_PATH)
    if val_raw_df is None:
        print("[ERROR] Failed to load validation data.")
        return None, None, None

    print("[INFO] Engineering features for validation data...")
    X_val, y_val = engineer_features(val_raw_df)
    if X_val is None:
        print("[ERROR] Feature engineering failed on validation data.")
        return None, None, None

    print("[INFO] Serializing validation data...")
    val_df = X_val.copy()
    val_df['drowsiness'] = y_val
    val_df['text'] = val_df.apply(lambda row: format_row_as_text(row, FEATURE_COLUMNS), axis=1)

    # --- 3. Process Test Data ---
    print("\n[INFO] Loading test data...")
    test_raw_df = load_data(TEST_CSV_PATH)
    if test_raw_df is None:
        print("[ERROR] Failed to load test data.")
        return None, None, None

    print("[INFO] Engineering features for test data...")
    X_test, y_test = engineer_features(test_raw_df)
    if X_test is None:
        print("[ERROR] Feature engineering failed on test data.")
        return None, None, None
    
    # This dataframe is for the *final* evaluation. It doesn't need 'text'.
    test_df_for_eval = X_test.copy() 
    test_df_for_eval['drowsiness'] = y_test
    
    # This dataframe is for the 'test' split in the DatasetDict
    # The Trainer *can* use this for a final eval, but we will do it manually
    print("[INFO] Serializing test data (for DatasetDict)...")
    test_df_for_tokenizing = test_df_for_eval.copy()
    test_df_for_tokenizing['text'] = test_df_for_tokenizing.apply(lambda row: format_row_as_text(row, FEATURE_COLUMNS), axis=1)


    print(f"\n[INFO] Total samples ready for training/evaluation:")
    print(f"       - Training:   {len(train_df)}")
    print(f"       - Validation: {len(val_df)}")
    print(f"       - Test:       {len(test_df_for_eval)}")
    
    # Convert to Hugging Face Dataset objects
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df_for_tokenizing)
    })
    
    # Return the datasets, the final test df, and the feature names
    return raw_datasets, test_df_for_eval, FEATURE_COLUMNS


def fine_tune_llama(raw_datasets):
    """
    Configures and runs the QLoRA fine-tuning process.
    """
    print("\n[INFO] Starting Llama 3.1 8B fine-tuning...")
    
    # --- 1. Configure Quantization (4-bit) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- 2. Load Tokenizer and Model ---
    print(f"[INFO] Loading base model from local path: {LOCAL_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load model from local path: {e}")
        print(f"        Ensure the path '{LOCAL_MODEL_PATH}' is correct and contains")
        print("        the full Hugging Face model files (not GGUF/Ollama files).")
        return None, None
    
    # --- 3. Configure LoRA ---
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    print("[INFO] Model wrapped with PEFT (LoRA) adapters.")
    model.print_trainable_parameters()

    # --- 4. Tokenize Dataset ---
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    print("[INFO] Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    # --- 5. Configure Training ---
    training_args = TrainingArguments(
        output_dir="./llama_training_output",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        num_train_epochs=3, 
        learning_rate=2e-4,
        logging_steps=10, 
        save_strategy="epoch", 
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        load_best_model_at_end=True, # Load the best model found during training
        report_to="none", 
        fp16=True, 
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], # <-- NOW USES VALIDATION SET
        data_collator=data_collator,
    )

    # --- 6. Start Fine-Tuning ---
    print("\n[INFO] *** Starting fine-tuning loop... ***")
    trainer.train()
    print("[INFO] *** Fine-tuning complete. ***")

    # --- 7. Save the Fine-Tuned Adapter ---
    print(f"[INFO] Saving best adapter model to {ADAPTER_SAVE_PATH}")
    trainer.save_model(ADAPTER_SAVE_PATH) # Save the best model
    
    return model, tokenizer

def evaluate_model(model, tokenizer, test_df, feature_cols):
    """
    Performs inference on the held-out test set and compares to ground truth.
    """
    print("\n[INFO] Starting evaluation on *held-out test set*...")
    model.eval() # Set model to evaluation mode
    
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        # 1. Get the true label
        true_label = str(int(row['drowsiness']))
        y_true.append(true_label)
        
        # 2. Create the inference prompt (text *without* the answer)
        prompt = create_inference_prompt(row, feature_cols)
        
        # 3. Tokenize and send to GPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 4. Generate the completion
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 5. Decode *only* the new tokens
        prediction_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 6. Parse the prediction
        match = re.search(r'\d+', prediction_text)
        if match:
            pred_label = match.group(0)
        else:
            pred_label = "-1" # Model failed to output a number
            
        y_pred.append(pred_label)
        
    print("[INFO] Evaluation complete. Generating report...")
    # --- Show final metrics ---
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    print("\n--- 1. LLM Classification Report (Test Set) ---")
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=[f"KSS {l}" for l in unique_labels], zero_division=0))

    print("\n--- 2. LLM Confusion Matrix (Test Set) ---")
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, 
                         index=[f"True {l}" for l in unique_labels], 
                         columns=[f"Pred {l}" for l in unique_labels])
    print(cm_df)


# --- Main execution block ---
if __name__ == "__main__":
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("[FATAL ERROR] NVIDIA CUDA GPU not detected.")
        print("              This script requires a powerful GPU to run.")
        sys.exit(1)
        
    print(f"[INFO] Found CUDA-compatible GPU: {torch.cuda.get_device_name(0)}")

    print("==============================================")
    print("     Llama 3.1 8B Fine-Tuning Pipeline: START ")
    print("==============================================")
    
    # --- Step 1: Load, Engineer, Serialize, and Split Data ---
    print("[INFO] Loading and preparing pre-split datasets...")
    raw_datasets, test_df, feature_cols = load_and_prepare_data()
    
    if raw_datasets is not None:
        # --- Step 2: Fine-Tune the Model ---
        model, tokenizer = fine_tune_llama(raw_datasets)
        
        # --- Step 3: Evaluate the Model on the *Test* Set ---
        if model and tokenizer:
            print("\n[INFO] Evaluating model on the *held-out test set*...")
            evaluate_model(model, tokenizer, test_df, feature_cols)
        else:
            print("[FAILURE] Model training failed. Cannot evaluate.")
            
    else:
        print("[FAILURE] Data preparation failed. Cannot continue pipeline.")
    
    print("\n==============================================")
    print("     Llama 3.1 8B Fine-Tuning Pipeline: END   ")
    print("==============================================")