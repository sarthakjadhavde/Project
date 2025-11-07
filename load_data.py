import pandas as pd
import sys # Import sys to check Python versionpthy

def load_data(file_path):
    print(f"[INFO] Attempting to load data from: {file_path}")
    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(file_path)
        print("[INFO] Data loaded successfully.")

        # --- Basic Data Verification ---
        print("\n--- Data Info ---")
        df.info() # Prints column names, non-null counts, and data types

        print("\n--- First 5 Rows ---")
        print(df.head()) # Shows the first few rows of your data

        print("\n--- Basic Statistics (for numeric columns) ---")
        print(df.describe()) # Shows count, mean, std dev, min, max, etc.

        return df

    except FileNotFoundError:
        print(f"[ERROR] File not found at the specified path.")
        print("Please ensure the path is correct and the file exists.")
        return None
    except pd.errors.EmptyDataError:
        print(f"[ERROR] The CSV file appears to be empty.")
        return None
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while loading the data: {e}")
        return None

# --- Main execution block ---
if __name__ == "__main__":
    # Print Python and Pandas versions for debugging
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")

    # --- Use the specific file path you provided ---
    # The 'r' before the string handles the backslashes in the Windows path
    csv_file_path = r'F:\Users\Aryan\Documents\IAE-M\THI_Notes_Files\Group Project\Source Code\Data_Files\new_dataset\train_dataset.csv'

    # Call the function to load the data
    data_frame = load_data(csv_file_path)

    if data_frame is not None:
        print(f"\n[SUCCESS] Loaded DataFrame with {len(data_frame)} rows and {len(data_frame.columns)} columns.")
    else:
        print("\n[FAILURE] Data loading failed. Please check the errors above.")
