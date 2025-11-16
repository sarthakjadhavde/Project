# File: clean_context_data.py
import pandas as pd
import sys
import config

def main():
    print(f"--- [Task 1] START: Cleaning {config.SMALL_DATASET_RAW_PATH} ---")
    
    try:
        # Load the original CSV, skipping row 1 (the junk row)
        df = pd.read_csv(config.SMALL_DATASET_RAW_PATH, skiprows=[1])
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{config.SMALL_DATASET_RAW_PATH}'")
        sys.exit()

    # Strip any extra whitespace from column names
    df.columns = df.columns.str.strip()

    # Convert HR columns to numbers
    df['HR Min'] = pd.to_numeric(df['HR Min'], errors='coerce')
    df['HR Max'] = pd.to_numeric(df['HR Max'], errors='coerce')

    # Drop any rows that had bad data in HR Min or Max
    df.dropna(subset=['HR Min', 'HR Max'], inplace=True)

    # Clean up the 'Time of Day' column
    df['Time of Day'] = df['Time of Day'].str.strip()
    df['Time of Day'] = df['Time of Day'].fillna('Unknown')
    
    # Save the cleaned data
    df.to_csv(config.SMALL_DATASET_CLEANED_PATH, index=False)
    
    print(f"SUCCESS: Cleaned data saved to '{config.SMALL_DATASET_CLEANED_PATH}'")
    print(f"--- [Task 1] END ---")

if __name__ == "__main__":
    main()