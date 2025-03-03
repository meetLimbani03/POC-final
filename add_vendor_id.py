import pandas as pd
import uuid

def add_vendor_id(csv_file):
    """
    Adds a vendor_id column with UUIDs to a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        df['vendor_id'] = [uuid.uuid4() for _ in range(len(df))]
        df.to_csv(csv_file, index=False)
        print(f"Successfully added vendor_id to {csv_file}")
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_file = "new_heinecan-sample-data_with_keywords.csv"
    add_vendor_id(csv_file)
