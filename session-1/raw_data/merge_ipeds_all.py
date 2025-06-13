import os
import pandas as pd
from functools import reduce

# Directory containing the CSV files
IPEDS_DIR = os.path.join(os.path.dirname(__file__), 'ipeds-all')

# Get all CSV file paths in the directory
def get_csv_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')]

# Read a CSV file into a DataFrame
def read_csv_file(filepath):
    return pd.read_csv(filepath)

# Merge a list of DataFrames on 'unitid'
def merge_dataframes_on_unitid(dfs):
    return reduce(lambda left, right: pd.merge(left, right, on='unitid', how='outer', suffixes=(None, '_dup')), dfs)

# Main function to merge all CSVs and save the result
def main():
    csv_files = get_csv_files(IPEDS_DIR)
    dfs = list(map(read_csv_file, csv_files))
    merged_df = merge_dataframes_on_unitid(dfs)
    merged_df.to_csv('ipeds_all_merged.csv', index=False)
    print(f"Merged DataFrame shape: {merged_df.shape}")

if __name__ == '__main__':
    main() 