import os
import pandas as pd

def test_merged_ipeds_all():
    merged_file = 'ipeds_all_merged.csv'
    assert os.path.exists(merged_file), f"{merged_file} does not exist. Run the merge script first."
    df = pd.read_csv(merged_file)
    # Check that 'unitid' is present
    assert 'unitid' in df.columns, "'unitid' column missing in merged file."
    # Check that the number of rows is at least as large as the largest input file
    input_files = [os.path.join('ipeds-all', f) for f in os.listdir('ipeds-all') if f.endswith('.csv')]
    max_rows = max(pd.read_csv(f).shape[0] for f in input_files)
    assert df.shape[0] >= max_rows, f"Merged file has fewer rows ({df.shape[0]}) than the largest input file ({max_rows})."
    print(f"Test passed: merged file has {df.shape[0]} rows and columns: {list(df.columns)}")

if __name__ == '__main__':
    test_merged_ipeds_all() 