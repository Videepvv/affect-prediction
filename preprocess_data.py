"""
Preprocessing script for affect prediction datasets.

This script:
1. Removes rows where 'Other' column is True
2. Removes rows with empty/missing transcripts
3. Collapses duplicate rows (same transcript + identifying columns) into one,
   combining their labels into label1, label2, label3 columns
"""

import pandas as pd
import argparse
from pathlib import Path


def preprocess_dataset(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Preprocess the affect prediction dataset.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path for the output CSV file (optional, will auto-generate if None)
    
    Returns:
        Preprocessed DataFrame
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows}")
    
    # Step 1: Remove rows where 'Other' column is True
    if 'Other' in df.columns:
        df = df[df['Other'] != True]
        rows_after_other_removal = len(df)
        print(f"Rows after removing 'Other=True': {rows_after_other_removal}")
        print(f"  Removed {initial_rows - rows_after_other_removal} rows with Other=True")
    else:
        rows_after_other_removal = initial_rows
        print("No 'Other' column found, skipping this filter")
    
    # Step 2: Remove rows with empty/missing transcripts
    df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
    rows_after_empty_removal = len(df)
    print(f"Rows after removing empty transcripts: {rows_after_empty_removal}")
    print(f"  Removed {rows_after_other_removal - rows_after_empty_removal} rows with empty transcripts")
    
    # Step 3: Identify columns that define a unique row (excluding 'labels')
    # Use only core identifying columns for deduplication, not all columns
    # This avoids issues where metadata columns like 'TimeDifference' differ
    core_id_columns = [
        'timestamp', 'videoTime', 'participantID', 'participant', 'groupID',
        'startTime', 'endTime', 'windowLength', 'transcript'
    ]
    # Only use columns that actually exist in the dataframe
    id_columns = [col for col in core_id_columns if col in df.columns]
    
    print(f"Using these columns for deduplication: {id_columns}")
    
    # Find duplicates based on core identifying columns
    duplicates_mask = df.duplicated(subset=id_columns, keep=False)
    num_duplicates = duplicates_mask.sum()
    print(f"Found {num_duplicates} rows that are part of duplicate groups")
    
    # Step 4: Collapse duplicates and combine labels
    # For numeric columns, take the first value (they should be the same for duplicates)
    # Get all other columns that we need to keep
    other_columns = [col for col in df.columns if col not in id_columns and col != 'labels']
    
    # Create aggregation dict: labels get combined, other columns take first value
    agg_dict = {'labels': lambda x: list(x.unique())}
    for col in other_columns:
        agg_dict[col] = 'first'
    
    grouped = df.groupby(id_columns, as_index=False, dropna=False).agg(agg_dict)
    
    # Create label1, label2, label3 columns
    max_labels = grouped['labels'].apply(len).max()
    print(f"Maximum number of labels per row: {max_labels}")
    
    # Create label columns
    grouped['label1'] = grouped['labels'].apply(lambda x: x[0] if len(x) > 0 else None)
    grouped['label2'] = grouped['labels'].apply(lambda x: x[1] if len(x) > 1 else None)
    grouped['label3'] = grouped['labels'].apply(lambda x: x[2] if len(x) > 2 else None)
    
    # If there are more than 3 labels, warn the user
    if max_labels > 3:
        print(f"WARNING: Some rows have more than 3 labels ({max_labels}). Only first 3 are kept.")
        rows_with_many_labels = grouped[grouped['labels'].apply(len) > 3]
        print(f"  Rows affected: {len(rows_with_many_labels)}")
    
    # Drop the temporary 'labels' column (which contains lists)
    grouped = grouped.drop(columns=['labels'])
    
    # Count rows with multiple labels
    multi_label_rows = grouped[grouped['label2'].notna()]
    print(f"Rows with 2+ labels: {len(multi_label_rows)}")
    
    three_label_rows = grouped[grouped['label3'].notna()]
    print(f"Rows with 3 labels: {len(three_label_rows)}")
    
    final_rows = len(grouped)
    print(f"Final row count: {final_rows}")
    print(f"  Collapsed {rows_after_empty_removal - final_rows} duplicate rows")
    
    # Reorder columns to put labels first after identifying columns
    # Find a good position for labels (after the original 'labels' position or near beginning)
    label_cols = ['label1', 'label2', 'label3']
    other_cols = [col for col in grouped.columns if col not in label_cols]
    
    # Insert label columns after typical identifying columns
    # Try to place them after 'labels' original position or after participantID
    insert_position = None
    for i, col in enumerate(other_cols):
        if col in ['participantID', 'participant', 'groupID']:
            insert_position = i + 1
    
    if insert_position is None:
        insert_position = min(5, len(other_cols))  # Default: near the beginning
    
    new_col_order = other_cols[:insert_position] + label_cols + other_cols[insert_position:]
    grouped = grouped[new_col_order]
    
    # Save to file
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.parent / f"{input_file.stem}_preprocessed{input_file.suffix}"
    
    grouped.to_csv(output_path, index=False)
    print(f"\nSaved preprocessed data to: {output_path}")
    
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess affect prediction datasets"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path for the output CSV file (default: adds '_preprocessed' suffix)",
        default=None
    )
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input_file, args.output)


if __name__ == "__main__":
    main()
