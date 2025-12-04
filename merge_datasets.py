"""
Merge Kickstarter CSV files from October and November datasets
This script only needs to be run once to create the merged datasets
"""

import pandas as pd
import glob
import os
from datetime import datetime

print("Starting merge process...")
start_time = datetime.now()

# Merge October CSVs
print("\n=== Merging October CSVs ===")
oct_path = 'data/Kickstarter_2025-10-13T07_42_31_884Z/'
oct_files = sorted(glob.glob(os.path.join(oct_path, 'Kickstarter*.csv')))

print(f"Found {len(oct_files)} October CSV files")

oct_dfs = []
for i, file in enumerate(oct_files, 1):
    df_temp = pd.read_csv(file)
    oct_dfs.append(df_temp)
    if i % 10 == 0:
        print(f"  Loaded {i}/{len(oct_files)} files...")
    
df_oct = pd.concat(oct_dfs, ignore_index=True)
print(f"October dataset shape: {df_oct.shape}")

# Save merged October dataset
oct_output = 'data/kickstarter_october_merged.csv'
df_oct.to_csv(oct_output, index=False)
print(f"Saved to: {oct_output}")

# Merge November CSVs
print("\n=== Merging November CSVs ===")
nov_path = 'data/Kickstarter_2025-11-12T12_09_07_111Z/'
nov_files = sorted(glob.glob(os.path.join(nov_path, 'Kickstarter*.csv')))

print(f"Found {len(nov_files)} November CSV files")

nov_dfs = []
for i, file in enumerate(nov_files, 1):
    df_temp = pd.read_csv(file)
    nov_dfs.append(df_temp)
    if i % 10 == 0:
        print(f"  Loaded {i}/{len(nov_files)} files...")
    
df_nov = pd.concat(nov_dfs, ignore_index=True)
print(f"November dataset shape: {df_nov.shape}")

# Save merged November dataset
nov_output = 'data/kickstarter_november_merged.csv'
df_nov.to_csv(nov_output, index=False)
print(f"Saved to: {nov_output}")

# Summary
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print("\n=== Merge Complete ===")
print(f"October: {df_oct.shape[0]:,} rows, {df_oct.shape[1]} columns")
print(f"November: {df_nov.shape[0]:,} rows, {df_nov.shape[1]} columns")
print(f"Total time: {duration:.2f} seconds")
print("\nMerged files saved:")
print(f"  - {oct_output}")
print(f"  - {nov_output}")
