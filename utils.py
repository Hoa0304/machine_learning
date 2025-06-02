import pandas as pd
import io
import os

temp_file = "real_estate_listings.csv"
df = pd.read_csv(temp_file)

def print_unique_values(df):
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        unique_values = sorted([str(val) for val in unique_values])
        print(f"Unique values in '{column}' ({len(unique_values)} values):")
        if len(unique_values) > 0:
            print(", ".join(unique_values))
        else:
            print("No unique values (all NaN)")
        print("\n")

print_unique_values(df)
