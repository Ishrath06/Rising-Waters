import pandas as pd

# Read the dataset
df = pd.read_excel("dataset/flood dataset.xlsx")

# First 5 rows
print("First 5 Rows:")
print(df.head())

# Shape
print("\nShape of Dataset:")
print(df.shape)

# Column Names
print("\nColumn Names:")
print(df.columns)

# Dataset Information
print("\nDataset Information:")
df.info()

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())