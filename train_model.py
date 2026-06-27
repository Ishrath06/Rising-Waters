import pandas as pd

# Read the dataset
df = pd.read_excel("dataset/flood dataset.xlsx")

# First 5 rows
print(df.head())

# Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns)

# Information about dataset
print("\nDataset Information:")
print(df.info())