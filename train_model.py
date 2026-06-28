import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_excel("dataset/flood dataset.xlsx")

# -------------------------
# Handling Missing Values (EPIC 3 - PART 1)
# -------------------------
print("===== MISSING VALUES COUNT =====")
print(df.isnull().sum())

print("\n===== ANY MISSING VALUES (True/False) =====")
print(df.isnull().any())

print("\n===== TOTAL MISSING VALUES =====")
print(df.isnull().sum().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# -------------------------
# Descriptive Analysis
# -------------------------
print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# -------------------------
# Univariate Analysis
# -------------------------
sns.displot(df['Temp'])
plt.title("Temperature Distribution")
plt.show()

sns.displot(df['Humidity'])
plt.title("Humidity Distribution")
plt.show()

sns.displot(df['ANNUAL'])
plt.title("Annual Rainfall Distribution")
plt.show()

sns.boxplot(x=df['Temp'])
plt.title("Temperature Box Plot")
plt.show()

sns.boxplot(x=df['ANNUAL'])
plt.title("Annual Rainfall Box Plot")
plt.show()

# -------------------------
# Multivariate Analysis
# -------------------------
plt.figure(figsize=(12, 8))

sns.heatmap(
    df.select_dtypes(include=['number']).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# Handling Outliers (EPIC 3 - PART 2)
# -------------------------
Q1 = df['Temp'].quantile(0.25)
Q3 = df['Temp'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n===== OUTLIER BOUNDS =====")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

df['Temp'] = df['Temp'].apply(
    lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
)

print("\n===== OUTLIERS HANDLED SUCCESSFULLY =====")

sns.boxplot(x=df['Temp'])
plt.title("Temperature After Outlier Handling")
plt.show()

# -------------------------
# Handling Categorical Values (EPIC 3 - PART 3)
# -------------------------

print("\n===== CATEGORICAL COLUMNS BEFORE ENCODING =====")
print(df.select_dtypes(include=['object']).columns)

# Label Encoding for categorical columns
le = LabelEncoder()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("\n===== CATEGORICAL ENCODING COMPLETED =====")
print(df.head())