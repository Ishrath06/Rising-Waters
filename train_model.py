import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("dataset/flood dataset.xlsx")

# -------------------------
# Descriptive Analysis
# -------------------------
print("===== FIRST 5 ROWS =====")
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
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=['number']).corr(),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()