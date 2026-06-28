import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("dataset/flood dataset.xlsx")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# -------------------------
# Univariate Analysis
# -------------------------

# Temperature (correct column: Temp)
sns.displot(df['Temp'])
plt.title("Temperature Distribution")
plt.show()

# Humidity
sns.displot(df['Humidity'])
plt.title("Humidity Distribution")
plt.show()

# Annual Rainfall (correct column: ANNUAL)
sns.displot(df['ANNUAL'])
plt.title("Annual Rainfall Distribution")
plt.show()

# Box plot - Temp
sns.boxplot(x=df['Temp'])
plt.title("Temperature Box Plot")
plt.show()

# Box plot - ANNUAL
sns.boxplot(x=df['ANNUAL'])
plt.title("Annual Rainfall Box Plot")
plt.show()