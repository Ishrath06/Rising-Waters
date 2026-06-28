import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

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

df['Temp'] = df['Temp'].apply(
    lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x
)

print("\n===== OUTLIERS HANDLED SUCCESSFULLY =====")

# -------------------------
# Handling Categorical Values (EPIC 3 - PART 3)
# -------------------------
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------
# Splitting Data
# -------------------------
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=10
)

# -------------------------
# Feature Scaling
# -------------------------
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("\n===== FEATURE SCALING COMPLETED =====")

# =========================================================
# 🚀 RANDOM FOREST FUNCTION
# =========================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def randomForest(X_train, X_test, y_train, y_test,
                 n_estimators=180,
                 random_state=42):

    print("\n===== RANDOM FOREST MODEL BUILDING =====")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, y_pred

# =========================================================
# 🚀 KNN FUNCTION (FIXED & CLEAN)
# =========================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def KNN(X_train, X_test, y_train, y_test, n_neighbors=5):

    print("\n===== KNN MODEL BUILDING =====")

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)

    return model, y_pred

# -------------------------
# Other Models (EPIC 4 - Part 2)
# -------------------------
from sklearn import tree
from xgboost import XGBClassifier

dtree = tree.DecisionTreeClassifier()
xgb = XGBClassifier()

dtree.fit(X_train, y_train)
xgb.fit(X_train, y_train)

print("\n===== MODELS TRAINED SUCCESSFULLY =====")

# -------------------------
# CALL FUNCTIONS
# -------------------------
rf_model, rf_predictions = randomForest(
    X_train,
    X_test,
    y_train,
    y_test
)

knn_model, knn_predictions = KNN(
    X_train,
    X_test,
    y_train,
    y_test
)