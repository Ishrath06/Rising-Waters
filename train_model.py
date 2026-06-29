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
# Handling Missing Values
# -------------------------
print("===== MISSING VALUES COUNT =====")
print(df.isnull().sum())

print("\n===== ANY MISSING VALUES =====")
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
# Univariate Analysis (NO SAVE - ONLY SHOW)
# -------------------------

sns.histplot(df['Temp'], kde=True)
plt.title("Temperature Distribution")
plt.show()
plt.close()

sns.histplot(df['Humidity'], kde=True)
plt.title("Humidity Distribution")
plt.show()
plt.close()

sns.histplot(df['ANNUAL'], kde=True)
plt.title("Annual Rainfall Distribution")
plt.show()
plt.close()

sns.boxplot(x=df['Temp'])
plt.title("Temperature Box Plot")
plt.show()
plt.close()

sns.boxplot(x=df['ANNUAL'])
plt.title("Annual Rainfall Box Plot")
plt.show()
plt.close()

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
plt.close()

# -------------------------
# Outlier Handling
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
# Encoding Categorical Data
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

# -------------------------
# MODELS
# -------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------
# RANDOM FOREST
# -------------------------
def randomForest(X_train, X_test, y_train, y_test,
                 n_estimators=180,
                 random_state=42):

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nRF Accuracy:", accuracy_score(y_test, y_pred))
    return model, y_pred

# -------------------------
# KNN
# -------------------------
def KNN(X_train, X_test, y_train, y_test, n_neighbors=5):

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nKNN Accuracy:", accuracy_score(y_test, y_pred))
    return model, y_pred

# -------------------------
# XGBOOST
# -------------------------
def XGBoost(X_train, X_test, y_train, y_test):

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nXGB Accuracy:", accuracy_score(y_test, y_pred))
    return model, y_pred

# -------------------------
# DECISION TREE
# -------------------------
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

print("\n===== MODELS TRAINED SUCCESSFULLY =====")

# -------------------------
# CALL MODELS
# -------------------------
rf_model, rf_predictions = randomForest(X_train, X_test, y_train, y_test)
knn_model, knn_predictions = KNN(X_train, X_test, y_train, y_test)
xgb_model, xgb_predictions = XGBoost(X_train, X_test, y_train, y_test)

# -------------------------
# MODEL COMPARISON
# -------------------------
def compareModel(y_test, dtree, rf_pred, knn_pred, xgb_pred):

    dtree_acc = accuracy_score(y_test, dtree.predict(X_test))
    rf_acc = accuracy_score(y_test, rf_pred)
    knn_acc = accuracy_score(y_test, knn_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    print("\nDT:", dtree_acc)
    print("RF:", rf_acc)
    print("KNN:", knn_acc)
    print("XGB:", xgb_acc)

    best_model = max({
        "DT": dtree_acc,
        "RF": rf_acc,
        "KNN": knn_acc,
        "XGB": xgb_acc
    }, key=lambda x: {
        "DT": dtree_acc,
        "RF": rf_acc,
        "KNN": knn_acc,
        "XGB": xgb_acc
    }[x])

    print("\nBEST MODEL:", best_model)
    return best_model

best_model = compareModel(y_test, dtree, rf_predictions, knn_predictions, xgb_predictions)

# -------------------------
# FINAL METRICS
# -------------------------
print("\n===== FINAL METRICS =====")

print("RF:", accuracy_score(y_test, rf_predictions))
print("KNN:", accuracy_score(y_test, knn_predictions))
print("XGB:", accuracy_score(y_test, xgb_predictions))

print("\nPrecision:", precision_score(y_test, xgb_predictions))
print("Recall:", recall_score(y_test, xgb_predictions))