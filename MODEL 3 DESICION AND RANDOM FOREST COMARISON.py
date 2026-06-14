import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\manthan barot\OneDrive\Desktop\WELTECH\MACHINE LEARNING\ML TEST - 13.6.26\ml_datasets\06_random_forest_fraud.csv")

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================
df["Merchant_Category"] = LabelEncoder().fit_transform(df["Merchant_Category"])
df["Country"] = LabelEncoder().fit_transform(df["Country"])

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("Is_Fraud", axis=1)
y = df["Is_Fraud"]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# DECISION TREE
# =========================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt.predict(X_test))

# =========================
# RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

# =========================
# OVERFITTING COMPARISON
# =========================
print("="*50)
print("OVERFITTING COMPARISON")
print("="*50)

print("\nDecision Tree")
print("Train Accuracy :", round(dt_train_acc*100,2), "%")
print("Test Accuracy  :", round(dt_test_acc*100,2), "%")
print("Difference     :", round((dt_train_acc-dt_test_acc)*100,2), "%")

print("\nRandom Forest")
print("Train Accuracy :", round(rf_train_acc*100,2), "%")
print("Test Accuracy  :", round(rf_test_acc*100,2), "%")
print("Difference     :", round((rf_train_acc-rf_test_acc)*100,2), "%")

# =========================
# FEATURE IMPORTANCE
# =========================
dt_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": dt.feature_importances_
})

dt_importance = dt_importance.sort_values(
    by="Importance",
    ascending=False
)

rf_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nDecision Tree Feature Importance")
print(dt_importance)

print("\nRandom Forest Feature Importance")
print(rf_importance)

# =========================
# PLOT DECISION TREE
# =========================
plt.figure(figsize=(8,5))
plt.bar(dt_importance["Feature"],
        dt_importance["Importance"])
plt.title("Decision Tree Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# PLOT RANDOM FOREST
# =========================
plt.figure(figsize=(8,5))
plt.bar(rf_importance["Feature"],
        rf_importance["Importance"])
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()